use alloc::{boxed::Box, vec::Vec};
use core::fmt;

use crate::control::dense_ops::{
    clone_mat, column_vector_norm, dense_add, dense_mul, dense_sub, hermitian_project_in_place,
};
use crate::control::estimation::CovarianceUpdate;
use crate::control::estimation::dense::{default_tolerance, solve_right_checked};
use crate::control::estimation::nonlinear_core::{
    normalized_innovation_norm, updated_covariance_ukf, validate_unscented_params,
    weighted_covariance, weighted_cross_covariance, weighted_mean,
};
use crate::decomp::{DenseDecompParams, dense_self_adjoint_eigen};
use crate::sparse::compensated::CompensatedField;
use faer::{Mat, MatRef};
use faer_traits::RealField;
use faer_traits::ext::ComplexFieldExt;
use num_traits::{Float, NumCast};

use super::{
    DiscreteNonlinearModel, NonlinearEstimatorError, NonlinearKalmanPrediction,
    NonlinearKalmanUpdate, SigmaPointProvider, SigmaPointSet, SigmaPointStrategy, UkfStage,
    UnscentedParams, validate_column_vector, validate_finite, validate_model_output,
    validate_nonlinear_filter_model, validate_prediction,
};

impl<R> SigmaPointStrategy<R> {
    /// Resolves the sigma-point set for one UKF stage.
    pub(super) fn sigma_points(
        &self,
        mean: MatRef<'_, R>,
        covariance: MatRef<'_, R>,
        input: MatRef<'_, R>,
        stage: UkfStage,
    ) -> Result<SigmaPointSet<R>, NonlinearEstimatorError>
    where
        R: CompensatedField + RealField,
        R::Real: Float + Copy,
    {
        match self {
            Self::Standard(params) => standard_sigma_points(mean, covariance, *params),
            Self::Custom(provider) => provider.sigma_points(mean, covariance, input, stage),
        }
    }
}

impl<R: fmt::Debug> fmt::Debug for SigmaPointStrategy<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard(params) => f.debug_tuple("Standard").field(params).finish(),
            Self::Custom(_) => f.write_str("Custom(<sigma-point-provider>)"),
        }
    }
}

/// Discrete-time unscented Kalman filter.
#[derive(Debug)]
pub struct UnscentedKalmanFilter<M, R>
where
    M: DiscreteNonlinearModel<R>,
    R: CompensatedField,
    R::Real: Float + Copy,
{
    /// User-supplied nonlinear model.
    pub model: M,
    /// Process-noise covariance in state coordinates.
    pub q: Mat<R>,
    /// Measurement-noise covariance in measurement coordinates.
    pub r: Mat<R>,
    /// Covariance-update policy used during measurement incorporation.
    pub covariance_update: CovarianceUpdate,
    /// Current posterior state estimate.
    pub x_hat: Mat<R>,
    /// Current posterior covariance.
    pub p: Mat<R>,
    pub(super) sigma_strategy: SigmaPointStrategy<R>,
}

impl<M, R> UnscentedKalmanFilter<M, R>
where
    M: DiscreteNonlinearModel<R>,
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    /// Builds a UKF using the standard scaled unscented transform.
    pub fn new_standard(
        model: M,
        q: Mat<R>,
        r: Mat<R>,
        x_hat: Mat<R>,
        p: Mat<R>,
        params: UnscentedParams<R>,
    ) -> Result<Self, NonlinearEstimatorError> {
        Self::new_standard_with_covariance_update(
            model,
            q,
            r,
            x_hat,
            p,
            params,
            CovarianceUpdate::default(),
        )
    }

    /// Builds a UKF using the standard scaled unscented transform and an
    /// explicit covariance-update policy.
    pub fn new_standard_with_covariance_update(
        model: M,
        q: Mat<R>,
        r: Mat<R>,
        x_hat: Mat<R>,
        p: Mat<R>,
        params: UnscentedParams<R>,
        covariance_update: CovarianceUpdate,
    ) -> Result<Self, NonlinearEstimatorError> {
        validate_unscented_params(params, model.nstates())?;
        Self::new_with_strategy(
            model,
            q,
            r,
            x_hat,
            p,
            SigmaPointStrategy::Standard(params),
            covariance_update,
        )
    }

    /// Builds a UKF using a custom sigma-point provider.
    pub fn new_custom<P>(
        model: M,
        q: Mat<R>,
        r: Mat<R>,
        x_hat: Mat<R>,
        p: Mat<R>,
        provider: P,
    ) -> Result<Self, NonlinearEstimatorError>
    where
        P: SigmaPointProvider<R> + 'static,
    {
        Self::new_custom_with_covariance_update(
            model,
            q,
            r,
            x_hat,
            p,
            provider,
            CovarianceUpdate::default(),
        )
    }

    /// Builds a UKF using a custom sigma-point provider and an explicit
    /// covariance-update policy.
    pub fn new_custom_with_covariance_update<P>(
        model: M,
        q: Mat<R>,
        r: Mat<R>,
        x_hat: Mat<R>,
        p: Mat<R>,
        provider: P,
        covariance_update: CovarianceUpdate,
    ) -> Result<Self, NonlinearEstimatorError>
    where
        P: SigmaPointProvider<R> + 'static,
    {
        Self::new_with_strategy(
            model,
            q,
            r,
            x_hat,
            p,
            SigmaPointStrategy::Custom(Box::new(provider)),
            covariance_update,
        )
    }

    fn new_with_strategy(
        model: M,
        q: Mat<R>,
        r: Mat<R>,
        x_hat: Mat<R>,
        p: Mat<R>,
        sigma_strategy: SigmaPointStrategy<R>,
        covariance_update: CovarianceUpdate,
    ) -> Result<Self, NonlinearEstimatorError> {
        validate_nonlinear_filter_model(
            model.nstates(),
            model.ninputs(),
            model.noutputs(),
            q.as_ref(),
            r.as_ref(),
            x_hat.as_ref(),
            p.as_ref(),
        )?;
        Ok(Self {
            model,
            q,
            r,
            covariance_update,
            x_hat,
            p,
            sigma_strategy,
        })
    }

    /// Returns the current posterior state estimate.
    #[must_use]
    pub fn state_estimate(&self) -> MatRef<'_, R> {
        self.x_hat.as_ref()
    }

    /// Returns the current posterior covariance.
    #[must_use]
    pub fn covariance(&self) -> MatRef<'_, R> {
        self.p.as_ref()
    }

    /// Returns the configured sigma-point strategy.
    #[must_use]
    pub fn sigma_strategy(&self) -> &SigmaPointStrategy<R> {
        &self.sigma_strategy
    }

    /// Computes the UKF prediction stage without mutating the filter state.
    pub fn predict(
        &self,
        input: MatRef<'_, R>,
    ) -> Result<NonlinearKalmanPrediction<R>, NonlinearEstimatorError> {
        validate_column_vector("input", input, self.model.ninputs())?;
        let sigma = self.sigma_strategy.sigma_points(
            self.x_hat.as_ref(),
            self.p.as_ref(),
            input,
            UkfStage::Predict,
        )?;
        validate_sigma_point_set(&sigma, self.model.nstates())?;

        let propagated = propagate_sigma_points(
            sigma.points.as_ref(),
            |point| self.model.transition(point, input),
            self.model.nstates(),
            "transition",
        )?;
        let state = weighted_mean(propagated.as_ref(), &sigma.mean_weights);
        let covariance = dense_add(
            weighted_covariance(propagated.as_ref(), state.as_ref(), &sigma.cov_weights).as_ref(),
            self.q.as_ref(),
        );
        let mut covariance = covariance;
        hermitian_project_in_place(&mut covariance);
        let measurement_sigma = self.sigma_strategy.sigma_points(
            state.as_ref(),
            covariance.as_ref(),
            input,
            UkfStage::Update,
        )?;
        validate_sigma_point_set(&measurement_sigma, self.model.nstates())?;

        let output_points = propagate_sigma_points(
            measurement_sigma.points.as_ref(),
            |point| self.model.output(point, input),
            self.model.noutputs(),
            "output",
        )?;
        let output = weighted_mean(output_points.as_ref(), &measurement_sigma.mean_weights);

        validate_finite("prediction.state", state.as_ref())?;
        validate_finite("prediction.covariance", covariance.as_ref())?;
        validate_finite("prediction.output", output.as_ref())?;

        Ok(NonlinearKalmanPrediction {
            state,
            covariance,
            output,
        })
    }

    /// Applies one UKF measurement update to an externally supplied prediction.
    pub fn update(
        &self,
        prediction: &NonlinearKalmanPrediction<R>,
        input: MatRef<'_, R>,
        measurement: MatRef<'_, R>,
    ) -> Result<NonlinearKalmanUpdate<R>, NonlinearEstimatorError> {
        validate_prediction(
            prediction,
            self.model.nstates(),
            self.model.noutputs(),
            "prediction",
        )?;
        validate_column_vector("input", input, self.model.ninputs())?;
        validate_column_vector("measurement", measurement, self.model.noutputs())?;

        let sigma = self.sigma_strategy.sigma_points(
            prediction.state.as_ref(),
            prediction.covariance.as_ref(),
            input,
            UkfStage::Update,
        )?;
        validate_sigma_point_set(&sigma, self.model.nstates())?;

        let output_points = propagate_sigma_points(
            sigma.points.as_ref(),
            |point| self.model.output(point, input),
            self.model.noutputs(),
            "output",
        )?;
        let predicted_output = weighted_mean(output_points.as_ref(), &sigma.mean_weights);
        let innovation = dense_sub(measurement, predicted_output.as_ref());
        let innovation_covariance = dense_add(
            weighted_covariance(
                output_points.as_ref(),
                predicted_output.as_ref(),
                &sigma.cov_weights,
            )
            .as_ref(),
            self.r.as_ref(),
        );
        let mut innovation_covariance = innovation_covariance;
        hermitian_project_in_place(&mut innovation_covariance);
        let cross = weighted_cross_covariance(
            sigma.points.as_ref(),
            prediction.state.as_ref(),
            output_points.as_ref(),
            predicted_output.as_ref(),
            &sigma.cov_weights,
        );
        let gain = solve_right_checked(
            cross.as_ref(),
            innovation_covariance.as_ref(),
            default_tolerance::<R>(),
            || NonlinearEstimatorError::SingularInnovationCovariance,
        )?;
        let state = dense_add(
            prediction.state.as_ref(),
            dense_mul(gain.as_ref(), innovation.as_ref()).as_ref(),
        );
        let covariance = updated_covariance_ukf(
            self.covariance_update,
            prediction.covariance.as_ref(),
            gain.as_ref(),
            cross.as_ref(),
            self.r.as_ref(),
            innovation_covariance.as_ref(),
        )?;
        let mut covariance = covariance;
        hermitian_project_in_place(&mut covariance);
        let output = self.model.output(state.as_ref(), input);
        validate_model_output("output", output.as_ref(), self.model.noutputs(), 1)?;

        validate_finite("update.innovation", innovation.as_ref())?;
        validate_finite("update.gain", gain.as_ref())?;
        validate_finite("update.state", state.as_ref())?;
        validate_finite("update.covariance", covariance.as_ref())?;
        validate_finite("update.output", output.as_ref())?;

        Ok(NonlinearKalmanUpdate {
            innovation_norm: column_vector_norm(innovation.as_ref()),
            normalized_innovation_norm: normalized_innovation_norm(
                innovation.as_ref(),
                innovation_covariance.as_ref(),
            )?,
            innovation,
            innovation_covariance,
            predicted_output,
            gain,
            state,
            covariance,
            output,
        })
    }

    /// Runs one full UKF predict/update cycle and stores the posterior state.
    pub fn step(
        &mut self,
        input: MatRef<'_, R>,
        measurement: MatRef<'_, R>,
    ) -> Result<NonlinearKalmanUpdate<R>, NonlinearEstimatorError> {
        let prediction = self.predict(input)?;
        let update = self.update(&prediction, input, measurement)?;
        self.x_hat = clone_mat(update.state.as_ref());
        self.p = clone_mat(update.covariance.as_ref());
        Ok(update)
    }
}

/// Validates an expert-supplied sigma-point set before the UKF consumes it.
fn validate_sigma_point_set<R>(
    sigma: &SigmaPointSet<R>,
    nstates: usize,
) -> Result<(), NonlinearEstimatorError>
where
    R: Float + Copy + NumCast + RealField,
{
    if sigma.points.nrows() != nstates {
        return Err(NonlinearEstimatorError::InvalidSigmaPointSet {
            which: "points.nrows",
        });
    }
    if sigma.points.ncols() == 0 {
        return Err(NonlinearEstimatorError::InvalidSigmaPointSet {
            which: "points.ncols",
        });
    }
    if sigma.mean_weights.len() != sigma.points.ncols() {
        return Err(NonlinearEstimatorError::InvalidSigmaPointSet {
            which: "mean_weights.len",
        });
    }
    if sigma.cov_weights.len() != sigma.points.ncols() {
        return Err(NonlinearEstimatorError::InvalidSigmaPointSet {
            which: "cov_weights.len",
        });
    }
    if !sigma.points.as_ref().is_all_finite()
        || sigma.mean_weights.iter().any(|w| !w.is_finite())
        || sigma.cov_weights.iter().any(|w| !w.is_finite())
    {
        return Err(NonlinearEstimatorError::InvalidSigmaPointSet {
            which: "non_finite",
        });
    }
    let mut sum = R::zero();
    for &weight in &sigma.mean_weights {
        sum = sum + weight;
    }
    let scale: R = NumCast::from(sigma.points.ncols().max(1)).unwrap();
    let tol = R::epsilon().sqrt() * scale;
    if (sum - R::one()).abs() > tol {
        return Err(NonlinearEstimatorError::InvalidSigmaPointSet {
            which: "mean_weights.sum",
        });
    }
    Ok(())
}

/// Builds the standard scaled sigma-point set from a mean/covariance pair.
fn standard_sigma_points<R>(
    mean: MatRef<'_, R>,
    covariance: MatRef<'_, R>,
    params: UnscentedParams<R>,
) -> Result<SigmaPointSet<R>, NonlinearEstimatorError>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    super::validate_column_vector("mean", mean, covariance.nrows())?;
    super::validate_square("covariance", covariance, mean.nrows())?;
    validate_unscented_params(params, mean.nrows())?;

    let n: R = NumCast::from(mean.nrows()).unwrap();
    let lambda = params.alpha * params.alpha * (n + params.kappa) - n;
    let scaling = n + lambda;
    if !scaling.is_finite() || scaling <= R::zero() {
        return Err(NonlinearEstimatorError::InvalidUnscentedParams {
            which: "n_plus_lambda",
        });
    }

    let mut covariance = clone_mat(covariance);
    hermitian_project_in_place(&mut covariance);
    let eig = dense_self_adjoint_eigen(covariance.as_ref(), &DenseDecompParams::<R>::new())
        .map_err(|_| NonlinearEstimatorError::NonPositiveDefiniteCovariance {
            which: "sigma_points.covariance",
        })?;
    let mut max_abs_eigenvalue = R::zero();
    for idx in 0..eig.values.nrows() {
        max_abs_eigenvalue = max_abs_eigenvalue.max(eig.values[idx].abs());
    }
    let tol = R::epsilon().sqrt() * R::one().max(max_abs_eigenvalue);
    let mut sqrt_eigenvalues = Vec::with_capacity(eig.values.nrows());
    for idx in 0..eig.values.nrows() {
        let value = eig.values[idx];
        if value < -tol {
            return Err(NonlinearEstimatorError::NonPositiveDefiniteCovariance {
                which: "sigma_points.covariance",
            });
        }
        sqrt_eigenvalues.push(value.max(R::zero()).sqrt());
    }

    let gamma = scaling.sqrt();
    let nstates = mean.nrows();
    let npoints = 2 * nstates + 1;
    let points = Mat::from_fn(nstates, npoints, |row, col| {
        if col == 0 {
            mean[(row, 0)]
        } else if col <= nstates {
            let idx = col - 1;
            mean[(row, 0)] + eig.vectors[(row, idx)] * sqrt_eigenvalues[idx] * gamma
        } else {
            let idx = col - nstates - 1;
            mean[(row, 0)] - eig.vectors[(row, idx)] * sqrt_eigenvalues[idx] * gamma
        }
    });

    let mut mean_weights = vec![R::zero(); npoints];
    let mut cov_weights = vec![R::zero(); npoints];
    mean_weights[0] = lambda / scaling;
    cov_weights[0] = mean_weights[0] + (R::one() - params.alpha * params.alpha + params.beta);
    let tail_weight = (R::one() + R::one()).recip() / scaling;
    for i in 1..npoints {
        mean_weights[i] = tail_weight;
        cov_weights[i] = tail_weight;
    }

    Ok(SigmaPointSet {
        points,
        mean_weights,
        cov_weights,
    })
}

/// Propagates a sigma-point matrix through one nonlinear callback.
fn propagate_sigma_points<R, F>(
    points: MatRef<'_, R>,
    mut map: F,
    expected_nrows: usize,
    which: &'static str,
) -> Result<Mat<R>, NonlinearEstimatorError>
where
    R: CompensatedField,
    R::Real: Float + Copy,
    F: FnMut(MatRef<'_, R>) -> Mat<R>,
{
    let npoints = points.ncols();
    let first = map(points.subcols(0, 1));
    super::validate_model_output(which, first.as_ref(), expected_nrows, 1)?;
    super::validate_finite(which, first.as_ref())?;
    let mut out = Mat::zeros(expected_nrows, npoints);
    out.as_mut().subcols_mut(0, 1).copy_from(first.as_ref());
    for idx in 1..npoints {
        let value = map(points.subcols(idx, 1));
        super::validate_model_output(which, value.as_ref(), expected_nrows, 1)?;
        super::validate_finite(which, value.as_ref())?;
        out.as_mut().subcols_mut(idx, 1).copy_from(value.as_ref());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::UnscentedKalmanFilter;
    use crate::control::estimation::DiscreteKalmanFilter;
    use crate::control::estimation::nonlinear::ExtendedKalmanFilter;
    use crate::control::estimation::nonlinear::{
        DiscreteExtendedKalmanModel, DiscreteNonlinearModel, NonlinearEstimatorError,
        SigmaPointProvider, SigmaPointSet, UkfStage, UnscentedParams,
    };
    use faer::Mat;
    use std::cell::Cell;
    use std::rc::Rc;

    #[derive(Clone, Copy, Debug)]
    struct QuadraticModel;

    impl DiscreteNonlinearModel<f64> for QuadraticModel {
        fn nstates(&self) -> usize {
            1
        }
        fn ninputs(&self) -> usize {
            1
        }
        fn noutputs(&self) -> usize {
            1
        }
        fn transition(&self, x: faer::MatRef<'_, f64>, u: faer::MatRef<'_, f64>) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| x[(0, 0)] * x[(0, 0)] + u[(0, 0)])
        }
        fn output(&self, x: faer::MatRef<'_, f64>, _u: faer::MatRef<'_, f64>) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| x[(0, 0)] * x[(0, 0)])
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct NonlinearOutputModel;

    impl DiscreteNonlinearModel<f64> for NonlinearOutputModel {
        fn nstates(&self) -> usize {
            1
        }
        fn ninputs(&self) -> usize {
            1
        }
        fn noutputs(&self) -> usize {
            1
        }
        fn transition(&self, x: faer::MatRef<'_, f64>, u: faer::MatRef<'_, f64>) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| x[(0, 0)] * x[(0, 0)] + u[(0, 0)])
        }
        fn output(&self, x: faer::MatRef<'_, f64>, _u: faer::MatRef<'_, f64>) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| x[(0, 0)])
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct LinearScalarModel {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
    }

    impl DiscreteNonlinearModel<f64> for LinearScalarModel {
        fn nstates(&self) -> usize {
            1
        }
        fn ninputs(&self) -> usize {
            1
        }
        fn noutputs(&self) -> usize {
            1
        }
        fn transition(&self, x: faer::MatRef<'_, f64>, u: faer::MatRef<'_, f64>) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| self.a * x[(0, 0)] + self.b * u[(0, 0)])
        }
        fn output(&self, x: faer::MatRef<'_, f64>, u: faer::MatRef<'_, f64>) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| self.c * x[(0, 0)] + self.d * u[(0, 0)])
        }
    }

    impl DiscreteExtendedKalmanModel<f64> for LinearScalarModel {
        fn transition_jacobian(
            &self,
            _x: faer::MatRef<'_, f64>,
            _u: faer::MatRef<'_, f64>,
        ) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| self.a)
        }
        fn output_jacobian(
            &self,
            _x: faer::MatRef<'_, f64>,
            _u: faer::MatRef<'_, f64>,
        ) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| self.c)
        }
    }

    struct CountingProvider {
        predict_calls: Rc<Cell<usize>>,
        update_calls: Rc<Cell<usize>>,
    }

    impl SigmaPointProvider<f64> for CountingProvider {
        fn sigma_points(
            &self,
            mean: faer::MatRef<'_, f64>,
            _covariance: faer::MatRef<'_, f64>,
            _input: faer::MatRef<'_, f64>,
            stage: UkfStage,
        ) -> Result<SigmaPointSet<f64>, NonlinearEstimatorError> {
            match stage {
                UkfStage::Predict => self.predict_calls.set(self.predict_calls.get() + 1),
                UkfStage::Update => self.update_calls.set(self.update_calls.get() + 1),
            }
            let x = mean[(0, 0)];
            Ok(SigmaPointSet {
                points: Mat::from_fn(1, 3, |_, col| match col {
                    0 => x,
                    1 => x + 0.25,
                    _ => x - 0.25,
                }),
                mean_weights: vec![0.0, 0.5, 0.5],
                cov_weights: vec![2.0, 0.5, 0.5],
            })
        }
    }

    struct BadProvider;

    impl SigmaPointProvider<f64> for BadProvider {
        fn sigma_points(
            &self,
            mean: faer::MatRef<'_, f64>,
            _covariance: faer::MatRef<'_, f64>,
            _input: faer::MatRef<'_, f64>,
            _stage: UkfStage,
        ) -> Result<SigmaPointSet<f64>, NonlinearEstimatorError> {
            Ok(SigmaPointSet {
                points: Mat::from_fn(1, 2, |_, col| if col == 0 { mean[(0, 0)] } else { 2.0 }),
                mean_weights: vec![0.2, 0.2],
                cov_weights: vec![0.5, 0.5],
            })
        }
    }

    struct InputAwareUpdateProvider;

    impl SigmaPointProvider<f64> for InputAwareUpdateProvider {
        fn sigma_points(
            &self,
            mean: faer::MatRef<'_, f64>,
            _covariance: faer::MatRef<'_, f64>,
            input: faer::MatRef<'_, f64>,
            stage: UkfStage,
        ) -> Result<SigmaPointSet<f64>, NonlinearEstimatorError> {
            let x = mean[(0, 0)];
            let delta = match stage {
                UkfStage::Predict => 0.25,
                UkfStage::Update => 0.25 + input[(0, 0)],
            };
            Ok(SigmaPointSet {
                points: Mat::from_fn(1, 3, |_, col| match col {
                    0 => x,
                    1 => x + delta,
                    _ => x - delta,
                }),
                mean_weights: vec![0.0, 0.5, 0.5],
                cov_weights: vec![0.0, 0.5, 0.5],
            })
        }
    }

    fn assert_close(lhs: f64, rhs: f64, tol: f64) {
        assert!(
            (lhs - rhs).abs() <= tol,
            "lhs={lhs:?}, rhs={rhs:?}, tol={tol:?}"
        );
    }

    #[test]
    fn ukf_standard_predict_matches_scalar_hand_calculation() {
        let q = Mat::from_fn(1, 1, |_, _| 0.1);
        let r = Mat::from_fn(1, 1, |_, _| 0.2);
        let x_hat = Mat::from_fn(1, 1, |_, _| 1.0);
        let p = Mat::from_fn(1, 1, |_, _| 0.25);
        let ukf = UnscentedKalmanFilter::new_standard(
            NonlinearOutputModel,
            q,
            r,
            x_hat,
            p,
            UnscentedParams {
                alpha: 1.0,
                beta: 2.0,
                kappa: 0.0,
            },
        )
        .unwrap();
        let u = Mat::from_fn(1, 1, |_, _| 0.0);
        let prediction = ukf.predict(u.as_ref()).unwrap();
        assert_close(prediction.state[(0, 0)], 1.25, 1.0e-12);
        assert_close(prediction.covariance[(0, 0)], 1.225, 1.0e-12);
        assert_close(prediction.output[(0, 0)], 1.25, 1.0e-12);
    }

    #[test]
    fn ukf_predict_returns_unscented_measurement_mean_for_nonlinear_output() {
        let q = Mat::from_fn(1, 1, |_, _| 0.0);
        let r = Mat::from_fn(1, 1, |_, _| 0.2);
        let x_hat = Mat::from_fn(1, 1, |_, _| 1.0);
        let p = Mat::from_fn(1, 1, |_, _| 0.25);
        let ukf = UnscentedKalmanFilter::new_standard(
            QuadraticModel,
            q,
            r,
            x_hat,
            p,
            UnscentedParams {
                alpha: 1.0,
                beta: 2.0,
                kappa: 0.0,
            },
        )
        .unwrap();
        let u = Mat::from_fn(1, 1, |_, _| 0.0);

        let prediction = ukf.predict(u.as_ref()).unwrap();
        assert_close(prediction.state[(0, 0)], 1.25, 1.0e-12);
        assert_close(prediction.output[(0, 0)], 2.6875, 1.0e-12);
    }

    #[test]
    fn ukf_update_rebuilds_measurement_statistics_from_predicted_pair() {
        let q = Mat::from_fn(1, 1, |_, _| 0.5);
        let r = Mat::from_fn(1, 1, |_, _| 0.2);
        let x_hat = Mat::from_fn(1, 1, |_, _| 1.0);
        let p = Mat::from_fn(1, 1, |_, _| 0.25);
        let ukf = UnscentedKalmanFilter::new_standard(
            QuadraticModel,
            q,
            r,
            x_hat,
            p,
            UnscentedParams {
                alpha: 1.0,
                beta: 2.0,
                kappa: 0.0,
            },
        )
        .unwrap();
        let u = Mat::from_fn(1, 1, |_, _| 0.0);
        let y = Mat::from_fn(1, 1, |_, _| 2.4);

        let prediction = ukf.predict(u.as_ref()).unwrap();
        let update = ukf.update(&prediction, u.as_ref(), y.as_ref()).unwrap();
        assert_close(prediction.output[(0, 0)], 3.1875, 1.0e-12);
        assert_close(update.predicted_output[(0, 0)], 3.1875, 1.0e-12);
    }

    #[test]
    fn ukf_custom_sigma_points_are_used() {
        let predict_calls = Rc::new(Cell::new(0));
        let update_calls = Rc::new(Cell::new(0));
        let provider = CountingProvider {
            predict_calls: predict_calls.clone(),
            update_calls: update_calls.clone(),
        };
        let q = Mat::from_fn(1, 1, |_, _| 0.0);
        let r = Mat::from_fn(1, 1, |_, _| 0.1);
        let x_hat = Mat::from_fn(1, 1, |_, _| 1.0);
        let p = Mat::from_fn(1, 1, |_, _| 0.25);
        let mut ukf =
            UnscentedKalmanFilter::new_custom(NonlinearOutputModel, q, r, x_hat, p, provider)
                .unwrap();
        let u = Mat::from_fn(1, 1, |_, _| 0.0);
        let y = Mat::from_fn(1, 1, |_, _| 1.0);
        let update = ukf.step(u.as_ref(), y.as_ref()).unwrap();
        assert_eq!(predict_calls.get(), 1);
        assert_eq!(update_calls.get(), 2);
        assert_close(update.predicted_output[(0, 0)], 1.0625, 1.0e-12);
    }

    #[test]
    fn ukf_split_update_rebuilds_custom_update_sigma_for_update_input() {
        let q = Mat::from_fn(1, 1, |_, _| 0.0);
        let r = Mat::from_fn(1, 1, |_, _| 0.25);
        let x_hat = Mat::from_fn(1, 1, |_, _| 1.0);
        let p = Mat::from_fn(1, 1, |_, _| 0.5);
        let ukf_split = UnscentedKalmanFilter::new_custom(
            LinearScalarModel {
                a: 1.0,
                b: 0.0,
                c: 1.0,
                d: 0.0,
            },
            q.clone(),
            r.clone(),
            x_hat.clone(),
            p.clone(),
            InputAwareUpdateProvider,
        )
        .unwrap();
        let mut ukf_step = UnscentedKalmanFilter::new_custom(
            LinearScalarModel {
                a: 1.0,
                b: 0.0,
                c: 1.0,
                d: 0.0,
            },
            q,
            r,
            x_hat,
            p,
            InputAwareUpdateProvider,
        )
        .unwrap();

        let predict_input = Mat::from_fn(1, 1, |_, _| 0.0);
        let update_input = Mat::from_fn(1, 1, |_, _| 1.0);
        let measurement = Mat::from_fn(1, 1, |_, _| 2.0);

        let prediction = ukf_split.predict(predict_input.as_ref()).unwrap();
        let split_update = ukf_split
            .update(&prediction, update_input.as_ref(), measurement.as_ref())
            .unwrap();
        let step_update = ukf_step
            .step(update_input.as_ref(), measurement.as_ref())
            .unwrap();

        assert_close(
            split_update.predicted_output[(0, 0)],
            step_update.predicted_output[(0, 0)],
            1.0e-12,
        );
        assert_close(split_update.gain[(0, 0)], step_update.gain[(0, 0)], 1.0e-12);
        assert_close(
            split_update.state[(0, 0)],
            step_update.state[(0, 0)],
            1.0e-12,
        );
        assert_close(
            split_update.covariance[(0, 0)],
            step_update.covariance[(0, 0)],
            1.0e-12,
        );
    }

    #[test]
    fn ukf_rejects_invalid_custom_sigma_points() {
        let q = Mat::from_fn(1, 1, |_, _| 0.0);
        let r = Mat::from_fn(1, 1, |_, _| 0.1);
        let x_hat = Mat::from_fn(1, 1, |_, _| 1.0);
        let p = Mat::from_fn(1, 1, |_, _| 0.25);
        let ukf =
            UnscentedKalmanFilter::new_custom(NonlinearOutputModel, q, r, x_hat, p, BadProvider)
                .unwrap();
        let u = Mat::from_fn(1, 1, |_, _| 0.0);
        let err = ukf.predict(u.as_ref()).unwrap_err();
        assert!(matches!(
            err,
            NonlinearEstimatorError::InvalidSigmaPointSet {
                which: "mean_weights.sum"
            }
        ));
    }

    #[test]
    fn ukf_accepts_semidefinite_covariance() {
        let q = Mat::from_fn(1, 1, |_, _| 0.1);
        let r = Mat::from_fn(1, 1, |_, _| 0.2);
        let x_hat = Mat::from_fn(1, 1, |_, _| 1.0);
        let p = Mat::from_fn(1, 1, |_, _| 0.0);
        let ukf = UnscentedKalmanFilter::new_standard(
            NonlinearOutputModel,
            q,
            r,
            x_hat,
            p,
            UnscentedParams {
                alpha: 1.0,
                beta: 2.0,
                kappa: 0.0,
            },
        )
        .unwrap();
        let u = Mat::from_fn(1, 1, |_, _| 0.0);

        let prediction = ukf.predict(u.as_ref()).unwrap();
        assert_close(prediction.state[(0, 0)], 1.0, 1.0e-12);
        assert_close(prediction.covariance[(0, 0)], 0.1, 1.0e-12);
        assert_close(prediction.output[(0, 0)], 1.0, 1.0e-12);
    }

    #[test]
    fn nonlinear_filters_track_linear_kalman_on_linear_model() {
        let model = LinearScalarModel {
            a: 1.1,
            b: 0.4,
            c: 0.8,
            d: 0.1,
        };
        let q = Mat::from_fn(1, 1, |_, _| 0.2);
        let r = Mat::from_fn(1, 1, |_, _| 0.3);
        let x_hat = Mat::from_fn(1, 1, |_, _| 0.5);
        let p = Mat::from_fn(1, 1, |_, _| 1.1);
        let u = Mat::from_fn(1, 1, |_, _| 0.25);
        let y = Mat::from_fn(1, 1, |_, _| 0.9);

        let mut linear = DiscreteKalmanFilter::new(
            Mat::from_fn(1, 1, |_, _| model.a),
            Mat::from_fn(1, 1, |_, _| model.b),
            Mat::from_fn(1, 1, |_, _| model.c),
            Mat::from_fn(1, 1, |_, _| model.d),
            q.clone(),
            r.clone(),
            x_hat.clone(),
            p.clone(),
        )
        .unwrap();
        let mut ekf =
            ExtendedKalmanFilter::new(model, q.clone(), r.clone(), x_hat.clone(), p.clone())
                .unwrap();
        let mut ukf = UnscentedKalmanFilter::new_standard(
            model,
            q,
            r,
            x_hat,
            p,
            UnscentedParams {
                alpha: 1.0,
                beta: 2.0,
                kappa: 0.0,
            },
        )
        .unwrap();

        let linear_update = linear.step(u.as_ref(), y.as_ref()).unwrap();
        let ekf_update = ekf.step(u.as_ref(), y.as_ref()).unwrap();
        let ukf_update = ukf.step(u.as_ref(), y.as_ref()).unwrap();

        assert_close(
            linear_update.state[(0, 0)],
            ekf_update.state[(0, 0)],
            1.0e-10,
        );
        assert_close(
            linear_update.covariance[(0, 0)],
            ekf_update.covariance[(0, 0)],
            1.0e-10,
        );
        assert_close(
            linear_update.state[(0, 0)],
            ukf_update.state[(0, 0)],
            1.0e-10,
        );
        assert_close(
            linear_update.covariance[(0, 0)],
            ukf_update.covariance[(0, 0)],
            1.0e-10,
        );
    }
}
