//! Discrete-time nonlinear state estimation with EKF and UKF runtimes.
//!
//! This module deliberately mirrors the structure of [`super::linear`]
//! where practical:
//!
//! - explicit `predict` / `update` / `step` stages
//! - innovation and normalized-innovation diagnostics
//! - configurable covariance-update policy through
//!   [`super::CovarianceUpdate`]
//!
//! The implementation is restricted to additive-noise discrete-time
//! models:
//!
//! - `x[k+1] = f(x[k], u[k]) + w[k]`
//! - `y[k] = h(x[k], u[k]) + v[k]`
//!
//! with `Q` interpreted directly in state coordinates and `R` interpreted
//! directly in measurement coordinates.
//!
//! # Two Intuitions
//!
//! 1. **Local-linear view.** EKF says "pretend the nonlinear model is linear
//!    right here" and then run a Kalman-style update on that local model.
//! 2. **Deterministic-sampling view.** UKF says "sample the uncertainty
//!    directly, push those points through the nonlinear model, then rebuild the
//!    mean and covariance from the transformed cloud."
//!
//! # Glossary
//!
//! - **Jacobian:** Local derivative used by EKF linearization.
//! - **Sigma points:** Deterministic support points used by the UKF.
//! - **Cross covariance:** Covariance between predicted state and predicted
//!   measurement, used to form the UKF gain.
//! - **Additive-noise model:** Process and measurement noise enter by
//!   covariance addition rather than through separate noise-input maps.
//!
//! # Mathematical Formulation
//!
//! The shared nonlinear model form is:
//!
//! - `x[k+1] = f(x[k], u[k]) + w[k]`
//! - `y[k] = h(x[k], u[k]) + v[k]`
//!
//! EKF uses Jacobians `F = df/dx`, `H = dh/dx`. UKF reconstructs means and
//! covariances from propagated weighted sigma points.
//!
//! # Implementation Notes
//!
//! - The first nonlinear surface is discrete-time only.
//! - UKF sigma-point placement is pluggable so callers can provide custom
//!   weighted point sets around discontinuities or hybrid boundaries.
//! - The same covariance-update policy enum as the linear layer is reused so
//!   Joseph-form and simpler updates stay aligned across estimator families.

use super::CovarianceUpdate;
use crate::decomp::{DenseDecompParams, dense_self_adjoint_eigen};
use crate::sparse::compensated::{CompensatedField, CompensatedSum};
use core::fmt;
use faer::prelude::Solve;
use faer::{Mat, MatRef};
use faer_traits::ext::ComplexFieldExt;
use faer_traits::{ComplexField, RealField};
use num_traits::{Float, NumCast, One, Zero};

/// Discrete nonlinear state/output model used by EKF and UKF.
///
/// The model object owns any user parameters, cached data, or externally
/// updated operating-condition metadata. The estimator runtime only assumes the
/// model can evaluate the state transition and measurement maps.
pub trait DiscreteNonlinearModel<R> {
    /// Number of state variables.
    fn nstates(&self) -> usize;
    /// Number of control inputs.
    fn ninputs(&self) -> usize;
    /// Number of measured outputs.
    fn noutputs(&self) -> usize;

    /// Evaluates the discrete-time transition map `f(x, u)`.
    fn transition(&self, x: MatRef<'_, R>, u: MatRef<'_, R>) -> Mat<R>;
    /// Evaluates the measurement map `h(x, u)`.
    fn output(&self, x: MatRef<'_, R>, u: MatRef<'_, R>) -> Mat<R>;
}

/// EKF-specific nonlinear model extension requiring explicit Jacobians.
///
/// This implementation does not estimate Jacobians numerically. Callers
/// must provide the local linearizations directly so the estimator does not
/// hide finite-difference step-size heuristics.
pub trait DiscreteExtendedKalmanModel<R>: DiscreteNonlinearModel<R> {
    /// Returns `df/dx` evaluated at the supplied state and input.
    fn transition_jacobian(&self, x: MatRef<'_, R>, u: MatRef<'_, R>) -> Mat<R>;
    /// Returns `dh/dx` evaluated at the supplied state and input.
    fn output_jacobian(&self, x: MatRef<'_, R>, u: MatRef<'_, R>) -> Mat<R>;
}

/// Prediction stage of a discrete nonlinear Kalman filter.
#[derive(Clone, Debug)]
pub struct NonlinearKalmanPrediction<R: CompensatedField>
where
    R::Real: Float + Copy,
{
    /// Predicted state estimate before measurement incorporation.
    pub state: Mat<R>,
    /// Predicted covariance before measurement incorporation.
    pub covariance: Mat<R>,
    /// Predicted output associated with the predicted state.
    ///
    /// For the UKF path this is the unscented predicted measurement mean
    /// reconstructed from the propagated sigma cloud, not merely `h(x^-)`.
    ///
    /// For the split EKF path this is the prediction-time measurement
    /// evaluation. [`ExtendedKalmanFilter::update`] may recompute the
    /// measurement-side prediction from its own input argument so callers can
    /// use a different measurement context than the transition-side input used
    /// during [`ExtendedKalmanFilter::predict`].
    pub output: Mat<R>,
    /// Measurement-side sigma cloud cached by the UKF prediction stage.
    ///
    /// This is built from the final predicted pair `(x^-, P^-)` after process
    /// noise injection, so the update stage uses measurement statistics
    /// consistent with the returned prediction covariance. The cache also
    /// preserves any custom `UkfStage::Update` sigma-point placement chosen by
    /// the active strategy.
    ukf_sigma: Option<SigmaPointSet<R>>,
}

/// Measurement update result of a discrete nonlinear Kalman filter.
#[derive(Clone, Debug)]
pub struct NonlinearKalmanUpdate<R: CompensatedField>
where
    R::Real: Float + Copy,
{
    /// Measurement innovation `y - y^-`.
    pub innovation: Mat<R>,
    /// Euclidean norm of the innovation.
    pub innovation_norm: R::Real,
    /// Innovation covariance.
    pub innovation_covariance: Mat<R>,
    /// Square root of the normalized innovation energy.
    pub normalized_innovation_norm: R::Real,
    /// Update gain.
    pub gain: Mat<R>,
    /// Predicted output used in the innovation.
    pub predicted_output: Mat<R>,
    /// Posterior state estimate.
    pub state: Mat<R>,
    /// Posterior covariance.
    pub covariance: Mat<R>,
    /// Posterior output.
    pub output: Mat<R>,
}

/// Parameters controlling the standard scaled unscented transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnscentedParams<R> {
    /// Primary spread parameter.
    pub alpha: R,
    /// Prior knowledge parameter; `2` is the common Gaussian default.
    pub beta: R,
    /// Secondary spread parameter.
    pub kappa: R,
}

impl<R> Default for UnscentedParams<R>
where
    R: Float + NumCast,
{
    fn default() -> Self {
        Self {
            alpha: NumCast::from(1.0e-3).unwrap(),
            beta: NumCast::from(2.0).unwrap(),
            kappa: R::zero(),
        }
    }
}

/// Weighted sigma-point set used by the UKF.
///
/// Custom sigma-point providers must supply both the point locations and the
/// corresponding mean/covariance weights. Once the point placement departs from
/// the standard symmetric rule, the usual built-in weights are no longer
/// implied by the algorithm.
#[derive(Clone, Debug)]
pub struct SigmaPointSet<R> {
    /// Sigma points, one point per column.
    pub points: Mat<R>,
    /// Weights used to reconstruct the mean.
    pub mean_weights: Vec<R>,
    /// Weights used to reconstruct covariance-like moments.
    pub cov_weights: Vec<R>,
}

/// Stage tag passed to custom sigma-point providers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UkfStage {
    /// Sigma points generated from the posterior state/covariance.
    Predict,
    /// Sigma points generated from the predicted state/covariance.
    Update,
}

/// Expert hook for overriding UKF sigma-point placement.
///
/// This is intended for advanced cases such as discontinuous models where the
/// standard sigma-point spread may bridge invalid regions of the state space.
pub trait SigmaPointProvider<R> {
    /// Returns the weighted sigma-point set to use for the supplied mean,
    /// covariance, input, and UKF stage.
    fn sigma_points(
        &self,
        mean: MatRef<'_, R>,
        covariance: MatRef<'_, R>,
        input: MatRef<'_, R>,
        stage: UkfStage,
    ) -> Result<SigmaPointSet<R>, NonlinearEstimatorError>;
}

/// UKF sigma-point strategy.
pub enum SigmaPointStrategy<R> {
    /// Standard scaled unscented transform.
    Standard(UnscentedParams<R>),
    /// User-supplied sigma-point provider.
    Custom(Box<dyn SigmaPointProvider<R>>),
}

impl<R> SigmaPointStrategy<R> {
    /// Resolves the sigma-point set for one UKF stage.
    ///
    /// The standard path derives points from the current mean/covariance. The
    /// custom path delegates the full point-and-weight choice to the caller so
    /// discontinuity-aware placement can be injected without changing the rest
    /// of the UKF algebra.
    fn sigma_points(
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

/// Errors produced by EKF and UKF runtimes.
#[derive(Debug)]
pub enum NonlinearEstimatorError {
    /// A supplied matrix or vector had incompatible dimensions.
    DimensionMismatch {
        /// Identifies the matrix or vector that failed the shape check.
        which: &'static str,
        /// Required row count.
        expected_nrows: usize,
        /// Required column count.
        expected_ncols: usize,
        /// Actual row count supplied by the caller.
        actual_nrows: usize,
        /// Actual column count supplied by the caller.
        actual_ncols: usize,
    },
    /// A model evaluation returned the wrong structural shape.
    InvalidModelOutput {
        /// Identifies the model output being validated.
        which: &'static str,
        /// Required row count.
        expected_nrows: usize,
        /// Required column count.
        expected_ncols: usize,
        /// Actual row count returned by the model.
        actual_nrows: usize,
        /// Actual column count returned by the model.
        actual_ncols: usize,
    },
    /// The innovation covariance was singular or numerically unusable.
    SingularInnovationCovariance,
    /// A predicted covariance solve needed by the Joseph-form UKF update was
    /// singular or numerically unusable.
    SingularPredictedCovariance,
    /// The covariance matrix was not positive definite enough for sigma-point
    /// generation.
    NonPositiveDefiniteCovariance {
        /// Identifies the covariance matrix that failed the factorization.
        which: &'static str,
    },
    /// Standard unscented-transform parameters were invalid.
    InvalidUnscentedParams {
        /// Identifies the invalid scalar or parameter combination.
        which: &'static str,
    },
    /// A custom sigma-point set was malformed.
    InvalidSigmaPointSet {
        /// Identifies the violated sigma-point-set constraint.
        which: &'static str,
    },
    /// A model evaluation or numerical solve produced non-finite output.
    NonFiniteResult {
        /// Identifies the computation that produced the non-finite value.
        which: &'static str,
    },
}

impl fmt::Display for NonlinearEstimatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for NonlinearEstimatorError {}

/// Discrete-time extended Kalman filter.
#[derive(Debug)]
pub struct ExtendedKalmanFilter<M, R>
where
    M: DiscreteExtendedKalmanModel<R>,
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
    sigma_strategy: SigmaPointStrategy<R>,
}

impl<M, R> ExtendedKalmanFilter<M, R>
where
    M: DiscreteExtendedKalmanModel<R>,
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    /// Builds an EKF with the default covariance-update policy.
    pub fn new(
        model: M,
        q: Mat<R>,
        r: Mat<R>,
        x_hat: Mat<R>,
        p: Mat<R>,
    ) -> Result<Self, NonlinearEstimatorError> {
        Self::new_with_covariance_update(model, q, r, x_hat, p, CovarianceUpdate::default())
    }

    /// Builds an EKF with an explicit covariance-update policy.
    pub fn new_with_covariance_update(
        model: M,
        q: Mat<R>,
        r: Mat<R>,
        x_hat: Mat<R>,
        p: Mat<R>,
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

    /// Computes the EKF prediction stage without mutating the filter state.
    ///
    /// In the split API, this method evaluates the transition and also stores
    /// the prediction-time measurement value `h(x^-, u_predict)`. The later
    /// [`update`](Self::update) call may use a different measurement-side input
    /// and will then recompute `h(x^-, u_update)` and `H(x^-, u_update)` from
    /// that update input so the innovation and Jacobian stay consistent.
    pub fn predict(
        &self,
        input: MatRef<'_, R>,
    ) -> Result<NonlinearKalmanPrediction<R>, NonlinearEstimatorError> {
        validate_column_vector("input", input, self.model.ninputs())?;
        let state = self.model.transition(self.x_hat.as_ref(), input);
        validate_model_output("transition", state.as_ref(), self.model.nstates(), 1)?;
        let f = self.model.transition_jacobian(self.x_hat.as_ref(), input);
        validate_square("transition_jacobian", f.as_ref(), self.model.nstates())?;
        let covariance = dense_add(
            dense_mul_adjoint_rhs(dense_mul(f.as_ref(), self.p.as_ref()).as_ref(), f.as_ref())
                .as_ref(),
            self.q.as_ref(),
        );
        let mut covariance = covariance;
        hermitian_project_in_place(&mut covariance);
        let output = self.model.output(state.as_ref(), input);
        validate_model_output("output", output.as_ref(), self.model.noutputs(), 1)?;

        validate_finite("prediction.state", state.as_ref())?;
        validate_finite("prediction.covariance", covariance.as_ref())?;
        validate_finite("prediction.output", output.as_ref())?;

        Ok(NonlinearKalmanPrediction {
            state,
            covariance,
            output,
            ukf_sigma: None,
        })
    }

    /// Applies one EKF measurement update to an externally supplied prediction.
    ///
    /// Unlike the monolithic [`step`](Self::step) path, the split EKF API lets
    /// callers provide a measurement-side input that differs from the
    /// transition-side input used during [`predict`](Self::predict). To support
    /// input-dependent sensors or late-arriving measurement context, this
    /// method recomputes the predicted measurement and output Jacobian from the
    /// supplied `input` instead of reusing the prediction-time output blindly.
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

        let predicted_output = self.model.output(prediction.state.as_ref(), input);
        validate_model_output(
            "predicted_output",
            predicted_output.as_ref(),
            self.model.noutputs(),
            1,
        )?;
        validate_finite("predicted_output", predicted_output.as_ref())?;
        let innovation = dense_sub(measurement, predicted_output.as_ref());
        let h = self.model.output_jacobian(prediction.state.as_ref(), input);
        validate_rect(
            "output_jacobian",
            h.as_ref(),
            self.model.noutputs(),
            self.model.nstates(),
        )?;
        let innovation_covariance = dense_add(
            dense_mul_adjoint_rhs(
                dense_mul(h.as_ref(), prediction.covariance.as_ref()).as_ref(),
                h.as_ref(),
            )
            .as_ref(),
            self.r.as_ref(),
        );
        let mut innovation_covariance = innovation_covariance;
        hermitian_project_in_place(&mut innovation_covariance);
        let cross = dense_mul_adjoint_rhs(prediction.covariance.as_ref(), h.as_ref());
        let gain = solve_right_checked(
            cross.as_ref(),
            innovation_covariance.as_ref(),
            default_tolerance::<R>(),
            NonlinearEstimatorError::SingularInnovationCovariance,
        )?;
        let state = dense_add(
            prediction.state.as_ref(),
            dense_mul(gain.as_ref(), innovation.as_ref()).as_ref(),
        );
        let covariance = updated_covariance(
            self.covariance_update,
            prediction.covariance.as_ref(),
            gain.as_ref(),
            h.as_ref(),
            self.r.as_ref(),
            innovation_covariance.as_ref(),
        );
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
            gain,
            predicted_output,
            state,
            covariance,
            output,
        })
    }

    /// Runs one full EKF predict/update cycle and stores the posterior state.
    ///
    /// This monolithic entry point uses the same input for both the transition
    /// and measurement sides of the EKF. Callers that need a different
    /// measurement-side input or sensor context should use the split
    /// [`predict`](Self::predict) and [`update`](Self::update) methods, which
    /// intentionally reevaluate the measurement model during `update`.
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
            ukf_sigma: Some(measurement_sigma),
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

        let owned_sigma;
        let sigma = if let Some(sigma) = prediction.ukf_sigma.as_ref() {
            sigma
        } else {
            owned_sigma = self.sigma_strategy.sigma_points(
                prediction.state.as_ref(),
                prediction.covariance.as_ref(),
                input,
                UkfStage::Update,
            )?;
            &owned_sigma
        };
        validate_sigma_point_set(sigma, self.model.nstates())?;

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
            NonlinearEstimatorError::SingularInnovationCovariance,
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

/// Validates the shared EKF/UKF model-state dimensions.
fn validate_nonlinear_filter_model<R>(
    nstates: usize,
    ninputs: usize,
    noutputs: usize,
    q: MatRef<'_, R>,
    r: MatRef<'_, R>,
    x_hat: MatRef<'_, R>,
    p: MatRef<'_, R>,
) -> Result<(), NonlinearEstimatorError> {
    validate_square("q", q, nstates)?;
    validate_square("r", r, noutputs)?;
    validate_square("p", p, nstates)?;
    validate_column_vector("x_hat", x_hat, nstates)?;
    if ninputs == 0 && x_hat.ncols() != 1 {
        return Err(NonlinearEstimatorError::DimensionMismatch {
            which: "x_hat",
            expected_nrows: nstates,
            expected_ncols: 1,
            actual_nrows: x_hat.nrows(),
            actual_ncols: x_hat.ncols(),
        });
    }
    Ok(())
}

/// Validates the structural shape of a nonlinear prediction bundle.
fn validate_prediction<R>(
    prediction: &NonlinearKalmanPrediction<R>,
    nstates: usize,
    noutputs: usize,
    which: &'static str,
) -> Result<(), NonlinearEstimatorError>
where
    R: CompensatedField,
    R::Real: Float + Copy,
{
    validate_column_vector(which_state(which), prediction.state.as_ref(), nstates)?;
    validate_square(
        which_covariance(which),
        prediction.covariance.as_ref(),
        nstates,
    )?;
    validate_column_vector(which_output(which), prediction.output.as_ref(), noutputs)?;
    Ok(())
}

fn which_state(prefix: &'static str) -> &'static str {
    match prefix {
        "prediction" => "prediction.state",
        _ => "state",
    }
}

fn which_covariance(prefix: &'static str) -> &'static str {
    match prefix {
        "prediction" => "prediction.covariance",
        _ => "covariance",
    }
}

fn which_output(prefix: &'static str) -> &'static str {
    match prefix {
        "prediction" => "prediction.output",
        _ => "output",
    }
}

/// Validates a square dense matrix shape.
fn validate_square<R>(
    which: &'static str,
    matrix: MatRef<'_, R>,
    expected_dim: usize,
) -> Result<(), NonlinearEstimatorError> {
    validate_rect(which, matrix, expected_dim, expected_dim)
}

/// Validates a general dense matrix shape against explicit row/column targets.
fn validate_rect<R>(
    which: &'static str,
    matrix: MatRef<'_, R>,
    expected_nrows: usize,
    expected_ncols: usize,
) -> Result<(), NonlinearEstimatorError> {
    if matrix.nrows() != expected_nrows || matrix.ncols() != expected_ncols {
        return Err(NonlinearEstimatorError::DimensionMismatch {
            which,
            expected_nrows,
            expected_ncols,
            actual_nrows: matrix.nrows(),
            actual_ncols: matrix.ncols(),
        });
    }
    Ok(())
}

/// Validates a dense column-vector shape.
fn validate_column_vector<R>(
    which: &'static str,
    matrix: MatRef<'_, R>,
    expected_nrows: usize,
) -> Result<(), NonlinearEstimatorError> {
    validate_rect(which, matrix, expected_nrows, 1)
}

/// Validates the output returned by a nonlinear model callback.
///
/// Model callbacks get their own error variant so user-model contract
/// violations are easier to distinguish from ordinary caller-side dimension
/// mismatches.
fn validate_model_output<R>(
    which: &'static str,
    matrix: MatRef<'_, R>,
    expected_nrows: usize,
    expected_ncols: usize,
) -> Result<(), NonlinearEstimatorError> {
    if matrix.nrows() != expected_nrows || matrix.ncols() != expected_ncols {
        return Err(NonlinearEstimatorError::InvalidModelOutput {
            which,
            expected_nrows,
            expected_ncols,
            actual_nrows: matrix.nrows(),
            actual_ncols: matrix.ncols(),
        });
    }
    Ok(())
}

/// Rejects non-finite matrices produced by model callbacks or dense solves.
fn validate_finite<R>(
    which: &'static str,
    matrix: MatRef<'_, R>,
) -> Result<(), NonlinearEstimatorError>
where
    R: ComplexField + Copy,
{
    if !matrix.is_all_finite() {
        return Err(NonlinearEstimatorError::NonFiniteResult { which });
    }
    Ok(())
}

/// Validates the standard scaled-unscented-transform parameters.
///
/// The key check is that the implied `n + lambda` spread scale stays strictly
/// positive. If it does not, the standard sigma-point construction degenerates.
fn validate_unscented_params<R>(
    params: UnscentedParams<R>,
    nstates: usize,
) -> Result<(), NonlinearEstimatorError>
where
    R: Float + NumCast,
{
    if !params.alpha.is_finite() || params.alpha <= R::zero() {
        return Err(NonlinearEstimatorError::InvalidUnscentedParams { which: "alpha" });
    }
    if !params.beta.is_finite() {
        return Err(NonlinearEstimatorError::InvalidUnscentedParams { which: "beta" });
    }
    if !params.kappa.is_finite() {
        return Err(NonlinearEstimatorError::InvalidUnscentedParams { which: "kappa" });
    }
    let n: R = NumCast::from(nstates).unwrap();
    let scaling = params.alpha * params.alpha * (n + params.kappa);
    if !scaling.is_finite() || scaling <= R::zero() {
        return Err(NonlinearEstimatorError::InvalidUnscentedParams {
            which: "n_plus_lambda",
        });
    }
    Ok(())
}

/// Validates an expert-supplied sigma-point set before the UKF consumes it.
///
/// This validation enforces the structural invariants that matter most for
/// correctness: point count, weight count, finiteness, and mean-weight
/// normalization.
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
///
/// The covariance spread comes from a dense Hermitian square root. This accepts
/// positive semidefinite covariances such as exact known-state initializations
/// by producing repeated sigma points in the zero-variance directions, while
/// still rejecting genuinely indefinite inputs rather than injecting silent
/// jitter.
fn standard_sigma_points<R>(
    mean: MatRef<'_, R>,
    covariance: MatRef<'_, R>,
    params: UnscentedParams<R>,
) -> Result<SigmaPointSet<R>, NonlinearEstimatorError>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    validate_column_vector("mean", mean, covariance.nrows())?;
    validate_square("covariance", covariance, mean.nrows())?;
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
///
/// Each input column is treated as one sigma point and the callback must return
/// a column vector of fixed length for every point.
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
    validate_model_output(which, first.as_ref(), expected_nrows, 1)?;
    validate_finite(which, first.as_ref())?;
    let mut out = Mat::zeros(expected_nrows, npoints);
    out.as_mut().subcols_mut(0, 1).copy_from(first.as_ref());
    for idx in 1..npoints {
        let value = map(points.subcols(idx, 1));
        validate_model_output(which, value.as_ref(), expected_nrows, 1)?;
        validate_finite(which, value.as_ref())?;
        out.as_mut().subcols_mut(idx, 1).copy_from(value.as_ref());
    }
    Ok(out)
}

/// Reconstructs a weighted mean from sigma points stored columnwise.
fn weighted_mean<R>(points: MatRef<'_, R>, weights: &[R]) -> Mat<R>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    Mat::from_fn(points.nrows(), 1, |row, _| {
        let mut acc = CompensatedSum::<R>::default();
        for col in 0..points.ncols() {
            acc.add(points[(row, col)] * weights[col]);
        }
        acc.finish()
    })
}

/// Reconstructs a covariance-like second moment from centered sigma points.
///
/// This is used both for predicted state covariance and predicted measurement
/// covariance in the UKF.
fn weighted_covariance<R>(points: MatRef<'_, R>, mean: MatRef<'_, R>, weights: &[R]) -> Mat<R>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    Mat::from_fn(points.nrows(), points.nrows(), |row, col| {
        let mut acc = CompensatedSum::<R>::default();
        for k in 0..points.ncols() {
            let dr = points[(row, k)] - mean[(row, 0)];
            let dc = points[(col, k)] - mean[(col, 0)];
            acc.add(dr * dc.conj() * weights[k]);
        }
        acc.finish()
    })
}

/// Reconstructs the state/measurement cross covariance used in the UKF gain.
fn weighted_cross_covariance<R>(
    lhs_points: MatRef<'_, R>,
    lhs_mean: MatRef<'_, R>,
    rhs_points: MatRef<'_, R>,
    rhs_mean: MatRef<'_, R>,
    weights: &[R],
) -> Mat<R>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    Mat::from_fn(lhs_points.nrows(), rhs_points.nrows(), |row, col| {
        let mut acc = CompensatedSum::<R>::default();
        for k in 0..lhs_points.ncols() {
            let dl = lhs_points[(row, k)] - lhs_mean[(row, 0)];
            let dr = rhs_points[(col, k)] - rhs_mean[(col, 0)];
            acc.add(dl * dr.conj() * weights[k]);
        }
        acc.finish()
    })
}

/// Applies the EKF covariance update using either the simple or Joseph form.
fn updated_covariance<R>(
    covariance_update: CovarianceUpdate,
    predicted_covariance: MatRef<'_, R>,
    gain: MatRef<'_, R>,
    h: MatRef<'_, R>,
    r: MatRef<'_, R>,
    innovation_covariance: MatRef<'_, R>,
) -> Mat<R>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    match covariance_update {
        CovarianceUpdate::Simple => dense_sub(
            predicted_covariance,
            dense_mul_adjoint_rhs(dense_mul(gain, innovation_covariance).as_ref(), gain).as_ref(),
        ),
        CovarianceUpdate::Joseph => {
            let identity = identity::<R>(predicted_covariance.nrows());
            let kh = dense_mul(gain, h);
            let i_minus_kh = dense_sub(identity.as_ref(), kh.as_ref());
            let first = dense_mul_adjoint_rhs(
                dense_mul(i_minus_kh.as_ref(), predicted_covariance).as_ref(),
                i_minus_kh.as_ref(),
            );
            let second = dense_mul_adjoint_rhs(dense_mul(gain, r).as_ref(), gain);
            dense_add(first.as_ref(), second.as_ref())
        }
    }
}

/// Applies the UKF covariance update.
///
/// The simple path uses the usual `P - K S K^H` subtraction. For the Joseph
/// path the UKF has no explicit measurement Jacobian, so this helper recovers
/// the effective linear sensitivity implied by the cross covariance before
/// delegating to the same Joseph-form algebra used by the EKF.
fn updated_covariance_ukf<R>(
    covariance_update: CovarianceUpdate,
    predicted_covariance: MatRef<'_, R>,
    gain: MatRef<'_, R>,
    cross: MatRef<'_, R>,
    r: MatRef<'_, R>,
    innovation_covariance: MatRef<'_, R>,
) -> Result<Mat<R>, NonlinearEstimatorError>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    match covariance_update {
        CovarianceUpdate::Simple => Ok(dense_sub(
            predicted_covariance,
            dense_mul_adjoint_rhs(dense_mul(gain, innovation_covariance).as_ref(), gain).as_ref(),
        )),
        CovarianceUpdate::Joseph => {
            // The UKF does not form an explicit measurement Jacobian. For the
            // Joseph update we recover the effective linearized sensitivity
            // implied by the cross covariance `P_xy = P^- H^T`.
            let h_t = solve_left_checked(
                predicted_covariance,
                cross,
                default_tolerance::<R>(),
                NonlinearEstimatorError::SingularPredictedCovariance,
            )?;
            Ok(updated_covariance(
                CovarianceUpdate::Joseph,
                predicted_covariance,
                gain,
                dense_transpose(h_t.as_ref()).as_ref(),
                r,
                innovation_covariance,
            ))
        }
    }
}

/// Returns the square root of the normalized innovation energy.
///
/// This matches the runtime consistency metric used by the linear estimator
/// layer, applied here to the nonlinear innovation covariance.
fn normalized_innovation_norm<R>(
    innovation: MatRef<'_, R>,
    innovation_covariance: MatRef<'_, R>,
) -> Result<R::Real, NonlinearEstimatorError>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    let whitened = solve_left_checked(
        innovation_covariance,
        innovation,
        default_tolerance::<R>(),
        NonlinearEstimatorError::SingularInnovationCovariance,
    )?;
    Ok(inner_product_real(innovation, whitened.as_ref())
        .max(<R::Real as Zero>::zero())
        .sqrt())
}

/// Solves `lhs * X = rhs` and rejects numerically unusable results.
///
/// The nonlinear layer keeps the same posture as the linear estimator code:
/// avoid explicit inversion, but also reject obviously inconsistent dense
/// solves using a residual check.
fn solve_left_checked<R>(
    lhs: MatRef<'_, R>,
    rhs: MatRef<'_, R>,
    tol: R::Real,
    err: NonlinearEstimatorError,
) -> Result<Mat<R>, NonlinearEstimatorError>
where
    R: ComplexField + Copy,
    R::Real: Float + Copy,
{
    let solution = lhs.full_piv_lu().solve(rhs);
    if !solution.as_ref().is_all_finite() {
        return Err(err);
    }
    let residual = dense_sub_plain(dense_mul_plain(lhs, solution.as_ref()).as_ref(), rhs);
    let residual_norm = frobenius_norm_plain(residual.as_ref());
    let scale = frobenius_norm_plain(lhs) * frobenius_norm_plain(solution.as_ref())
        + frobenius_norm_plain(rhs);
    let one = <R::Real as One>::one();
    let threshold = scale.max(one) * tol * (one + one);
    if !residual_norm.is_finite() || residual_norm > threshold {
        return Err(err);
    }
    Ok(solution)
}

/// Solves `X * lhs = rhs` by transposing into [`solve_left_checked`].
fn solve_right_checked<R>(
    rhs_left: MatRef<'_, R>,
    lhs_right: MatRef<'_, R>,
    tol: R::Real,
    err: NonlinearEstimatorError,
) -> Result<Mat<R>, NonlinearEstimatorError>
where
    R: ComplexField + Copy,
    R::Real: Float + Copy,
{
    let lhs_t = dense_transpose(lhs_right);
    let rhs_t = dense_transpose(rhs_left);
    let solved_t = solve_left_checked(lhs_t.as_ref(), rhs_t.as_ref(), tol, err)?;
    Ok(dense_transpose(solved_t.as_ref()))
}

fn default_tolerance<R>() -> R::Real
where
    R: CompensatedField,
    R::Real: Float + Copy,
{
    R::Real::epsilon().sqrt()
}

/// Clones a dense matrix reference into an owned matrix.
fn clone_mat<R: Copy>(matrix: MatRef<'_, R>) -> Mat<R> {
    Mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        matrix[(row, col)]
    })
}

/// Returns a dense identity matrix of the requested size.
fn identity<R>(dim: usize) -> Mat<R>
where
    R: ComplexField + Copy,
{
    Mat::from_fn(
        dim,
        dim,
        |row, col| if row == col { R::one() } else { R::zero() },
    )
}

/// Returns the plain transpose of a dense matrix.
fn dense_transpose<R: Copy>(matrix: MatRef<'_, R>) -> Mat<R> {
    Mat::from_fn(matrix.ncols(), matrix.nrows(), |row, col| {
        matrix[(col, row)]
    })
}

/// Dense matrix multiply using compensated accumulation per output entry.
fn dense_mul<R>(lhs: MatRef<'_, R>, rhs: MatRef<'_, R>) -> Mat<R>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    Mat::from_fn(lhs.nrows(), rhs.ncols(), |row, col| {
        let mut acc = CompensatedSum::<R>::default();
        for k in 0..lhs.ncols() {
            acc.add(lhs[(row, k)] * rhs[(k, col)]);
        }
        acc.finish()
    })
}

/// Dense multiply with an adjoint on the right-hand factor.
///
/// This is the common covariance-like pattern `A B^H` used throughout the
/// estimator implementation.
fn dense_mul_adjoint_rhs<R>(lhs: MatRef<'_, R>, rhs: MatRef<'_, R>) -> Mat<R>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    Mat::from_fn(lhs.nrows(), rhs.nrows(), |row, col| {
        let mut acc = CompensatedSum::<R>::default();
        for k in 0..lhs.ncols() {
            acc.add(lhs[(row, k)] * rhs[(col, k)].conj());
        }
        acc.finish()
    })
}

/// Dense matrix addition using compensated accumulation per entry.
fn dense_add<R>(lhs: MatRef<'_, R>, rhs: MatRef<'_, R>) -> Mat<R>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| {
        let mut acc = CompensatedSum::<R>::default();
        acc.add(lhs[(row, col)]);
        acc.add(rhs[(row, col)]);
        acc.finish()
    })
}

/// Dense matrix subtraction using compensated accumulation per entry.
fn dense_sub<R>(lhs: MatRef<'_, R>, rhs: MatRef<'_, R>) -> Mat<R>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| {
        let mut acc = CompensatedSum::<R>::default();
        acc.add(lhs[(row, col)]);
        acc.add(-rhs[(row, col)]);
        acc.finish()
    })
}

/// Plain dense matrix multiply used only in residual checks.
fn dense_mul_plain<R>(lhs: MatRef<'_, R>, rhs: MatRef<'_, R>) -> Mat<R>
where
    R: ComplexField + Copy,
{
    Mat::from_fn(lhs.nrows(), rhs.ncols(), |row, col| {
        let mut acc = R::zero();
        for k in 0..lhs.ncols() {
            acc = acc + lhs[(row, k)] * rhs[(k, col)];
        }
        acc
    })
}

/// Plain dense subtraction used only in residual checks.
fn dense_sub_plain<R>(lhs: MatRef<'_, R>, rhs: MatRef<'_, R>) -> Mat<R>
where
    R: ComplexField + Copy,
{
    Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| {
        lhs[(row, col)] - rhs[(row, col)]
    })
}

/// Returns the real part of the Hermitian inner product between two vectors.
fn inner_product_real<R>(lhs: MatRef<'_, R>, rhs: MatRef<'_, R>) -> R::Real
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    let mut acc = CompensatedSum::<R>::default();
    for row in 0..lhs.nrows() {
        acc.add(lhs[(row, 0)].conj() * rhs[(row, 0)]);
    }
    acc.finish().real()
}

/// Returns the Euclidean norm of a dense column vector.
fn column_vector_norm<R>(vector: MatRef<'_, R>) -> R::Real
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    let mut acc = <R::Real as Zero>::zero();
    for row in 0..vector.nrows() {
        acc = acc + vector[(row, 0)].abs2();
    }
    acc.sqrt()
}

/// Projects a dense matrix onto the Hermitian subspace in place.
///
/// Small skew drift is expected after repeated covariance operations.
/// Projecting back prevents that drift from accumulating into obviously
/// inconsistent covariance matrices.
fn hermitian_project_in_place<R>(matrix: &mut Mat<R>)
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    let one = <R::Real as One>::one();
    let half = one / (one + one);
    for col in 0..matrix.ncols() {
        for row in 0..=col {
            let avg = (matrix[(row, col)] + matrix[(col, row)].conj()).mul_real(half);
            matrix[(row, col)] = avg;
            matrix[(col, row)] = avg.conj();
        }
    }
}

/// Returns the Frobenius norm of a dense matrix without compensation.
///
/// This is only used inside residual checks where a simple secondary norm
/// calculation is sufficient.
fn frobenius_norm_plain<R>(matrix: MatRef<'_, R>) -> R::Real
where
    R: ComplexField + Copy,
    R::Real: Float + Copy,
{
    let mut acc = <R::Real as Zero>::zero();
    for col in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            acc = acc + matrix[(row, col)].abs2();
        }
    }
    acc.sqrt()
}

#[cfg(test)]
mod tests {
    use super::{
        DiscreteExtendedKalmanModel, DiscreteNonlinearModel, ExtendedKalmanFilter,
        NonlinearEstimatorError, SigmaPointProvider, SigmaPointSet, UkfStage,
        UnscentedKalmanFilter, UnscentedParams,
    };
    use crate::control::estimation::{CovarianceUpdate, DiscreteKalmanFilter};
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

    impl DiscreteExtendedKalmanModel<f64> for QuadraticModel {
        fn transition_jacobian(
            &self,
            x: faer::MatRef<'_, f64>,
            _u: faer::MatRef<'_, f64>,
        ) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| 2.0 * x[(0, 0)])
        }

        fn output_jacobian(&self, x: faer::MatRef<'_, f64>, _u: faer::MatRef<'_, f64>) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| 2.0 * x[(0, 0)])
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

    fn assert_close(lhs: f64, rhs: f64, tol: f64) {
        assert!(
            (lhs - rhs).abs() <= tol,
            "lhs={lhs:?}, rhs={rhs:?}, tol={tol:?}"
        );
    }

    #[test]
    fn ekf_matches_scalar_manual_reference() {
        let model = QuadraticModel;
        let q = Mat::from_fn(1, 1, |_, _| 0.1);
        let r = Mat::from_fn(1, 1, |_, _| 0.2);
        let x_hat = Mat::from_fn(1, 1, |_, _| 2.0);
        let p = Mat::from_fn(1, 1, |_, _| 0.25);
        let ekf = ExtendedKalmanFilter::new(model, q, r, x_hat, p).unwrap();
        let u = Mat::from_fn(1, 1, |_, _| 0.5);
        let y = Mat::from_fn(1, 1, |_, _| 4.1);

        let prediction = ekf.predict(u.as_ref()).unwrap();
        assert_close(prediction.state[(0, 0)], 4.5, 1.0e-12);
        assert_close(prediction.covariance[(0, 0)], 4.1, 1.0e-12);
        assert_close(prediction.output[(0, 0)], 20.25, 1.0e-12);

        let update = ekf.update(&prediction, u.as_ref(), y.as_ref()).unwrap();
        let s = 9.0 * 4.1 * 9.0 + 0.2;
        let k = 4.1 * 9.0 / s;
        let expected_state = 4.5 + k * (4.1 - 20.25);
        let expected_cov = (1.0 - k * 9.0) * 4.1 * (1.0 - k * 9.0) + k * 0.2 * k;
        assert_close(update.gain[(0, 0)], k, 1.0e-12);
        assert_close(update.state[(0, 0)], expected_state, 1.0e-12);
        assert_close(update.covariance[(0, 0)], expected_cov, 1.0e-12);
    }

    #[test]
    fn ekf_joseph_and_simple_match_on_scalar_problem() {
        let q = Mat::from_fn(1, 1, |_, _| 0.1);
        let r = Mat::from_fn(1, 1, |_, _| 0.2);
        let x_hat = Mat::from_fn(1, 1, |_, _| 2.0);
        let p = Mat::from_fn(1, 1, |_, _| 0.25);
        let simple = ExtendedKalmanFilter::new_with_covariance_update(
            QuadraticModel,
            q.clone(),
            r.clone(),
            x_hat.clone(),
            p.clone(),
            CovarianceUpdate::Simple,
        )
        .unwrap();
        let joseph = ExtendedKalmanFilter::new_with_covariance_update(
            QuadraticModel,
            q,
            r,
            x_hat,
            p,
            CovarianceUpdate::Joseph,
        )
        .unwrap();
        let u = Mat::from_fn(1, 1, |_, _| 0.5);
        let y = Mat::from_fn(1, 1, |_, _| 4.1);

        let pred_simple = simple.predict(u.as_ref()).unwrap();
        let pred_joseph = joseph.predict(u.as_ref()).unwrap();
        let upd_simple = simple.update(&pred_simple, u.as_ref(), y.as_ref()).unwrap();
        let upd_joseph = joseph.update(&pred_joseph, u.as_ref(), y.as_ref()).unwrap();
        assert_close(upd_simple.state[(0, 0)], upd_joseph.state[(0, 0)], 1.0e-12);
        assert_close(
            upd_simple.covariance[(0, 0)],
            upd_joseph.covariance[(0, 0)],
            1.0e-12,
        );
    }

    struct InputScaledMeasurementModel;

    impl DiscreteNonlinearModel<f64> for InputScaledMeasurementModel {
        fn nstates(&self) -> usize {
            1
        }

        fn ninputs(&self) -> usize {
            1
        }

        fn noutputs(&self) -> usize {
            1
        }

        fn transition(&self, x: faer::MatRef<'_, f64>, _u: faer::MatRef<'_, f64>) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| x[(0, 0)])
        }

        fn output(&self, x: faer::MatRef<'_, f64>, u: faer::MatRef<'_, f64>) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| u[(0, 0)] * x[(0, 0)])
        }
    }

    impl DiscreteExtendedKalmanModel<f64> for InputScaledMeasurementModel {
        fn transition_jacobian(
            &self,
            _x: faer::MatRef<'_, f64>,
            _u: faer::MatRef<'_, f64>,
        ) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| 1.0)
        }

        fn output_jacobian(&self, _x: faer::MatRef<'_, f64>, u: faer::MatRef<'_, f64>) -> Mat<f64> {
            Mat::from_fn(1, 1, |_, _| u[(0, 0)])
        }
    }

    #[test]
    fn ekf_split_update_recomputes_measurement_model_from_update_input() {
        let q = Mat::from_fn(1, 1, |_, _| 0.0);
        let r = Mat::from_fn(1, 1, |_, _| 0.25);
        let x_hat = Mat::from_fn(1, 1, |_, _| 2.0);
        let p = Mat::from_fn(1, 1, |_, _| 0.5);
        let ekf_split = ExtendedKalmanFilter::new(
            InputScaledMeasurementModel,
            q.clone(),
            r.clone(),
            x_hat.clone(),
            p.clone(),
        )
        .unwrap();
        let mut ekf_step =
            ExtendedKalmanFilter::new(InputScaledMeasurementModel, q, r, x_hat, p).unwrap();

        let predict_input = Mat::from_fn(1, 1, |_, _| 0.0);
        let update_input = Mat::from_fn(1, 1, |_, _| 1.0);
        let measurement = Mat::from_fn(1, 1, |_, _| 1.5);

        let prediction = ekf_split.predict(predict_input.as_ref()).unwrap();
        let split_update = ekf_split
            .update(&prediction, update_input.as_ref(), measurement.as_ref())
            .unwrap();
        let step_update = ekf_step
            .step(update_input.as_ref(), measurement.as_ref())
            .unwrap();

        assert_close(
            split_update.predicted_output[(0, 0)],
            step_update.predicted_output[(0, 0)],
            1.0e-12,
        );
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
    fn ekf_rejects_dimension_mismatch() {
        let q = Mat::from_fn(2, 2, |_, _| 0.0);
        let r = Mat::from_fn(1, 1, |_, _| 1.0);
        let x_hat = Mat::from_fn(1, 1, |_, _| 0.0);
        let p = Mat::from_fn(1, 1, |_, _| 1.0);
        let err = ExtendedKalmanFilter::new(QuadraticModel, q, r, x_hat, p).unwrap_err();
        assert!(matches!(
            err,
            NonlinearEstimatorError::DimensionMismatch { which: "q", .. }
        ));
    }

    #[test]
    fn ekf_rejects_singular_innovation_covariance() {
        let model = LinearScalarModel {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
        };
        let q = Mat::from_fn(1, 1, |_, _| 0.0);
        let r = Mat::from_fn(1, 1, |_, _| 0.0);
        let x_hat = Mat::from_fn(1, 1, |_, _| 0.0);
        let p = Mat::from_fn(1, 1, |_, _| 1.0);
        let ekf = ExtendedKalmanFilter::new(model, q, r, x_hat, p).unwrap();
        let u = Mat::from_fn(1, 1, |_, _| 0.0);
        let y = Mat::from_fn(1, 1, |_, _| 0.0);
        let pred = ekf.predict(u.as_ref()).unwrap();
        let err = ekf.update(&pred, u.as_ref(), y.as_ref()).unwrap_err();
        assert!(matches!(
            err,
            NonlinearEstimatorError::SingularInnovationCovariance
        ));
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
    fn ukf_update_reuses_prediction_measurement_sigma_points() {
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

        // The prediction stage caches the update-side sigma set built from the
        // final `(x^-, P^-)` pair, so the measurement mean seen by `update`
        // must agree exactly with the one returned by `predict`.
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
        assert_eq!(update_calls.get(), 1);
        assert_close(update.predicted_output[(0, 0)], 1.0625, 1.0e-12);
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
