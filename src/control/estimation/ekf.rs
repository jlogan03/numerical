use crate::control::dense_ops::{
    column_vector_norm, dense_mul, dense_mul_adjoint_rhs, hermitian_project_in_place,
};
use crate::control::estimation::CovarianceUpdate;
use crate::control::estimation::dense::{default_tolerance, solve_right_checked};
use crate::control::estimation::nonlinear_core::{
    normalized_innovation_norm, predict_covariance, updated_covariance,
};
use crate::sparse::compensated::CompensatedField;
use faer::{Mat, MatRef};
use faer_traits::RealField;
use num_traits::Float;

use super::{
    DiscreteExtendedKalmanModel, NonlinearEstimatorError, NonlinearKalmanPrediction,
    NonlinearKalmanUpdate, validate_column_vector, validate_finite, validate_model_output,
    validate_nonlinear_filter_model, validate_prediction, validate_rect, validate_square,
};

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
    pub fn predict(
        &self,
        input: MatRef<'_, R>,
    ) -> Result<NonlinearKalmanPrediction<R>, NonlinearEstimatorError> {
        validate_column_vector("input", input, self.model.ninputs())?;
        let state = self.model.transition(self.x_hat.as_ref(), input);
        validate_model_output("transition", state.as_ref(), self.model.nstates(), 1)?;
        let f = self.model.transition_jacobian(self.x_hat.as_ref(), input);
        validate_square("transition_jacobian", f.as_ref(), self.model.nstates())?;
        let covariance = predict_covariance(f.as_ref(), self.p.as_ref(), self.q.as_ref());
        let output = self.model.output(state.as_ref(), input);
        validate_model_output("output", output.as_ref(), self.model.noutputs(), 1)?;

        validate_finite("prediction.state", state.as_ref())?;
        validate_finite("prediction.covariance", covariance.as_ref())?;
        validate_finite("prediction.output", output.as_ref())?;

        Ok(NonlinearKalmanPrediction {
            state,
            covariance,
            output,
        })
    }

    /// Applies one EKF measurement update to an externally supplied prediction.
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
        let innovation = measurement.to_owned() - predicted_output.as_ref();
        let h = self.model.output_jacobian(prediction.state.as_ref(), input);
        validate_rect(
            "output_jacobian",
            h.as_ref(),
            self.model.noutputs(),
            self.model.nstates(),
        )?;
        let innovation_covariance =
            predict_covariance(h.as_ref(), prediction.covariance.as_ref(), self.r.as_ref());
        let cross = dense_mul_adjoint_rhs(prediction.covariance.as_ref(), h.as_ref());
        let gain = solve_right_checked(
            cross.as_ref(),
            innovation_covariance.as_ref(),
            default_tolerance::<R>(),
            || NonlinearEstimatorError::SingularInnovationCovariance,
        )?;
        let state =
            prediction.state.to_owned() + dense_mul(gain.as_ref(), innovation.as_ref()).as_ref();
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
    pub fn step(
        &mut self,
        input: MatRef<'_, R>,
        measurement: MatRef<'_, R>,
    ) -> Result<NonlinearKalmanUpdate<R>, NonlinearEstimatorError> {
        let prediction = self.predict(input)?;
        let update = self.update(&prediction, input, measurement)?;
        self.x_hat = update.state.to_owned();
        self.p = update.covariance.to_owned();
        Ok(update)
    }
}

#[cfg(test)]
mod tests {
    use super::ExtendedKalmanFilter;
    use crate::control::estimation::CovarianceUpdate;
    use crate::control::estimation::nonlinear::{
        DiscreteExtendedKalmanModel, DiscreteNonlinearModel, NonlinearEstimatorError,
    };
    use faer::Mat;

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
}
