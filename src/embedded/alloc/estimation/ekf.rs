//! Simplified dynamic-size discrete-time extended Kalman filtering.

use super::dense::{
    identity_matrix, llt_solve, llt_solve_vector, mat_add, mat_mul, mat_mul_vec, mat_sub,
    transpose, vec_add, vec_as_slice, vec_as_slice_mut, vec_dot, vec_norm, vec_sub,
    vector_from_slice, zero_matrix, zero_vector,
};
use crate::embedded::EmbeddedError;
use crate::embedded::alloc::{Matrix, Vector};
use faer_traits::ComplexField;
use num_traits::Float;

/// Dynamic nonlinear discrete-time model.
pub trait DiscreteNonlinearModel<T> {
    /// State dimension.
    fn state_dim(&self) -> usize;
    /// Input dimension.
    fn input_dim(&self) -> usize;
    /// Output dimension.
    fn output_dim(&self) -> usize;
    /// Evaluates the transition `x[k+1] = f(x[k], u[k])`.
    fn transition(&self, state: &[T], input: &[T], next_state: &mut [T]);
    /// Evaluates the output `y[k] = h(x[k], u[k])`.
    fn output(&self, state: &[T], input: &[T], output: &mut [T]);
}

/// EKF-specific nonlinear model with explicit Jacobians.
pub trait DiscreteExtendedKalmanModel<T>: DiscreteNonlinearModel<T> {
    /// Evaluates the transition Jacobian `∂f/∂x`.
    fn transition_jacobian(&self, state: &[T], input: &[T], jacobian: &mut Matrix<T>);
    /// Evaluates the output Jacobian `∂h/∂x`.
    fn output_jacobian(&self, state: &[T], input: &[T], jacobian: &mut Matrix<T>);
}

/// One EKF prediction stage.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtendedKalmanPrediction<T> {
    /// Predicted state estimate.
    pub state: Vector<T>,
    /// Predicted covariance.
    pub covariance: Matrix<T>,
    /// Predicted output.
    pub output: Vector<T>,
}

/// One EKF update stage.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtendedKalmanUpdate<T> {
    /// Innovation `y - h(x^-, u)`.
    pub innovation: Vector<T>,
    /// Euclidean innovation norm.
    pub innovation_norm: T,
    /// Innovation covariance.
    pub innovation_covariance: Matrix<T>,
    /// Normalized innovation norm.
    pub normalized_innovation_norm: T,
    /// Kalman gain.
    pub gain: Matrix<T>,
    /// Posterior state estimate.
    pub state: Vector<T>,
    /// Posterior covariance.
    pub covariance: Matrix<T>,
    /// Posterior output estimate.
    pub output: Vector<T>,
}

/// Dynamic-size extended Kalman filter runtime.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtendedKalmanFilter<T, M> {
    /// Nonlinear model.
    pub model: M,
    /// Process-noise covariance.
    pub w: Matrix<T>,
    /// Measurement-noise covariance.
    pub v: Matrix<T>,
    /// Current posterior state estimate.
    pub x_hat: Vector<T>,
    /// Current posterior covariance.
    pub p: Matrix<T>,
}

impl<T, M> ExtendedKalmanFilter<T, M>
where
    T: ComplexField<Real = T> + Float + Copy,
    M: DiscreteExtendedKalmanModel<T>,
{
    /// Creates a validated dynamic-size EKF runtime.
    pub fn new(
        model: M,
        w: Matrix<T>,
        v: Matrix<T>,
        x_hat: Vector<T>,
        p: Matrix<T>,
    ) -> Result<Self, EmbeddedError> {
        let nx = model.state_dim();
        let ny = model.output_dim();
        if x_hat.nrows() != nx {
            return Err(EmbeddedError::LengthMismatch {
                which: "embedded.alloc.ekf.x_hat",
                expected: nx,
                actual: x_hat.nrows(),
            });
        }
        if w.nrows() != nx || w.ncols() != nx {
            return Err(EmbeddedError::DimensionMismatch {
                which: "embedded.alloc.ekf.w",
                expected_rows: nx,
                expected_cols: nx,
                actual_rows: w.nrows(),
                actual_cols: w.ncols(),
            });
        }
        if v.nrows() != ny || v.ncols() != ny {
            return Err(EmbeddedError::DimensionMismatch {
                which: "embedded.alloc.ekf.v",
                expected_rows: ny,
                expected_cols: ny,
                actual_rows: v.nrows(),
                actual_cols: v.ncols(),
            });
        }
        if p.nrows() != nx || p.ncols() != nx {
            return Err(EmbeddedError::DimensionMismatch {
                which: "embedded.alloc.ekf.p",
                expected_rows: nx,
                expected_cols: nx,
                actual_rows: p.nrows(),
                actual_cols: p.ncols(),
            });
        }

        Ok(Self {
            model,
            w,
            v,
            x_hat,
            p,
        })
    }

    /// Returns the current posterior state estimate.
    #[must_use]
    pub fn state_estimate(&self) -> &Vector<T> {
        &self.x_hat
    }

    /// Returns the current posterior covariance.
    #[must_use]
    pub fn covariance(&self) -> &Matrix<T> {
        &self.p
    }

    /// Computes the non-mutating prediction stage.
    pub fn predict(&self, input: &[T]) -> Result<ExtendedKalmanPrediction<T>, EmbeddedError> {
        validate_input_dim(&self.model, input)?;

        let nx = self.model.state_dim();
        let mut f = zero_matrix(nx, nx);
        self.model
            .transition_jacobian(vec_as_slice(&self.x_hat), input, &mut f);

        let mut state = zero_vector(nx);
        self.model.transition(
            vec_as_slice(&self.x_hat),
            input,
            vec_as_slice_mut(&mut state),
        );

        let covariance = mat_add(&mat_mul(&mat_mul(&f, &self.p)?, &transpose(&f))?, &self.w)?;
        let mut output = zero_vector(self.model.output_dim());
        self.model
            .output(vec_as_slice(&state), input, vec_as_slice_mut(&mut output));

        Ok(ExtendedKalmanPrediction {
            state,
            covariance,
            output,
        })
    }

    /// Computes the non-mutating measurement update.
    pub fn update(
        &self,
        prediction: &ExtendedKalmanPrediction<T>,
        input: &[T],
        measurement: &[T],
    ) -> Result<ExtendedKalmanUpdate<T>, EmbeddedError> {
        validate_input_dim(&self.model, input)?;
        validate_output_dim(&self.model, measurement)?;

        let ny = self.model.output_dim();
        let mut h = zero_matrix(ny, self.model.state_dim());
        self.model
            .output_jacobian(vec_as_slice(&prediction.state), input, &mut h);

        let innovation = vec_sub(&vector_from_slice(measurement), &prediction.output)?;
        let innovation_covariance = mat_add(
            &mat_mul(&mat_mul(&h, &prediction.covariance)?, &transpose(&h))?,
            &self.v,
        )?;
        let cross_covariance = mat_mul(&prediction.covariance, &transpose(&h))?;
        let gain = transpose(&llt_solve(
            &innovation_covariance,
            &transpose(&cross_covariance),
            "embedded.alloc.ekf.innovation_covariance",
        )?);
        let whitened_innovation = llt_solve_vector(
            &innovation_covariance,
            &innovation,
            "embedded.alloc.ekf.innovation_covariance",
        )?;
        let state = vec_add(&prediction.state, &mat_mul_vec(&gain, &innovation)?)?;
        let identity = identity_matrix(self.model.state_dim());
        let residual = mat_sub(&identity, &mat_mul(&gain, &h)?)?;
        let covariance = mat_add(
            &mat_mul(
                &mat_mul(&residual, &prediction.covariance)?,
                &transpose(&residual),
            )?,
            &mat_mul(&mat_mul(&gain, &self.v)?, &transpose(&gain))?,
        )?;

        let mut output = zero_vector(ny);
        self.model
            .output(vec_as_slice(&state), input, vec_as_slice_mut(&mut output));

        Ok(ExtendedKalmanUpdate {
            innovation_norm: vec_norm(&innovation),
            normalized_innovation_norm: vec_dot(&innovation, &whitened_innovation)?
                .max(T::zero())
                .sqrt(),
            innovation,
            innovation_covariance,
            gain,
            state,
            covariance,
            output,
        })
    }

    /// Runs one full EKF step and stores the posterior estimate.
    pub fn step(
        &mut self,
        input: &[T],
        measurement: &[T],
    ) -> Result<ExtendedKalmanUpdate<T>, EmbeddedError> {
        let prediction = self.predict(input)?;
        let update = self.update(&prediction, input, measurement)?;
        self.x_hat = update.state.clone();
        self.p = update.covariance.clone();
        Ok(update)
    }
}

/// Validates the input dimension for one nonlinear model call.
fn validate_input_dim<T, M>(model: &M, input: &[T]) -> Result<(), EmbeddedError>
where
    M: DiscreteNonlinearModel<T>,
{
    if input.len() == model.input_dim() {
        Ok(())
    } else {
        Err(EmbeddedError::LengthMismatch {
            which: "embedded.alloc.estimation.input",
            expected: model.input_dim(),
            actual: input.len(),
        })
    }
}

/// Validates the measurement dimension for one nonlinear model call.
fn validate_output_dim<T, M>(model: &M, output: &[T]) -> Result<(), EmbeddedError>
where
    M: DiscreteNonlinearModel<T>,
{
    if output.len() == model.output_dim() {
        Ok(())
    } else {
        Err(EmbeddedError::LengthMismatch {
            which: "embedded.alloc.estimation.output",
            expected: model.output_dim(),
            actual: output.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::dense::{identity_matrix, scale_matrix, vector_from_slice};
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct QuadraticSensor;

    impl DiscreteNonlinearModel<f64> for QuadraticSensor {
        fn state_dim(&self) -> usize {
            1
        }

        fn input_dim(&self) -> usize {
            1
        }

        fn output_dim(&self) -> usize {
            1
        }

        fn transition(&self, state: &[f64], input: &[f64], next_state: &mut [f64]) {
            next_state[0] = state[0] + input[0];
        }

        fn output(&self, state: &[f64], _input: &[f64], output: &mut [f64]) {
            output[0] = state[0] * state[0];
        }
    }

    impl DiscreteExtendedKalmanModel<f64> for QuadraticSensor {
        fn transition_jacobian(&self, _state: &[f64], _input: &[f64], jacobian: &mut Matrix<f64>) {
            jacobian[(0, 0)] = 1.0;
        }

        fn output_jacobian(&self, state: &[f64], _input: &[f64], jacobian: &mut Matrix<f64>) {
            jacobian[(0, 0)] = 2.0 * state[0];
        }
    }

    #[test]
    fn alloc_ekf_step_runs() {
        let mut filter = ExtendedKalmanFilter::new(
            QuadraticSensor,
            scale_matrix(&identity_matrix(1), 1.0e-3),
            scale_matrix(&identity_matrix(1), 1.0e-2),
            vector_from_slice(&[0.5]),
            identity_matrix(1),
        )
        .unwrap();

        let update = filter.step(&[0.1], &[0.49]).unwrap();
        assert!(update.output[0].is_finite());
        assert!(filter.state_estimate()[0].is_finite());
    }
}
