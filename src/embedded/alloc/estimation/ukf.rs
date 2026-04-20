//! Simplified SPD-only dynamic-size discrete-time unscented Kalman filtering.

use super::dense::{
    cholesky_lower, column_matrix_to_vector, llt_solve, mat_add, mat_mul_vec, scale_matrix,
    transpose, vec_add, vec_as_slice, vec_as_slice_mut, vec_norm, vec_sub, vector_as_column_matrix,
    vector_from_slice, vectors_as_columns, zero_vector,
};
use super::ekf::DiscreteNonlinearModel;
use super::map_nonlinear_error;
use crate::control::estimation::CovarianceUpdate;
use crate::control::estimation::nonlinear_core::{
    normalized_innovation_norm, updated_covariance_ukf, weighted_covariance,
    weighted_cross_covariance, weighted_mean,
};
use crate::embedded::EmbeddedError;
use crate::embedded::alloc::{Matrix, Vector};
use crate::sparse::compensated::CompensatedField;
use alloc::vec::Vec;
use faer_traits::RealField;
use num_traits::Float;

/// One UKF prediction stage.
#[derive(Clone, Debug, PartialEq)]
pub struct UnscentedKalmanPrediction<T> {
    /// Predicted state estimate.
    pub state: Vector<T>,
    /// Predicted covariance.
    pub covariance: Matrix<T>,
    /// Propagated state sigma points.
    pub sigma_points: Vec<Vector<T>>,
}

/// One UKF update stage.
#[derive(Clone, Debug, PartialEq)]
pub struct UnscentedKalmanUpdate<T> {
    /// Innovation `y - y^-`.
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
    /// Predicted measurement.
    pub predicted_output: Vector<T>,
}

/// Simplified SPD-only unscented Kalman filter runtime.
#[derive(Clone, Debug, PartialEq)]
pub struct UnscentedKalmanFilter<T, M> {
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
    /// Spread parameter.
    pub alpha: T,
    /// Prior-knowledge parameter.
    pub beta: T,
    /// Secondary spread parameter.
    pub kappa: T,
}

impl<T, M> UnscentedKalmanFilter<T, M>
where
    T: CompensatedField + RealField,
    T::Real: Float + Copy,
    M: DiscreteNonlinearModel<T>,
{
    /// Creates a validated SPD-only UKF runtime.
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
                which: "embedded.alloc.ukf.x_hat",
                expected: nx,
                actual: x_hat.nrows(),
            });
        }
        if w.nrows() != nx || w.ncols() != nx {
            return Err(EmbeddedError::DimensionMismatch {
                which: "embedded.alloc.ukf.w",
                expected_rows: nx,
                expected_cols: nx,
                actual_rows: w.nrows(),
                actual_cols: w.ncols(),
            });
        }
        if v.nrows() != ny || v.ncols() != ny {
            return Err(EmbeddedError::DimensionMismatch {
                which: "embedded.alloc.ukf.v",
                expected_rows: ny,
                expected_cols: ny,
                actual_rows: v.nrows(),
                actual_cols: v.ncols(),
            });
        }
        if p.nrows() != nx || p.ncols() != nx {
            return Err(EmbeddedError::DimensionMismatch {
                which: "embedded.alloc.ukf.p",
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
            alpha: T::epsilon().sqrt(),
            beta: T::one() + T::one(),
            kappa: T::zero(),
        })
    }

    /// Overrides the sigma-point spread parameters.
    pub fn with_sigma_parameters(
        mut self,
        alpha: T,
        beta: T,
        kappa: T,
    ) -> Result<Self, EmbeddedError> {
        if !alpha.is_finite() || alpha <= T::zero() {
            return Err(EmbeddedError::InvalidParameter {
                which: "embedded.alloc.ukf.alpha",
            });
        }
        if !beta.is_finite() || beta < T::zero() {
            return Err(EmbeddedError::InvalidParameter {
                which: "embedded.alloc.ukf.beta",
            });
        }
        if !kappa.is_finite() {
            return Err(EmbeddedError::InvalidParameter {
                which: "embedded.alloc.ukf.kappa",
            });
        }
        self.alpha = alpha;
        self.beta = beta;
        self.kappa = kappa;
        Ok(self)
    }

    /// Returns the current posterior estimate.
    #[must_use]
    pub fn state_estimate(&self) -> &Vector<T> {
        &self.x_hat
    }

    /// Returns the current posterior covariance.
    #[must_use]
    pub fn covariance(&self) -> &Matrix<T> {
        &self.p
    }

    /// Computes the non-mutating sigma-point prediction stage.
    pub fn predict(&self, input: &[T]) -> Result<UnscentedKalmanPrediction<T>, EmbeddedError> {
        validate_input_dim(&self.model, input)?;
        let weights = sigma_weights(self.model.state_dim(), self.alpha, self.beta, self.kappa)?;
        let sigma_points = sigma_points(&self.x_hat, &self.p, self.alpha, self.kappa)?;

        let mut propagated = Vec::with_capacity(sigma_points.len());
        for sigma_point in &sigma_points {
            let mut next_state = zero_vector(self.model.state_dim());
            self.model.transition(
                vec_as_slice(sigma_point),
                input,
                vec_as_slice_mut(&mut next_state),
            );
            propagated.push(next_state);
        }

        let propagated_matrix =
            vectors_as_columns(&propagated, "embedded.alloc.ukf.propagated_sigma_points")?;
        let state = column_matrix_to_vector(
            &weighted_mean(propagated_matrix.as_ref(), vec_as_slice(&weights.mean)),
            "embedded.alloc.ukf.predicted_state",
        )?;
        let covariance = mat_add(
            &weighted_covariance(
                propagated_matrix.as_ref(),
                vector_as_column_matrix(&state).as_ref(),
                vec_as_slice(&weights.covariance),
            ),
            &self.w,
        )?;

        Ok(UnscentedKalmanPrediction {
            state,
            covariance,
            sigma_points: propagated,
        })
    }

    /// Computes the non-mutating UKF measurement update.
    pub fn update(
        &self,
        prediction: &UnscentedKalmanPrediction<T>,
        input: &[T],
        measurement: &[T],
    ) -> Result<UnscentedKalmanUpdate<T>, EmbeddedError> {
        validate_input_dim(&self.model, input)?;
        validate_output_dim(&self.model, measurement)?;

        let weights = sigma_weights(self.model.state_dim(), self.alpha, self.beta, self.kappa)?;
        let mut measurement_points = Vec::with_capacity(prediction.sigma_points.len());
        for sigma_point in &prediction.sigma_points {
            let mut output = zero_vector(self.model.output_dim());
            self.model.output(
                vec_as_slice(sigma_point),
                input,
                vec_as_slice_mut(&mut output),
            );
            measurement_points.push(output);
        }

        let state_points_matrix = vectors_as_columns(
            &prediction.sigma_points,
            "embedded.alloc.ukf.state_sigma_points",
        )?;
        let measurement_points_matrix = vectors_as_columns(
            &measurement_points,
            "embedded.alloc.ukf.measurement_sigma_points",
        )?;
        let predicted_output = column_matrix_to_vector(
            &weighted_mean(
                measurement_points_matrix.as_ref(),
                vec_as_slice(&weights.mean),
            ),
            "embedded.alloc.ukf.predicted_output",
        )?;
        let innovation = vec_sub(&vector_from_slice(measurement), &predicted_output)?;
        let innovation_covariance = mat_add(
            &weighted_covariance(
                measurement_points_matrix.as_ref(),
                vector_as_column_matrix(&predicted_output).as_ref(),
                vec_as_slice(&weights.covariance),
            ),
            &self.v,
        )?;
        let cross_covariance = weighted_cross_covariance(
            state_points_matrix.as_ref(),
            vector_as_column_matrix(&prediction.state).as_ref(),
            measurement_points_matrix.as_ref(),
            vector_as_column_matrix(&predicted_output).as_ref(),
            vec_as_slice(&weights.covariance),
        );
        let gain = transpose(&llt_solve(
            &innovation_covariance,
            &transpose(&cross_covariance),
            "embedded.alloc.ukf.innovation_covariance",
        )?);
        let state = vec_add(&prediction.state, &mat_mul_vec(&gain, &innovation)?)?;
        let covariance = updated_covariance_ukf(
            CovarianceUpdate::Simple,
            prediction.covariance.as_ref(),
            gain.as_ref(),
            cross_covariance.as_ref(),
            self.v.as_ref(),
            innovation_covariance.as_ref(),
        )
        .map_err(map_nonlinear_error)?;

        Ok(UnscentedKalmanUpdate {
            innovation_norm: vec_norm(&innovation),
            normalized_innovation_norm: normalized_innovation_norm(
                vector_as_column_matrix(&innovation).as_ref(),
                innovation_covariance.as_ref(),
            )
            .map_err(map_nonlinear_error)?,
            innovation,
            innovation_covariance,
            gain,
            state,
            covariance,
            predicted_output,
        })
    }

    /// Runs one full UKF step and stores the posterior estimate.
    pub fn step(
        &mut self,
        input: &[T],
        measurement: &[T],
    ) -> Result<UnscentedKalmanUpdate<T>, EmbeddedError> {
        let prediction = self.predict(input)?;
        let update = self.update(&prediction, input, measurement)?;
        self.x_hat = update.state.clone();
        self.p = update.covariance.clone();
        Ok(update)
    }
}

/// Weight vectors used by the standard scaled unscented transform.
struct SigmaWeights<T> {
    mean: Vector<T>,
    covariance: Vector<T>,
}

/// Computes the standard scaled unscented weights.
fn sigma_weights<T>(n: usize, alpha: T, beta: T, kappa: T) -> Result<SigmaWeights<T>, EmbeddedError>
where
    T: Float + Copy,
{
    let n_t = T::from(n).ok_or(EmbeddedError::InvalidParameter {
        which: "embedded.alloc.ukf.state_dim",
    })?;
    let lambda = alpha * alpha * (n_t + kappa) - n_t;
    let scale = n_t + lambda;
    if !scale.is_finite() || scale <= T::zero() {
        return Err(EmbeddedError::InvalidParameter {
            which: "embedded.alloc.ukf.scale",
        });
    }
    let count = 2 * n + 1;
    let mut mean = zero_vector(count);
    let mut covariance = zero_vector(count);
    let repeated = T::one() / ((T::one() + T::one()) * scale);
    for idx in 0..count {
        mean[idx] = repeated;
        covariance[idx] = repeated;
    }
    mean[0] = lambda / scale;
    covariance[0] = mean[0] + (T::one() - alpha * alpha + beta);
    Ok(SigmaWeights { mean, covariance })
}

/// Builds the sigma points around the current state estimate.
fn sigma_points<T>(
    state: &Vector<T>,
    covariance: &Matrix<T>,
    alpha: T,
    kappa: T,
) -> Result<Vec<Vector<T>>, EmbeddedError>
where
    T: Float + Copy,
{
    let n = state.nrows();
    let n_t = T::from(n).ok_or(EmbeddedError::InvalidParameter {
        which: "embedded.alloc.ukf.state_dim",
    })?;
    let lambda = alpha * alpha * (n_t + kappa) - n_t;
    let scale = n_t + lambda;
    let factor = cholesky_lower(
        &scale_matrix(covariance, scale),
        "embedded.alloc.ukf.cholesky",
    )?;

    let mut points = Vec::with_capacity(2 * n + 1);
    points.push(state.clone());
    for col in 0..n {
        let mut plus = state.clone();
        let mut minus = state.clone();
        for row in 0..n {
            plus[row] = plus[row] + factor[(row, col)];
            minus[row] = minus[row] - factor[(row, col)];
        }
        points.push(plus);
        points.push(minus);
    }
    Ok(points)
}

/// Validates the nonlinear-model input dimension.
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

/// Validates the nonlinear-model output dimension.
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

    #[test]
    fn alloc_ukf_step_runs() {
        let mut filter = UnscentedKalmanFilter::new(
            QuadraticSensor,
            scale_matrix(&identity_matrix(1), 1.0e-3),
            scale_matrix(&identity_matrix(1), 1.0e-2),
            vector_from_slice(&[0.5]),
            identity_matrix(1),
        )
        .unwrap();

        let update = filter.step(&[0.1], &[0.49]).unwrap();
        assert!(update.predicted_output[0].is_finite());
        assert!(filter.state_estimate()[0].is_finite());
    }
}
