//! Simplified SPD-only dynamic-size discrete-time unscented Kalman filtering.

use super::ekf::DiscreteNonlinearModel;
use crate::embedded::EmbeddedError;
use crate::embedded::alloc::Matrix;
use crate::embedded::alloc::matrix::{outer_product, quadratic_form, vec_add, vec_norm, vec_sub};
use alloc::vec;
use alloc::vec::Vec;
use num_traits::Float;

/// One UKF prediction stage.
#[derive(Clone, Debug, PartialEq)]
pub struct UnscentedKalmanPrediction<T> {
    /// Predicted state estimate.
    pub state: Vec<T>,
    /// Predicted covariance.
    pub covariance: Matrix<T>,
    /// Propagated state sigma points.
    pub sigma_points: Vec<Vec<T>>,
}

/// One UKF update stage.
#[derive(Clone, Debug, PartialEq)]
pub struct UnscentedKalmanUpdate<T> {
    /// Innovation `y - y^-`.
    pub innovation: Vec<T>,
    /// Euclidean innovation norm.
    pub innovation_norm: T,
    /// Innovation covariance.
    pub innovation_covariance: Matrix<T>,
    /// Normalized innovation norm.
    pub normalized_innovation_norm: T,
    /// Kalman gain.
    pub gain: Matrix<T>,
    /// Posterior state estimate.
    pub state: Vec<T>,
    /// Posterior covariance.
    pub covariance: Matrix<T>,
    /// Predicted measurement.
    pub predicted_output: Vec<T>,
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
    pub x_hat: Vec<T>,
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
    T: Float + Copy,
    M: DiscreteNonlinearModel<T>,
{
    /// Creates a validated SPD-only UKF runtime.
    pub fn new(
        model: M,
        w: Matrix<T>,
        v: Matrix<T>,
        x_hat: Vec<T>,
        p: Matrix<T>,
    ) -> Result<Self, EmbeddedError> {
        let nx = model.state_dim();
        let ny = model.output_dim();
        if x_hat.len() != nx {
            return Err(EmbeddedError::LengthMismatch {
                which: "embedded.alloc.ukf.x_hat",
                expected: nx,
                actual: x_hat.len(),
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
    pub fn state_estimate(&self) -> &[T] {
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
        let mut idx = 0usize;
        while idx < sigma_points.len() {
            let mut next_state = vec![T::zero(); self.model.state_dim()];
            self.model
                .transition(&sigma_points[idx], input, &mut next_state);
            propagated.push(next_state);
            idx += 1;
        }

        let state = weighted_mean(&propagated, &weights.mean);
        let covariance =
            weighted_covariance(&propagated, &state, &weights.covariance)?.add(&self.w)?;

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
        let mut idx = 0usize;
        while idx < prediction.sigma_points.len() {
            let mut output = vec![T::zero(); self.model.output_dim()];
            self.model
                .output(&prediction.sigma_points[idx], input, &mut output);
            measurement_points.push(output);
            idx += 1;
        }

        let predicted_output = weighted_mean(&measurement_points, &weights.mean);
        let innovation = vec_sub(measurement, &predicted_output)?;
        let innovation_covariance =
            weighted_covariance(&measurement_points, &predicted_output, &weights.covariance)?
                .add(&self.v)?;
        let innovation_covariance_inv =
            innovation_covariance.inverse("embedded.alloc.ukf.innovation_covariance")?;
        let cross_covariance = cross_covariance(
            &prediction.sigma_points,
            &prediction.state,
            &measurement_points,
            &predicted_output,
            &weights.covariance,
        )?;
        let gain = cross_covariance.mul(&innovation_covariance_inv)?;
        let state = vec_add(&prediction.state, &gain.mul_vec(&innovation)?)?;
        let covariance = prediction
            .covariance
            .sub(&gain.mul(&innovation_covariance)?.mul(&gain.transpose())?)?;

        Ok(UnscentedKalmanUpdate {
            innovation_norm: vec_norm(&innovation),
            normalized_innovation_norm: quadratic_form(&innovation_covariance_inv, &innovation)?
                .max(T::zero())
                .sqrt(),
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
    mean: Vec<T>,
    covariance: Vec<T>,
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
    let mut mean = vec![T::one() / ((T::one() + T::one()) * scale); count];
    let mut covariance = mean.clone();
    mean[0] = lambda / scale;
    covariance[0] = mean[0] + (T::one() - alpha * alpha + beta);
    Ok(SigmaWeights { mean, covariance })
}

/// Builds the sigma points around the current state estimate.
fn sigma_points<T>(
    state: &[T],
    covariance: &Matrix<T>,
    alpha: T,
    kappa: T,
) -> Result<Vec<Vec<T>>, EmbeddedError>
where
    T: Float + Copy,
{
    let n = state.len();
    let n_t = T::from(n).ok_or(EmbeddedError::InvalidParameter {
        which: "embedded.alloc.ukf.state_dim",
    })?;
    let lambda = alpha * alpha * (n_t + kappa) - n_t;
    let scale = n_t + lambda;
    let factor = covariance
        .scale(scale)
        .cholesky_lower("embedded.alloc.ukf.cholesky")?;

    let mut points = Vec::with_capacity(2 * n + 1);
    points.push(state.to_vec());
    let mut col = 0usize;
    while col < n {
        let mut plus = state.to_vec();
        let mut minus = state.to_vec();
        let mut row = 0usize;
        while row < n {
            plus[row] = plus[row] + factor[(row, col)];
            minus[row] = minus[row] - factor[(row, col)];
            row += 1;
        }
        points.push(plus);
        points.push(minus);
        col += 1;
    }
    Ok(points)
}

/// Computes the weighted sigma-point mean.
fn weighted_mean<T>(points: &[Vec<T>], weights: &[T]) -> Vec<T>
where
    T: Float + Copy,
{
    let dim = points.first().map_or(0, Vec::len);
    let mut mean = vec![T::zero(); dim];
    let mut idx = 0usize;
    while idx < points.len() {
        let mut j = 0usize;
        while j < dim {
            mean[j] = mean[j] + weights[idx] * points[idx][j];
            j += 1;
        }
        idx += 1;
    }
    mean
}

/// Computes the weighted covariance of one sigma-point cloud.
fn weighted_covariance<T>(
    points: &[Vec<T>],
    mean: &[T],
    weights: &[T],
) -> Result<Matrix<T>, EmbeddedError>
where
    T: Float + Copy,
{
    let dim = mean.len();
    let mut covariance = Matrix::zeros(dim, dim);
    let mut idx = 0usize;
    while idx < points.len() {
        let delta = vec_sub(&points[idx], mean)?;
        covariance = covariance.add(&outer_product(&delta, &delta).scale(weights[idx]))?;
        idx += 1;
    }
    Ok(covariance)
}

/// Computes the weighted state/measurement cross covariance.
fn cross_covariance<T>(
    state_points: &[Vec<T>],
    state_mean: &[T],
    measurement_points: &[Vec<T>],
    measurement_mean: &[T],
    weights: &[T],
) -> Result<Matrix<T>, EmbeddedError>
where
    T: Float + Copy,
{
    let mut covariance = Matrix::zeros(state_mean.len(), measurement_mean.len());
    let mut idx = 0usize;
    while idx < state_points.len() {
        let dx = vec_sub(&state_points[idx], state_mean)?;
        let dz = vec_sub(&measurement_points[idx], measurement_mean)?;
        covariance = covariance.add(&outer_product(&dx, &dz).scale(weights[idx]))?;
        idx += 1;
    }
    Ok(covariance)
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
            Matrix::identity(1).scale(1.0e-3),
            Matrix::identity(1).scale(1.0e-2),
            vec![0.5],
            Matrix::identity(1),
        )
        .unwrap();

        let update = filter.step(&[0.1], &[0.49]).unwrap();
        assert!(update.predicted_output[0].is_finite());
        assert!(filter.state_estimate()[0].is_finite());
    }
}
