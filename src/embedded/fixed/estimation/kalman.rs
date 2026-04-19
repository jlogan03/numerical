//! Fixed-size discrete-time linear Kalman filtering.

use crate::embedded::error::EmbeddedError;
use crate::embedded::fixed::linalg::{
    Matrix, Vector, identity_matrix, mat_add, mat_mul, mat_sub, mat_vec_mul, solve_linear_system,
    transpose, vec_add, vec_norm, vec_sub,
};
use crate::embedded::fixed::lti::DiscreteStateSpace;
use num_traits::Float;

/// Covariance update policy used by [`DiscreteKalmanFilter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CovarianceUpdate {
    /// `P^+ = P^- - K S K^T`
    Simple,
    /// `P^+ = (I - K C) P^- (I - K C)^T + K V K^T`
    Joseph,
}

impl Default for CovarianceUpdate {
    fn default() -> Self {
        Self::Joseph
    }
}

/// One non-mutating prediction stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanPrediction<T, const NX: usize, const NY: usize> {
    /// Predicted state estimate.
    pub state: Vector<T, NX>,
    /// Predicted covariance.
    pub covariance: Matrix<T, NX, NX>,
    /// Predicted output.
    pub output: Vector<T, NY>,
}

/// One non-mutating measurement update stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanUpdate<T, const NX: usize, const NY: usize> {
    /// Innovation `y - (C x^- + D u)`.
    pub innovation: Vector<T, NY>,
    /// Euclidean innovation norm.
    pub innovation_norm: T,
    /// Innovation covariance.
    pub innovation_covariance: Matrix<T, NY, NY>,
    /// Normalized innovation norm.
    pub normalized_innovation_norm: T,
    /// Kalman gain.
    pub gain: Matrix<T, NX, NY>,
    /// Measurement-side predicted output.
    pub predicted_output: Vector<T, NY>,
    /// Posterior state estimate.
    pub state: Vector<T, NX>,
    /// Posterior covariance.
    pub covariance: Matrix<T, NX, NX>,
    /// Posterior output.
    pub output: Vector<T, NY>,
}

/// Fixed-size recursive discrete-time Kalman filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiscreteKalmanFilter<T, const NX: usize, const NU: usize, const NY: usize> {
    /// Discrete-time plant model.
    pub system: DiscreteStateSpace<T, NX, NU, NY>,
    /// Process-noise covariance.
    pub w: Matrix<T, NX, NX>,
    /// Measurement-noise covariance.
    pub v: Matrix<T, NY, NY>,
    /// Covariance update policy.
    pub covariance_update: CovarianceUpdate,
    /// Current posterior state estimate.
    pub x_hat: Vector<T, NX>,
    /// Current posterior covariance.
    pub p: Matrix<T, NX, NX>,
}

/// Fixed-gain steady-state discrete-time observer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SteadyStateKalmanFilter<T, const NX: usize, const NU: usize, const NY: usize> {
    /// Discrete-time plant model.
    pub system: DiscreteStateSpace<T, NX, NU, NY>,
    /// Fixed observer gain.
    pub gain: Matrix<T, NX, NY>,
    /// Current state estimate.
    pub x_hat: Vector<T, NX>,
    /// Optional steady-state covariance.
    pub steady_state_covariance: Option<Matrix<T, NX, NX>>,
}

impl<T, const NX: usize, const NU: usize, const NY: usize> DiscreteKalmanFilter<T, NX, NU, NY>
where
    T: Float + Copy,
{
    /// Creates a fixed-size discrete Kalman filter.
    pub fn new(
        system: DiscreteStateSpace<T, NX, NU, NY>,
        w: Matrix<T, NX, NX>,
        v: Matrix<T, NY, NY>,
        x_hat: Vector<T, NX>,
        p: Matrix<T, NX, NX>,
    ) -> Result<Self, EmbeddedError> {
        Self::new_with_covariance_update(system, w, v, x_hat, p, CovarianceUpdate::default())
    }

    /// Creates a fixed-size discrete Kalman filter with an explicit covariance
    /// update policy.
    pub fn new_with_covariance_update(
        system: DiscreteStateSpace<T, NX, NU, NY>,
        w: Matrix<T, NX, NX>,
        v: Matrix<T, NY, NY>,
        x_hat: Vector<T, NX>,
        p: Matrix<T, NX, NX>,
        covariance_update: CovarianceUpdate,
    ) -> Result<Self, EmbeddedError> {
        Ok(Self {
            system,
            w,
            v,
            covariance_update,
            x_hat,
            p,
        })
    }

    /// Returns the current posterior estimate.
    #[must_use]
    pub fn state_estimate(&self) -> &Vector<T, NX> {
        &self.x_hat
    }

    /// Returns the current posterior covariance.
    #[must_use]
    pub fn covariance(&self) -> &Matrix<T, NX, NX> {
        &self.p
    }

    /// Computes the non-mutating prediction stage.
    pub fn predict(
        &self,
        input: Vector<T, NU>,
    ) -> Result<KalmanPrediction<T, NX, NY>, EmbeddedError> {
        let state = self.system.next_state(&self.x_hat, &input);
        let covariance = mat_add(
            &mat_mul(
                &mat_mul(self.system.a(), &self.p),
                &transpose(self.system.a()),
            ),
            &self.w,
        );
        let output = self.system.output(&state, &input);
        Ok(KalmanPrediction {
            state,
            covariance,
            output,
        })
    }

    /// Computes the non-mutating measurement update.
    pub fn update(
        &self,
        prediction: &KalmanPrediction<T, NX, NY>,
        input: Vector<T, NU>,
        measurement: Vector<T, NY>,
    ) -> Result<KalmanUpdate<T, NX, NY>, EmbeddedError> {
        let predicted_output = self.system.output(&prediction.state, &input);
        let innovation = vec_sub(&measurement, &predicted_output);
        let innovation_covariance = mat_add(
            &mat_mul(
                &mat_mul(self.system.c(), &prediction.covariance),
                &transpose(self.system.c()),
            ),
            &self.v,
        );
        let cross_covariance = mat_mul(&prediction.covariance, &transpose(self.system.c()));
        let gain = transpose(&solve_linear_system(
            &innovation_covariance,
            &transpose(&cross_covariance),
            "kalman.innovation_covariance",
        )?);
        let whitened_innovation = solve_linear_system(
            &innovation_covariance,
            &column_matrix(&innovation),
            "kalman.innovation_covariance",
        )?;
        let state = vec_add(&prediction.state, &mat_vec_mul(&gain, &innovation));
        let covariance = updated_covariance(
            self.covariance_update,
            &prediction.covariance,
            &gain,
            self.system.c(),
            &self.v,
            &innovation_covariance,
        );
        let output = self.system.output(&state, &input);
        let innovation_norm = vec_norm(&innovation);
        let normalized_innovation_norm =
            normalized_innovation_norm(&innovation, &whitened_innovation).sqrt();

        Ok(KalmanUpdate {
            innovation,
            innovation_norm,
            innovation_covariance,
            normalized_innovation_norm,
            gain,
            predicted_output,
            state,
            covariance,
            output,
        })
    }

    /// Runs one full recursive predict/update cycle and stores the posterior
    /// estimate.
    pub fn step(
        &mut self,
        input: Vector<T, NU>,
        measurement: Vector<T, NY>,
    ) -> Result<KalmanUpdate<T, NX, NY>, EmbeddedError> {
        let prediction = self.predict(input)?;
        let update = self.update(&prediction, input, measurement)?;
        self.x_hat = update.state;
        self.p = update.covariance;
        Ok(update)
    }
}

impl<T, const NX: usize, const NU: usize, const NY: usize> SteadyStateKalmanFilter<T, NX, NU, NY>
where
    T: Float + Copy,
{
    /// Creates a fixed-gain steady-state observer.
    pub fn new(
        system: DiscreteStateSpace<T, NX, NU, NY>,
        gain: Matrix<T, NX, NY>,
        x_hat: Vector<T, NX>,
        steady_state_covariance: Option<Matrix<T, NX, NX>>,
    ) -> Self {
        Self {
            system,
            gain,
            x_hat,
            steady_state_covariance,
        }
    }

    /// Returns the current state estimate.
    #[must_use]
    pub fn state_estimate(&self) -> &Vector<T, NX> {
        &self.x_hat
    }

    /// Computes the predictor stage `A x + B u`.
    pub fn predict(&self, input: Vector<T, NU>) -> Vector<T, NX> {
        self.system.next_state(&self.x_hat, &input)
    }

    /// Applies one fixed-gain update from an externally supplied prediction.
    pub fn update(
        &self,
        prediction_state: Vector<T, NX>,
        input: Vector<T, NU>,
        measurement: Vector<T, NY>,
    ) -> (Vector<T, NY>, Vector<T, NX>, Vector<T, NY>) {
        let predicted_output = self.system.output(&prediction_state, &input);
        let innovation = vec_sub(&measurement, &predicted_output);
        let state = vec_add(&prediction_state, &mat_vec_mul(&self.gain, &innovation));
        let output = self.system.output(&state, &input);
        (innovation, state, output)
    }

    /// Runs one full fixed-gain observer step and stores the new estimate.
    pub fn step(&mut self, input: Vector<T, NU>, measurement: Vector<T, NY>) -> Vector<T, NY> {
        let prediction = self.predict(input);
        let (_innovation, state, output) = self.update(prediction, input, measurement);
        self.x_hat = state;
        output
    }
}

/// Packs one fixed-size vector into a single-column right-hand side block.
fn column_matrix<T, const N: usize>(vector: &Vector<T, N>) -> Matrix<T, N, 1>
where
    T: Float + Copy,
{
    let mut out = [[T::zero(); 1]; N];
    for idx in 0..N {
        out[idx][0] = vector[idx];
    }
    out
}

/// Applies the configured covariance update law.
fn updated_covariance<T, const NX: usize, const NY: usize>(
    covariance_update: CovarianceUpdate,
    prediction_covariance: &Matrix<T, NX, NX>,
    gain: &Matrix<T, NX, NY>,
    c: &Matrix<T, NY, NX>,
    v: &Matrix<T, NY, NY>,
    innovation_covariance: &Matrix<T, NY, NY>,
) -> Matrix<T, NX, NX>
where
    T: Float + Copy,
{
    match covariance_update {
        CovarianceUpdate::Simple => mat_sub(
            prediction_covariance,
            &mat_mul(&mat_mul(gain, innovation_covariance), &transpose(gain)),
        ),
        CovarianceUpdate::Joseph => {
            let identity = identity_matrix::<T, NX>();
            let residual = mat_sub(&identity, &mat_mul(gain, c));
            mat_add(
                &mat_mul(
                    &mat_mul(&residual, prediction_covariance),
                    &transpose(&residual),
                ),
                &mat_mul(&mat_mul(gain, v), &transpose(gain)),
            )
        }
    }
}

/// Computes the normalized innovation energy `r^T z` after solving `S z = r`.
fn normalized_innovation_norm<T, const NY: usize>(
    innovation: &Vector<T, NY>,
    whitened_innovation: &Matrix<T, NY, 1>,
) -> T
where
    T: Float + Copy,
{
    let mut acc = T::zero();
    for idx in 0..NY {
        acc = acc + innovation[idx] * whitened_innovation[idx][0];
    }
    acc.max(T::zero())
}

#[cfg(feature = "alloc")]
impl<T, const NX: usize, const NU: usize, const NY: usize> DiscreteKalmanFilter<T, NX, NU, NY>
where
    T: Float + Copy + faer_traits::RealField,
{
    /// Builds a fixed-size embedded Kalman filter from the dynamic control-side
    /// state-space model and covariance data.
    pub fn from_control_state_space(
        system: &crate::control::lti::DiscreteStateSpace<T>,
        w: Matrix<T, NX, NX>,
        v: Matrix<T, NY, NY>,
        x_hat: Vector<T, NX>,
        p: Matrix<T, NX, NX>,
    ) -> Result<Self, EmbeddedError> {
        Self::new(DiscreteStateSpace::try_from(system)?, w, v, x_hat, p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_kalman_step_runs() {
        let system = DiscreteStateSpace::new(
            [[1.0, 0.1], [0.0, 1.0]],
            [[0.0], [0.1]],
            [[1.0, 0.0]],
            [[0.0]],
            0.1,
        )
        .unwrap();
        let mut filter = DiscreteKalmanFilter::new(
            system,
            [[1.0e-3, 0.0], [0.0, 1.0e-3]],
            [[1.0e-2]],
            [0.0, 0.0],
            identity_matrix::<f64, 2>(),
        )
        .unwrap();

        let update = filter.step([0.0], [1.0]).unwrap();
        assert!(update.output[0].is_finite());
        assert!(filter.state_estimate()[0].is_finite());
    }
}
