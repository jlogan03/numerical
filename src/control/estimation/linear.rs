//! Dense linear estimator design and discrete-time Kalman filtering.
//!
//! This module is split into two related but distinct layers:
//!
//! - steady-state observer design through continuous/discrete Riccati duality
//! - runtime discrete-time linear Kalman filtering
//!
//! The first layer computes the observer gain `L` and steady-state error
//! covariance `P`. The second layer uses explicit predict/update recursions for
//! a discrete linear Gaussian state estimator.
//!
//! The module now also exposes two fixed-gain runtime forms built from those
//! steady-state design results:
//!
//! - [`SteadyStateKalmanFilter`] for discrete-time fixed-gain observation
//! - [`ContinuousObserver`] for continuous-time fixed-gain observation
//!
//! # Two Intuitions
//!
//! 1. **Design view.** The `LQE` / `DLQE` side computes the observer gain you
//!    would choose if the model and noise covariances were fixed forever.
//! 2. **Runtime view.** The Kalman-filter side turns that design into an online
//!    predict/update process that combines the model prediction with each new
//!    measurement.
//!
//! # Glossary
//!
//! - **Observer gain `L`:** Injection gain multiplying the innovation.
//! - **Innovation:** Residual between the measured and predicted output.
//! - **Posterior / prior:** After- and before-measurement estimates.
//! - **Joseph update:** Covariance update that better preserves positive
//!   semidefiniteness in floating point.
//!
//! # Mathematical Formulation
//!
//! The core linear runtime equations are:
//!
//! - predict: `x^- = A x + B u`, `P^- = A P A^H + W`
//! - update: `K = P^- C^H (C P^- C^H + V)^-1`
//! - posterior: `x^+ = x^- + K (y - C x^- - D u)`
//!
//! Steady-state design solves the dual Riccati equation to obtain a fixed
//! observer gain `L`.
//!
//! # Implementation Notes
//!
//! - The runtime Kalman filter is discrete-time only.
//! - Continuous runtime support is fixed-gain observer evaluation,
//!   not a continuous covariance ODE integrator.
//! - Design and runtime live together so the fixed-gain wrappers can be built
//!   directly from `LQE` / `DLQE` results.

use crate::control::lti::{ContinuousStateSpace, DiscreteStateSpace};
use crate::control::matrix_equations::{RiccatiError, solve_care_dense, solve_dare_dense};
use crate::sparse::compensated::{CompensatedField, CompensatedSum};
use core::fmt;
use faer::prelude::Solve;
use faer::{Mat, MatRef};
use faer_traits::ext::ComplexFieldExt;
use faer_traits::{ComplexField, RealField};
use num_traits::{Float, One, Zero};

/// Result of a dense continuous- or discrete-time steady-state estimator solve.
///
/// The gain `L` is the observer gain used in `A - L C`. `covariance` is the
/// steady-state error covariance returned by the dual Riccati equation. On the
/// discrete-time `DLQE` path this is the steady-state a-priori covariance
/// `P^-`, because the dual DARE directly yields the predictor-form solution.
#[derive(Clone, Debug)]
pub struct LqeSolve<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    /// Observer gain.
    pub gain: Mat<T>,
    /// Steady-state estimation-error covariance.
    ///
    /// For discrete-time `DLQE`, this is the a-priori covariance `P^-` used to
    /// form the predictor-form observer gain.
    pub covariance: Mat<T>,
    /// Closed-loop estimator state matrix `A - L C`.
    pub estimator_a: Mat<T>,
    /// Compensated Riccati residual norm.
    pub residual_norm: T::Real,
    /// Whether the dual Riccati solve passed the stabilizing check.
    pub stabilizing: bool,
}

/// Prediction stage of the discrete linear Kalman filter.
#[derive(Clone, Debug)]
pub struct KalmanPrediction<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    /// Predicted state estimate before incorporating the new measurement.
    pub state: Mat<T>,
    /// Predicted covariance before incorporating the new measurement.
    pub covariance: Mat<T>,
    /// Predicted output `C x^- + D u` corresponding to the prediction-stage
    /// input.
    ///
    /// The split-step update path may recompute the measurement-side
    /// predicted output from its own input argument so callers can supply a
    /// different measurement-side feedthrough context than the one used during
    /// prediction.
    pub output: Mat<T>,
}

/// Update result of one discrete linear Kalman filter measurement step.
#[derive(Clone, Debug)]
pub struct KalmanUpdate<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    /// Measurement innovation `y - (C x^- + D u)`.
    pub innovation: Mat<T>,
    /// Euclidean norm of the innovation.
    pub innovation_norm: T::Real,
    /// Innovation covariance `S = C P^- C^H + V`.
    pub innovation_covariance: Mat<T>,
    /// Square root of the normalized innovation energy `r^H S^-1 r`.
    pub normalized_innovation_norm: T::Real,
    /// Kalman gain for this update.
    pub gain: Mat<T>,
    /// Predicted output `C x^- + D u` used in the innovation.
    pub predicted_output: Mat<T>,
    /// Updated state estimate.
    pub state: Mat<T>,
    /// Updated covariance.
    pub covariance: Mat<T>,
    /// Posterior output `C x^+ + D u`.
    pub output: Mat<T>,
}

/// Covariance-update policy used by [`DiscreteKalmanFilter`].
///
/// `Simple` is the compact textbook update. `Joseph` is algebraically
/// equivalent in exact arithmetic, but is usually better at preserving
/// Hermitian symmetry and positive semidefiniteness in floating point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CovarianceUpdate {
    /// `P^+ = P^- - K S K^H`
    Simple,
    /// `P^+ = (I - K C) P^- (I - K C)^H + K V K^H`
    Joseph,
}

impl Default for CovarianceUpdate {
    fn default() -> Self {
        Self::Joseph
    }
}

/// Prediction stage of the fixed-gain steady-state discrete observer.
#[derive(Clone, Debug)]
pub struct SteadyStateKalmanPrediction<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    /// Predicted state estimate before measurement correction.
    pub state: Mat<T>,
    /// Predicted output `C x^- + D u` corresponding to the prediction-stage
    /// input.
    ///
    /// The split fixed-gain update path may recompute the measurement-side
    /// predicted output from its own input argument so callers can supply a
    /// different measurement-side feedthrough context than the one used during
    /// prediction.
    pub output: Mat<T>,
}

/// Update result of the fixed-gain steady-state discrete observer.
#[derive(Clone, Debug)]
pub struct SteadyStateKalmanUpdate<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    /// Measurement innovation `y - (C x^- + D u)`.
    pub innovation: Mat<T>,
    /// Euclidean norm of the innovation.
    pub innovation_norm: T::Real,
    /// Updated state estimate.
    pub state: Mat<T>,
    /// Updated output estimate `C x^+ + D u`.
    pub output: Mat<T>,
}

/// Runtime derivative information for a continuous fixed-gain observer.
#[derive(Clone, Debug)]
pub struct ContinuousObserverDerivative<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    /// Estimated output `C x_hat + D u`.
    pub output: Mat<T>,
    /// Measurement innovation `y - (C x_hat + D u)`.
    pub innovation: Mat<T>,
    /// Euclidean norm of the innovation.
    pub innovation_norm: T::Real,
    /// Observer state derivative.
    pub state_derivative: Mat<T>,
}

/// Errors produced by dense LQE/DLQE design and discrete Kalman filtering.
#[derive(Debug)]
pub enum EstimatorError {
    /// The dual Riccati solve failed.
    Riccati(RiccatiError),
    /// A supplied matrix had incompatible dimensions.
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
    /// The innovation covariance was singular or numerically unusable.
    SingularInnovationCovariance,
    /// A solve or update produced non-finite output.
    NonFiniteResult {
        /// Identifies the computation that produced the non-finite value.
        which: &'static str,
    },
}

impl fmt::Display for EstimatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl core::error::Error for EstimatorError {}

impl From<RiccatiError> for EstimatorError {
    fn from(value: RiccatiError) -> Self {
        Self::Riccati(value)
    }
}

/// Dense discrete-time linear Kalman filter with explicit predict/update
/// recursions.
///
/// This implementation assumes the process-noise covariance is already
/// expressed in state coordinates, so the prediction covariance update is
///
/// `P^- = A P A^H + W`
///
/// without a separate process-noise input matrix.
#[derive(Clone, Debug)]
pub struct DiscreteKalmanFilter<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    /// State transition matrix.
    pub a: Mat<T>,
    /// Input matrix.
    pub b: Mat<T>,
    /// Output matrix.
    pub c: Mat<T>,
    /// Feedthrough matrix.
    pub d: Mat<T>,
    /// Process-noise covariance.
    pub w: Mat<T>,
    /// Measurement-noise covariance.
    pub v: Mat<T>,
    /// Covariance-update policy used during the measurement step.
    pub covariance_update: CovarianceUpdate,
    /// Current posterior state estimate.
    pub x_hat: Mat<T>,
    /// Current posterior covariance.
    pub p: Mat<T>,
}

/// Fixed-gain steady-state discrete-time observer built from `DLQE`.
///
/// This wrapper uses a constant observer gain and does not propagate a
/// time-varying covariance. It is the common deployment form once the steady-
/// state Riccati solve has already been performed.
#[derive(Clone, Debug)]
pub struct SteadyStateKalmanFilter<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    /// State transition matrix.
    pub a: Mat<T>,
    /// Input matrix.
    pub b: Mat<T>,
    /// Output matrix.
    pub c: Mat<T>,
    /// Feedthrough matrix.
    pub d: Mat<T>,
    /// Fixed observer gain.
    pub gain: Mat<T>,
    /// Current state estimate.
    pub x_hat: Mat<T>,
    /// Optional steady-state covariance returned by `DLQE`.
    pub steady_state_covariance: Option<Mat<T>>,
}

/// Continuous fixed-gain observer built from `LQE`.
///
/// This is the continuous-time steady-state counterpart to
/// [`SteadyStateKalmanFilter`]. It exposes the observer differential equation
/// and leaves time integration to the caller.
#[derive(Clone, Debug)]
pub struct ContinuousObserver<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    /// State matrix.
    pub a: Mat<T>,
    /// Input matrix.
    pub b: Mat<T>,
    /// Output matrix.
    pub c: Mat<T>,
    /// Feedthrough matrix.
    pub d: Mat<T>,
    /// Fixed observer gain.
    pub gain: Mat<T>,
    /// Current state estimate.
    pub x_hat: Mat<T>,
}

/// Solves the dense continuous-time steady-state LQE problem.
///
/// This is the estimator-side dual of continuous-time LQR. The implementation
/// solves the CARE on `(A^H, C^H, W, V)` and converts the resulting dual gain
/// back into the observer gain `L`.
pub fn lqe_dense<T>(
    a: MatRef<'_, T>,
    c: MatRef<'_, T>,
    w: MatRef<'_, T>,
    v: MatRef<'_, T>,
) -> Result<LqeSolve<T>, EstimatorError>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    validate_lqe_dims(a, c, w, v)?;
    // Continuous-time LQE is the dual CARE problem with
    // `(A_dual, B_dual, Q_dual, R_dual) = (A^H, C^H, W, V)`.
    let dual = solve_care_dense(dense_adjoint(a).as_ref(), dense_adjoint(c).as_ref(), w, v)?;
    // The regulator-side gain lives in the dual coordinates, so transpose it
    // back to obtain the observer gain `L`.
    let gain = dense_adjoint(dual.gain.as_ref());
    Ok(LqeSolve {
        estimator_a: estimator_matrix(a, gain.as_ref(), c),
        gain,
        covariance: dual.solution,
        residual_norm: dual.residual_norm,
        stabilizing: dual.stabilizing,
    })
}

/// Solves the dense discrete-time steady-state DLQE problem.
///
/// This is the estimator-side dual of DLQR. The implementation solves the
/// DARE on `(A^H, C^H, W, V)` and converts the resulting dual gain into the
/// predictor-form observer gain `L`.
pub fn dlqe_dense<T>(
    a: MatRef<'_, T>,
    c: MatRef<'_, T>,
    w: MatRef<'_, T>,
    v: MatRef<'_, T>,
) -> Result<LqeSolve<T>, EstimatorError>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    validate_lqe_dims(a, c, w, v)?;
    // Discrete-time DLQE is the DARE dual of DLQR on `(A^H, C^H, W, V)`.
    let dual = solve_dare_dense(dense_adjoint(a).as_ref(), dense_adjoint(c).as_ref(), w, v)?;
    let gain = dense_adjoint(dual.gain.as_ref());
    Ok(LqeSolve {
        estimator_a: estimator_matrix(a, gain.as_ref(), c),
        gain,
        covariance: dual.solution,
        residual_norm: dual.residual_norm,
        stabilizing: dual.stabilizing,
    })
}

impl<T> ContinuousStateSpace<T>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    /// Designs the dense steady-state continuous-time LQE observer.
    pub fn lqe(&self, w: MatRef<'_, T>, v: MatRef<'_, T>) -> Result<LqeSolve<T>, EstimatorError> {
        lqe_dense(self.a(), self.c(), w, v)
    }
}

impl<T> DiscreteStateSpace<T>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    /// Designs the dense steady-state discrete-time DLQE observer.
    pub fn dlqe(&self, w: MatRef<'_, T>, v: MatRef<'_, T>) -> Result<LqeSolve<T>, EstimatorError> {
        dlqe_dense(self.a(), self.c(), w, v)
    }

    /// Builds a fixed-gain steady-state discrete observer from `DLQE`.
    pub fn steady_state_kalman(
        &self,
        w: MatRef<'_, T>,
        v: MatRef<'_, T>,
        x_hat: Mat<T>,
    ) -> Result<SteadyStateKalmanFilter<T>, EstimatorError> {
        SteadyStateKalmanFilter::from_dlqe(self, w, v, x_hat)
    }
}

impl<T> ContinuousStateSpace<T>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    /// Builds a fixed-gain continuous observer from `LQE`.
    pub fn observer(
        &self,
        w: MatRef<'_, T>,
        v: MatRef<'_, T>,
        x_hat: Mat<T>,
    ) -> Result<ContinuousObserver<T>, EstimatorError> {
        ContinuousObserver::from_lqe(self, w, v, x_hat)
    }
}

impl<T> DiscreteKalmanFilter<T>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    /// Builds a discrete Kalman filter from explicit model and covariance
    /// matrices.
    pub fn new(
        a: Mat<T>,
        b: Mat<T>,
        c: Mat<T>,
        d: Mat<T>,
        w: Mat<T>,
        v: Mat<T>,
        x_hat: Mat<T>,
        p: Mat<T>,
    ) -> Result<Self, EstimatorError> {
        Self::new_with_covariance_update(a, b, c, d, w, v, x_hat, p, CovarianceUpdate::default())
    }

    /// Builds a discrete Kalman filter with an explicit covariance-update
    /// policy.
    pub fn new_with_covariance_update(
        a: Mat<T>,
        b: Mat<T>,
        c: Mat<T>,
        d: Mat<T>,
        w: Mat<T>,
        v: Mat<T>,
        x_hat: Mat<T>,
        p: Mat<T>,
        covariance_update: CovarianceUpdate,
    ) -> Result<Self, EstimatorError> {
        validate_filter_model(
            a.as_ref(),
            b.as_ref(),
            c.as_ref(),
            d.as_ref(),
            w.as_ref(),
            v.as_ref(),
            x_hat.as_ref(),
            p.as_ref(),
        )?;
        Ok(Self {
            a,
            b,
            c,
            d,
            w,
            v,
            covariance_update,
            x_hat,
            p,
        })
    }

    /// Builds a discrete Kalman filter from a validated discrete state-space
    /// model plus initial estimate data.
    pub fn from_state_space(
        system: &DiscreteStateSpace<T>,
        w: Mat<T>,
        v: Mat<T>,
        x_hat: Mat<T>,
        p: Mat<T>,
    ) -> Result<Self, EstimatorError> {
        Self::new_with_covariance_update(
            clone_mat(system.a()),
            clone_mat(system.b()),
            clone_mat(system.c()),
            clone_mat(system.d()),
            w,
            v,
            x_hat,
            p,
            CovarianceUpdate::default(),
        )
    }

    /// Builds a discrete Kalman filter from a validated state-space model with
    /// an explicit covariance-update policy.
    pub fn from_state_space_with_covariance_update(
        system: &DiscreteStateSpace<T>,
        w: Mat<T>,
        v: Mat<T>,
        x_hat: Mat<T>,
        p: Mat<T>,
        covariance_update: CovarianceUpdate,
    ) -> Result<Self, EstimatorError> {
        Self::new_with_covariance_update(
            clone_mat(system.a()),
            clone_mat(system.b()),
            clone_mat(system.c()),
            clone_mat(system.d()),
            w,
            v,
            x_hat,
            p,
            covariance_update,
        )
    }

    /// Builds a full recursive discrete Kalman filter initialized from the
    /// steady-state `DLQE` fixed point.
    ///
    /// `DLQE` returns the steady-state a-priori covariance `P^-`. The runtime
    /// Kalman filter stores its mutable covariance state as the current
    /// posterior covariance `P^+`, so this constructor converts the Riccati
    /// solution to the corresponding posterior fixed point before seeding the
    /// recursion.
    pub fn from_dlqe(
        system: &DiscreteStateSpace<T>,
        w: MatRef<'_, T>,
        v: MatRef<'_, T>,
        x_hat: Mat<T>,
    ) -> Result<Self, EstimatorError> {
        let solve = system.dlqe(w, v)?;
        let (gain, innovation_covariance) =
            steady_state_filter_gain(system.c(), v, solve.covariance.as_ref())?;
        let mut posterior_covariance = updated_covariance(
            CovarianceUpdate::default(),
            solve.covariance.as_ref(),
            gain.as_ref(),
            system.c(),
            v,
            innovation_covariance.as_ref(),
        );
        hermitian_project_in_place(&mut posterior_covariance);
        Self::new_with_covariance_update(
            clone_mat(system.a()),
            clone_mat(system.b()),
            clone_mat(system.c()),
            clone_mat(system.d()),
            clone_mat(w),
            clone_mat(v),
            x_hat,
            posterior_covariance,
            CovarianceUpdate::default(),
        )
    }

    /// Returns the current posterior state estimate.
    #[must_use]
    pub fn state_estimate(&self) -> MatRef<'_, T> {
        self.x_hat.as_ref()
    }

    /// Returns the current posterior covariance.
    #[must_use]
    pub fn covariance(&self) -> MatRef<'_, T> {
        self.p.as_ref()
    }

    /// Returns the covariance-update policy currently used by the filter.
    #[must_use]
    pub fn covariance_update(&self) -> CovarianceUpdate {
        self.covariance_update
    }

    /// Computes the prediction step for the supplied input without mutating the
    /// filter state.
    pub fn predict(&self, input: MatRef<'_, T>) -> Result<KalmanPrediction<T>, EstimatorError> {
        validate_column_vector("input", input, self.b.ncols())?;

        // Prediction uses the standard one-step propagation:
        //
        // `x^- = A x + B u`
        // `P^- = A P A^H + W`
        //
        // where `W` is interpreted directly in state coordinates.
        let state = dense_add(
            dense_mul(self.a.as_ref(), self.x_hat.as_ref()).as_ref(),
            dense_mul(self.b.as_ref(), input).as_ref(),
        );
        let covariance = dense_add(
            dense_mul_adjoint_rhs(
                dense_mul(self.a.as_ref(), self.p.as_ref()).as_ref(),
                self.a.as_ref(),
            )
            .as_ref(),
            self.w.as_ref(),
        );
        let mut covariance = covariance;
        hermitian_project_in_place(&mut covariance);
        let output = dense_add(
            dense_mul(self.c.as_ref(), state.as_ref()).as_ref(),
            dense_mul(self.d.as_ref(), input).as_ref(),
        );

        if !state.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "prediction.state",
            });
        }
        if !covariance.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "prediction.covariance",
            });
        }
        if !output.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "prediction.output",
            });
        }

        Ok(KalmanPrediction {
            state,
            covariance,
            output,
        })
    }

    /// Applies one measurement update to an externally supplied prediction.
    ///
    /// Unlike the monolithic [`step`](Self::step) path, the split API lets
    /// callers provide a measurement-side input that differs from the input
    /// used during [`predict`](Self::predict). To keep the innovation
    /// consistent with the supplied update input, this method recomputes the
    /// measurement-side predicted output `C x^- + D u`.
    pub fn update(
        &self,
        prediction: &KalmanPrediction<T>,
        input: MatRef<'_, T>,
        measurement: MatRef<'_, T>,
    ) -> Result<KalmanUpdate<T>, EstimatorError> {
        validate_column_vector("input", input, self.b.ncols())?;
        validate_column_vector("measurement", measurement, self.c.nrows())?;
        validate_column_vector(
            "prediction.state",
            prediction.state.as_ref(),
            self.a.nrows(),
        )?;
        validate_square(
            "prediction.covariance",
            prediction.covariance.as_ref(),
            self.a.nrows(),
        )?;
        validate_column_vector(
            "prediction.output",
            prediction.output.as_ref(),
            self.c.nrows(),
        )?;

        // The update stage forms the innovation
        //
        // `r = y - (C x^- + D u)`
        //
        // and innovation covariance
        //
        // `S = C P^- C^H + V`
        //
        // before solving for the Kalman gain.
        let predicted_output = dense_add(
            dense_mul(self.c.as_ref(), prediction.state.as_ref()).as_ref(),
            dense_mul(self.d.as_ref(), input).as_ref(),
        );
        let innovation = dense_sub(measurement, predicted_output.as_ref());
        let innovation_covariance = dense_add(
            dense_mul_adjoint_rhs(
                dense_mul(self.c.as_ref(), prediction.covariance.as_ref()).as_ref(),
                self.c.as_ref(),
            )
            .as_ref(),
            self.v.as_ref(),
        );
        let mut innovation_covariance = innovation_covariance;
        hermitian_project_in_place(&mut innovation_covariance);
        let cross = dense_mul_adjoint_rhs(prediction.covariance.as_ref(), self.c.as_ref());
        // `cross * S^-1` is the Kalman gain `K = P^- C^H S^-1`. The helper is
        // written as a right solve so the algebra stays close to that formula.
        let gain = solve_right_checked(
            cross.as_ref(),
            innovation_covariance.as_ref(),
            default_tolerance::<T>(),
            EstimatorError::SingularInnovationCovariance,
        )?;
        let state = dense_add(
            prediction.state.as_ref(),
            dense_mul(gain.as_ref(), innovation.as_ref()).as_ref(),
        );
        let covariance = updated_covariance(
            self.covariance_update,
            prediction.covariance.as_ref(),
            gain.as_ref(),
            self.c.as_ref(),
            self.v.as_ref(),
            innovation_covariance.as_ref(),
        );
        let mut covariance = covariance;
        hermitian_project_in_place(&mut covariance);
        let output = dense_add(
            dense_mul(self.c.as_ref(), state.as_ref()).as_ref(),
            dense_mul(self.d.as_ref(), input).as_ref(),
        );
        let innovation_norm = column_vector_norm(innovation.as_ref());
        let normalized_innovation_norm =
            normalized_innovation_norm(innovation.as_ref(), innovation_covariance.as_ref())?;

        if !predicted_output.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "update.predicted_output",
            });
        }
        if !innovation.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "update.innovation",
            });
        }
        if !gain.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "update.gain",
            });
        }
        if !state.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "update.state",
            });
        }
        if !covariance.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "update.covariance",
            });
        }
        if !output.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "update.output",
            });
        }

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

    /// Runs one full predict/update cycle and stores the posterior estimate.
    ///
    /// This monolithic entry point uses the same input for both prediction and
    /// measurement feedthrough. Callers that need a different update-side
    /// input should use the split [`predict`](Self::predict) and
    /// [`update`](Self::update) methods.
    pub fn step(
        &mut self,
        input: MatRef<'_, T>,
        measurement: MatRef<'_, T>,
    ) -> Result<KalmanUpdate<T>, EstimatorError> {
        // `step` is just the stateful convenience wrapper around the pure
        // predict/update stages above.
        let prediction = self.predict(input)?;
        let update = self.update(&prediction, input, measurement)?;
        self.x_hat = clone_mat(update.state.as_ref());
        self.p = clone_mat(update.covariance.as_ref());
        Ok(update)
    }
}

impl<T> SteadyStateKalmanFilter<T>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    /// Builds a fixed-gain steady-state discrete observer from explicit
    /// matrices.
    ///
    /// `gain` must be the filter-form correction gain `K` used in
    ///
    /// `x^+ = x^- + K (y - y^-)`
    ///
    /// rather than the predictor-form `DLQE` observer gain used in `A - L C`.
    pub fn new(
        a: Mat<T>,
        b: Mat<T>,
        c: Mat<T>,
        d: Mat<T>,
        gain: Mat<T>,
        x_hat: Mat<T>,
        steady_state_covariance: Option<Mat<T>>,
    ) -> Result<Self, EstimatorError> {
        validate_fixed_gain_observer_model(
            a.as_ref(),
            b.as_ref(),
            c.as_ref(),
            d.as_ref(),
            gain.as_ref(),
            x_hat.as_ref(),
        )?;
        if let Some(covariance) = &steady_state_covariance {
            validate_square("steady_state_covariance", covariance.as_ref(), a.nrows())?;
        }
        Ok(Self {
            a,
            b,
            c,
            d,
            gain,
            x_hat,
            steady_state_covariance,
        })
    }

    /// Builds a fixed-gain steady-state discrete observer from a validated
    /// discrete state-space model and an explicit filter-form correction gain.
    ///
    /// This constructor expects the discrete Kalman gain `K` that appears in
    ///
    /// `x^+ = x^- + K (y - y^-)`
    ///
    /// not the predictor-form `DLQE` gain used in `A - L C`.
    pub fn from_filter_gain(
        system: &DiscreteStateSpace<T>,
        gain: Mat<T>,
        x_hat: Mat<T>,
        steady_state_covariance: Option<Mat<T>>,
    ) -> Result<Self, EstimatorError> {
        Self::new(
            clone_mat(system.a()),
            clone_mat(system.b()),
            clone_mat(system.c()),
            clone_mat(system.d()),
            gain,
            x_hat,
            steady_state_covariance,
        )
    }

    /// Builds a fixed-gain steady-state discrete observer from `DLQE`.
    ///
    /// `DLQE` returns the steady-state a-priori covariance `P^-` together with
    /// the predictor-form observer dynamics `A - L C`. The runtime wrapper
    /// here uses explicit filter-form `predict` / `update` steps, so it forms
    /// the corresponding fixed filter gain directly from that prior
    /// covariance.
    ///
    /// This is the correct constructor when the gain originates from
    /// [`dlqe_dense`] or [`DiscreteStateSpace::dlqe`]. Use
    /// [`Self::from_filter_gain`] only when the supplied gain is already in the
    /// filter-form correction equation `x^+ = x^- + K (y - y^-)`.
    pub fn from_dlqe(
        system: &DiscreteStateSpace<T>,
        w: MatRef<'_, T>,
        v: MatRef<'_, T>,
        x_hat: Mat<T>,
    ) -> Result<Self, EstimatorError> {
        let solve = system.dlqe(w, v)?;
        let (gain, _) = steady_state_filter_gain(system.c(), v, solve.covariance.as_ref())?;
        Self::from_filter_gain(system, gain, x_hat, Some(solve.covariance))
    }

    /// Returns the current state estimate.
    #[must_use]
    pub fn state_estimate(&self) -> MatRef<'_, T> {
        self.x_hat.as_ref()
    }

    /// Returns the fixed observer gain.
    #[must_use]
    pub fn gain(&self) -> MatRef<'_, T> {
        self.gain.as_ref()
    }

    /// Returns the stored steady-state covariance when available.
    #[must_use]
    pub fn steady_state_covariance(&self) -> Option<MatRef<'_, T>> {
        self.steady_state_covariance.as_ref().map(|p| p.as_ref())
    }

    /// Computes the fixed-gain prediction step for the supplied input.
    ///
    /// This follows the same filter-form state propagation as the full
    /// discrete Kalman filter, but without covariance evolution:
    ///
    /// `x^- = A x + B u`
    pub fn predict(
        &self,
        input: MatRef<'_, T>,
    ) -> Result<SteadyStateKalmanPrediction<T>, EstimatorError> {
        validate_column_vector("input", input, self.b.ncols())?;
        let state = dense_add(
            dense_mul(self.a.as_ref(), self.x_hat.as_ref()).as_ref(),
            dense_mul(self.b.as_ref(), input).as_ref(),
        );
        let output = dense_add(
            dense_mul(self.c.as_ref(), state.as_ref()).as_ref(),
            dense_mul(self.d.as_ref(), input).as_ref(),
        );
        if !state.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "steady_state_prediction.state",
            });
        }
        if !output.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "steady_state_prediction.output",
            });
        }
        Ok(SteadyStateKalmanPrediction { state, output })
    }

    /// Applies one fixed-gain measurement correction to an externally supplied
    /// prediction.
    ///
    /// The update uses the stored constant gain:
    ///
    /// `x^+ = x^- + L (y - (C x^- + D u))`
    ///
    /// so this wrapper behaves like the converged steady-state version of the
    /// full discrete Kalman recursion. As with the full split Kalman API, the
    /// measurement-side predicted output is recomputed from the supplied
    /// update input so callers can use a different feedthrough context than
    /// the one used during prediction.
    pub fn update(
        &self,
        prediction: &SteadyStateKalmanPrediction<T>,
        input: MatRef<'_, T>,
        measurement: MatRef<'_, T>,
    ) -> Result<SteadyStateKalmanUpdate<T>, EstimatorError> {
        validate_column_vector("input", input, self.b.ncols())?;
        validate_column_vector("measurement", measurement, self.c.nrows())?;
        validate_column_vector(
            "prediction.state",
            prediction.state.as_ref(),
            self.a.nrows(),
        )?;
        validate_column_vector(
            "prediction.output",
            prediction.output.as_ref(),
            self.c.nrows(),
        )?;

        let predicted_output = dense_add(
            dense_mul(self.c.as_ref(), prediction.state.as_ref()).as_ref(),
            dense_mul(self.d.as_ref(), input).as_ref(),
        );
        let innovation = dense_sub(measurement, predicted_output.as_ref());
        let state = dense_add(
            prediction.state.as_ref(),
            dense_mul(self.gain.as_ref(), innovation.as_ref()).as_ref(),
        );
        let output = dense_add(
            dense_mul(self.c.as_ref(), state.as_ref()).as_ref(),
            dense_mul(self.d.as_ref(), input).as_ref(),
        );
        let innovation_norm = column_vector_norm(innovation.as_ref());

        if !predicted_output.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "steady_state_update.predicted_output",
            });
        }
        if !innovation.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "steady_state_update.innovation",
            });
        }
        if !state.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "steady_state_update.state",
            });
        }
        if !output.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "steady_state_update.output",
            });
        }

        Ok(SteadyStateKalmanUpdate {
            innovation,
            innovation_norm,
            state,
            output,
        })
    }

    /// Runs one full fixed-gain predict/update cycle and stores the posterior
    /// state estimate.
    ///
    /// This monolithic entry point uses the same input for both prediction and
    /// measurement feedthrough. Callers that need a different update-side
    /// input should use the split [`predict`](Self::predict) and
    /// [`update`](Self::update) methods.
    pub fn step(
        &mut self,
        input: MatRef<'_, T>,
        measurement: MatRef<'_, T>,
    ) -> Result<SteadyStateKalmanUpdate<T>, EstimatorError> {
        let prediction = self.predict(input)?;
        let update = self.update(&prediction, input, measurement)?;
        self.x_hat = clone_mat(update.state.as_ref());
        Ok(update)
    }
}

impl<T> ContinuousObserver<T>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    /// Builds a fixed-gain continuous observer from explicit matrices.
    pub fn new(
        a: Mat<T>,
        b: Mat<T>,
        c: Mat<T>,
        d: Mat<T>,
        gain: Mat<T>,
        x_hat: Mat<T>,
    ) -> Result<Self, EstimatorError> {
        validate_fixed_gain_observer_model(
            a.as_ref(),
            b.as_ref(),
            c.as_ref(),
            d.as_ref(),
            gain.as_ref(),
            x_hat.as_ref(),
        )?;
        Ok(Self {
            a,
            b,
            c,
            d,
            gain,
            x_hat,
        })
    }

    /// Builds a fixed-gain continuous observer from a validated state-space
    /// model and an explicit observer gain.
    pub fn from_gain(
        system: &ContinuousStateSpace<T>,
        gain: Mat<T>,
        x_hat: Mat<T>,
    ) -> Result<Self, EstimatorError> {
        Self::new(
            clone_mat(system.a()),
            clone_mat(system.b()),
            clone_mat(system.c()),
            clone_mat(system.d()),
            gain,
            x_hat,
        )
    }

    /// Builds a fixed-gain continuous observer from `LQE`.
    ///
    /// Unlike the discrete steady-state wrapper, this path does not need a
    /// predictor-to-filter gain conversion. The continuous `LQE` gain already
    /// appears directly in the observer differential equation.
    pub fn from_lqe(
        system: &ContinuousStateSpace<T>,
        w: MatRef<'_, T>,
        v: MatRef<'_, T>,
        x_hat: Mat<T>,
    ) -> Result<Self, EstimatorError> {
        let solve = system.lqe(w, v)?;
        Self::from_gain(system, solve.gain, x_hat)
    }

    /// Returns the current state estimate.
    #[must_use]
    pub fn state_estimate(&self) -> MatRef<'_, T> {
        self.x_hat.as_ref()
    }

    /// Returns the fixed observer gain.
    #[must_use]
    pub fn gain(&self) -> MatRef<'_, T> {
        self.gain.as_ref()
    }

    /// Evaluates the continuous observer differential equation for the current
    /// estimate and supplied signals.
    ///
    /// This returns the derivative data rather than stepping time internally so
    /// callers can integrate it with whatever ODE scheme they are already
    /// using.
    pub fn derivative(
        &self,
        input: MatRef<'_, T>,
        measurement: MatRef<'_, T>,
    ) -> Result<ContinuousObserverDerivative<T>, EstimatorError> {
        validate_column_vector("input", input, self.b.ncols())?;
        validate_column_vector("measurement", measurement, self.c.nrows())?;

        let output = dense_add(
            dense_mul(self.c.as_ref(), self.x_hat.as_ref()).as_ref(),
            dense_mul(self.d.as_ref(), input).as_ref(),
        );
        let innovation = dense_sub(measurement, output.as_ref());
        let state_derivative = dense_add(
            dense_add(
                dense_mul(self.a.as_ref(), self.x_hat.as_ref()).as_ref(),
                dense_mul(self.b.as_ref(), input).as_ref(),
            )
            .as_ref(),
            dense_mul(self.gain.as_ref(), innovation.as_ref()).as_ref(),
        );
        let innovation_norm = column_vector_norm(innovation.as_ref());

        if !output.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "continuous_observer.output",
            });
        }
        if !innovation.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "continuous_observer.innovation",
            });
        }
        if !state_derivative.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "continuous_observer.state_derivative",
            });
        }

        Ok(ContinuousObserverDerivative {
            output,
            innovation,
            innovation_norm,
            state_derivative,
        })
    }
}

fn validate_lqe_dims<T>(
    a: MatRef<'_, T>,
    c: MatRef<'_, T>,
    w: MatRef<'_, T>,
    v: MatRef<'_, T>,
) -> Result<(), EstimatorError> {
    // The first estimator API assumes process noise already acts in state
    // coordinates, so `W` is `n x n` and `V` is `p x p`.
    validate_square("a", a, a.nrows())?;
    validate_square("w", w, a.nrows())?;
    validate_square("v", v, c.nrows())?;
    if c.ncols() != a.ncols() {
        return Err(EstimatorError::DimensionMismatch {
            which: "c",
            expected_nrows: c.nrows(),
            expected_ncols: a.ncols(),
            actual_nrows: c.nrows(),
            actual_ncols: c.ncols(),
        });
    }
    Ok(())
}

fn validate_filter_model<T>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    c: MatRef<'_, T>,
    d: MatRef<'_, T>,
    w: MatRef<'_, T>,
    v: MatRef<'_, T>,
    x_hat: MatRef<'_, T>,
    p: MatRef<'_, T>,
) -> Result<(), EstimatorError> {
    let n = a.nrows();
    validate_square("a", a, n)?;
    validate_square("w", w, n)?;
    validate_square("p", p, n)?;
    if b.nrows() != n {
        return Err(EstimatorError::DimensionMismatch {
            which: "b",
            expected_nrows: n,
            expected_ncols: b.ncols(),
            actual_nrows: b.nrows(),
            actual_ncols: b.ncols(),
        });
    }
    if c.ncols() != n {
        return Err(EstimatorError::DimensionMismatch {
            which: "c",
            expected_nrows: c.nrows(),
            expected_ncols: n,
            actual_nrows: c.nrows(),
            actual_ncols: c.ncols(),
        });
    }
    if d.nrows() != c.nrows() || d.ncols() != b.ncols() {
        return Err(EstimatorError::DimensionMismatch {
            which: "d",
            expected_nrows: c.nrows(),
            expected_ncols: b.ncols(),
            actual_nrows: d.nrows(),
            actual_ncols: d.ncols(),
        });
    }
    validate_square("v", v, c.nrows())?;
    validate_column_vector("x_hat", x_hat, n)?;
    Ok(())
}

fn validate_fixed_gain_observer_model<T>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    c: MatRef<'_, T>,
    d: MatRef<'_, T>,
    gain: MatRef<'_, T>,
    x_hat: MatRef<'_, T>,
) -> Result<(), EstimatorError> {
    // Fixed-gain observers share the same `A/B/C/D` compatibility rules as the
    // full filter, but they replace covariance state with an explicit gain.
    let n = a.nrows();
    validate_square("a", a, n)?;
    if b.nrows() != n {
        return Err(EstimatorError::DimensionMismatch {
            which: "b",
            expected_nrows: n,
            expected_ncols: b.ncols(),
            actual_nrows: b.nrows(),
            actual_ncols: b.ncols(),
        });
    }
    if c.ncols() != n {
        return Err(EstimatorError::DimensionMismatch {
            which: "c",
            expected_nrows: c.nrows(),
            expected_ncols: n,
            actual_nrows: c.nrows(),
            actual_ncols: c.ncols(),
        });
    }
    if d.nrows() != c.nrows() || d.ncols() != b.ncols() {
        return Err(EstimatorError::DimensionMismatch {
            which: "d",
            expected_nrows: c.nrows(),
            expected_ncols: b.ncols(),
            actual_nrows: d.nrows(),
            actual_ncols: d.ncols(),
        });
    }
    validate_column_vector("x_hat", x_hat, n)?;
    if gain.nrows() != a.nrows() || gain.ncols() != c.nrows() {
        return Err(EstimatorError::DimensionMismatch {
            which: "gain",
            expected_nrows: a.nrows(),
            expected_ncols: c.nrows(),
            actual_nrows: gain.nrows(),
            actual_ncols: gain.ncols(),
        });
    }
    Ok(())
}

fn validate_square<T>(
    which: &'static str,
    matrix: MatRef<'_, T>,
    expected_dim: usize,
) -> Result<(), EstimatorError> {
    if matrix.nrows() != expected_dim || matrix.ncols() != expected_dim {
        return Err(EstimatorError::DimensionMismatch {
            which,
            expected_nrows: expected_dim,
            expected_ncols: expected_dim,
            actual_nrows: matrix.nrows(),
            actual_ncols: matrix.ncols(),
        });
    }
    Ok(())
}

fn validate_column_vector<T>(
    which: &'static str,
    matrix: MatRef<'_, T>,
    expected_nrows: usize,
) -> Result<(), EstimatorError> {
    if matrix.nrows() != expected_nrows || matrix.ncols() != 1 {
        return Err(EstimatorError::DimensionMismatch {
            which,
            expected_nrows,
            expected_ncols: 1,
            actual_nrows: matrix.nrows(),
            actual_ncols: matrix.ncols(),
        });
    }
    Ok(())
}

fn default_tolerance<T>() -> T::Real
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    T::Real::epsilon().sqrt()
}

fn estimator_matrix<T>(a: MatRef<'_, T>, l: MatRef<'_, T>, c: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    // Both continuous and discrete steady-state observer design report the
    // estimator dynamics in the same algebraic form `A - L C`.
    dense_sub(a, dense_mul(l, c).as_ref())
}

fn steady_state_filter_gain<T>(
    c: MatRef<'_, T>,
    v: MatRef<'_, T>,
    prior_covariance: MatRef<'_, T>,
) -> Result<(Mat<T>, Mat<T>), EstimatorError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    // The discrete-time Riccati solve already returns the steady-state
    // a-priori covariance `P^-`. The fixed-gain filter-form observer therefore
    // uses that covariance directly when forming `K = P^- C^H S^-1`.
    let mut innovation_covariance = dense_add(
        dense_mul_adjoint_rhs(dense_mul(c, prior_covariance).as_ref(), c).as_ref(),
        v,
    );
    hermitian_project_in_place(&mut innovation_covariance);

    let cross = dense_mul_adjoint_rhs(prior_covariance, c);
    let gain = solve_right_checked(
        cross.as_ref(),
        innovation_covariance.as_ref(),
        default_tolerance::<T>(),
        EstimatorError::SingularInnovationCovariance,
    )?;
    Ok((gain, innovation_covariance))
}

fn updated_covariance<T>(
    covariance_update: CovarianceUpdate,
    predicted_covariance: MatRef<'_, T>,
    gain: MatRef<'_, T>,
    c: MatRef<'_, T>,
    v: MatRef<'_, T>,
    innovation_covariance: MatRef<'_, T>,
) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    match covariance_update {
        CovarianceUpdate::Simple => dense_sub(
            predicted_covariance,
            dense_mul_adjoint_rhs(dense_mul(gain, innovation_covariance).as_ref(), gain).as_ref(),
        ),
        CovarianceUpdate::Joseph => {
            // The Joseph form is more expensive, but it is also the more
            // conservative floating-point update because it tends to preserve
            // Hermitian symmetry and positive semidefiniteness better than the
            // compact subtraction formula.
            let identity = identity::<T>(predicted_covariance.nrows());
            let kc = dense_mul(gain, c);
            let i_minus_kc = dense_sub(identity.as_ref(), kc.as_ref());
            let first = dense_mul_adjoint_rhs(
                dense_mul(i_minus_kc.as_ref(), predicted_covariance).as_ref(),
                i_minus_kc.as_ref(),
            );
            let second = dense_mul_adjoint_rhs(dense_mul(gain, v).as_ref(), gain);
            dense_add(first.as_ref(), second.as_ref())
        }
    }
}

fn normalized_innovation_norm<T>(
    innovation: MatRef<'_, T>,
    innovation_covariance: MatRef<'_, T>,
) -> Result<T::Real, EstimatorError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    // Whitening the innovation by `S^-1` gives the standard normalized
    // innovation energy used for runtime consistency checks.
    let whitened = solve_left_checked(
        innovation_covariance,
        innovation,
        default_tolerance::<T>(),
        EstimatorError::SingularInnovationCovariance,
    )?;
    Ok(inner_product_real(innovation, whitened.as_ref())
        .max(<T::Real as Zero>::zero())
        .sqrt())
}

/// Solves `lhs * X = rhs` and rejects numerically unusable results.
///
/// This is used for innovation-covariance solves in the runtime filter and for
/// the small dense dual-gain recovery steps.
fn solve_left_checked<T>(
    lhs: MatRef<'_, T>,
    rhs: MatRef<'_, T>,
    tol: T::Real,
    err: EstimatorError,
) -> Result<Mat<T>, EstimatorError>
where
    T: ComplexField + Copy,
    T::Real: Float + Copy,
{
    let solution = lhs.full_piv_lu().solve(rhs);
    if !solution.as_ref().is_all_finite() {
        return Err(err);
    }

    let residual = dense_sub_plain(dense_mul_plain(lhs, solution.as_ref()).as_ref(), rhs);
    let residual_norm = frobenius_norm_plain(residual.as_ref());
    let scale = frobenius_norm_plain(lhs) * frobenius_norm_plain(solution.as_ref())
        + frobenius_norm_plain(rhs);
    let one = <T::Real as One>::one();
    let threshold = scale.max(one) * tol * (one + one);
    if !residual_norm.is_finite() || residual_norm > threshold {
        return Err(err);
    }

    Ok(solution)
}

/// Solves `X * lhs = rhs` by transposing into [`solve_left_checked`].
///
/// The Kalman gain formula naturally appears as a right solve
/// `K S = P^- C^H`, so this wrapper keeps the calling code in that form.
fn solve_right_checked<T>(
    rhs_left: MatRef<'_, T>,
    lhs_right: MatRef<'_, T>,
    tol: T::Real,
    err: EstimatorError,
) -> Result<Mat<T>, EstimatorError>
where
    T: ComplexField + Copy,
    T::Real: Float + Copy,
{
    let lhs_t = dense_transpose(lhs_right);
    let rhs_t = dense_transpose(rhs_left);
    let solved_t = solve_left_checked(lhs_t.as_ref(), rhs_t.as_ref(), tol, err)?;
    Ok(dense_transpose(solved_t.as_ref()))
}

fn clone_mat<T: Copy>(matrix: MatRef<'_, T>) -> Mat<T> {
    Mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        matrix[(row, col)]
    })
}

/// Returns a dense identity matrix of the requested dimension.
fn identity<T>(dim: usize) -> Mat<T>
where
    T: ComplexField + Copy,
{
    Mat::from_fn(
        dim,
        dim,
        |row, col| {
            if row == col { T::one() } else { T::zero() }
        },
    )
}

fn dense_adjoint<T>(matrix: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    Mat::from_fn(matrix.ncols(), matrix.nrows(), |row, col| {
        matrix[(col, row)].conj()
    })
}

fn dense_transpose<T: Copy>(matrix: MatRef<'_, T>) -> Mat<T> {
    Mat::from_fn(matrix.ncols(), matrix.nrows(), |row, col| {
        matrix[(col, row)]
    })
}

fn dense_mul<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    Mat::from_fn(lhs.nrows(), rhs.ncols(), |row, col| {
        let mut acc = CompensatedSum::<T>::default();
        for k in 0..lhs.ncols() {
            acc.add(lhs[(row, k)] * rhs[(k, col)]);
        }
        acc.finish()
    })
}

fn dense_mul_adjoint_rhs<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    Mat::from_fn(lhs.nrows(), rhs.nrows(), |row, col| {
        let mut acc = CompensatedSum::<T>::default();
        for k in 0..lhs.ncols() {
            acc.add(lhs[(row, k)] * rhs[(col, k)].conj());
        }
        acc.finish()
    })
}

fn dense_add<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| {
        let mut acc = CompensatedSum::<T>::default();
        acc.add(lhs[(row, col)]);
        acc.add(rhs[(row, col)]);
        acc.finish()
    })
}

fn dense_sub<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| {
        let mut acc = CompensatedSum::<T>::default();
        acc.add(lhs[(row, col)]);
        acc.add(-rhs[(row, col)]);
        acc.finish()
    })
}

fn dense_mul_plain<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    Mat::from_fn(lhs.nrows(), rhs.ncols(), |row, col| {
        let mut acc = T::zero();
        for k in 0..lhs.ncols() {
            acc = acc + lhs[(row, k)] * rhs[(k, col)];
        }
        acc
    })
}

fn dense_sub_plain<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| {
        lhs[(row, col)] - rhs[(row, col)]
    })
}

fn inner_product_real<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> T::Real
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    // The normalized-innovation metric only needs the real scalar value of the
    // Hermitian inner product.
    let mut acc = CompensatedSum::<T>::default();
    for row in 0..lhs.nrows() {
        acc.add(lhs[(row, 0)].conj() * rhs[(row, 0)]);
    }
    acc.finish().real()
}

/// Returns the Euclidean norm of a dense column vector.
fn column_vector_norm<T>(vector: MatRef<'_, T>) -> T::Real
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let mut acc = <T::Real as Zero>::zero();
    for row in 0..vector.nrows() {
        acc = acc + vector[(row, 0)].abs2();
    }
    acc.sqrt()
}

/// Projects a dense matrix onto the Hermitian subspace in place.
///
/// This is used after covariance-like updates so small floating-point skew does
/// not accumulate into obviously non-Hermitian covariance matrices.
fn hermitian_project_in_place<T>(matrix: &mut Mat<T>)
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let one = <T::Real as One>::one();
    let half = one / (one + one);
    for col in 0..matrix.ncols() {
        for row in 0..=col {
            let avg = (matrix[(row, col)] + matrix[(col, row)].conj()).mul_real(half);
            matrix[(row, col)] = avg;
            matrix[(col, row)] = avg.conj();
        }
    }
}

fn frobenius_norm_plain<T>(matrix: MatRef<'_, T>) -> T::Real
where
    T: ComplexField + Copy,
    T::Real: Float + Copy,
{
    let mut acc = <T::Real as Zero>::zero();
    for col in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            acc = acc + matrix[(row, col)].abs2();
        }
    }
    acc.sqrt()
}

#[cfg(test)]
mod test {
    use super::{
        ContinuousObserver, CovarianceUpdate, DiscreteKalmanFilter, EstimatorError,
        SteadyStateKalmanFilter, dlqe_dense, lqe_dense,
    };
    use crate::control::lti::state_space::{ContinuousStateSpace, DiscreteStateSpace};
    use faer::Mat;
    use faer_traits::ext::ComplexFieldExt;

    fn assert_close<T>(lhs: &Mat<T>, rhs: &Mat<T>, tol: T::Real)
    where
        T: crate::sparse::compensated::CompensatedField,
        T::Real: num_traits::Float + Copy,
    {
        assert_eq!(lhs.nrows(), rhs.nrows());
        assert_eq!(lhs.ncols(), rhs.ncols());
        for col in 0..lhs.ncols() {
            for row in 0..lhs.nrows() {
                let err = (lhs[(row, col)] - rhs[(row, col)]).abs();
                assert!(
                    err <= tol,
                    "entry ({row}, {col}) mismatch: err={err:?}, tol={tol:?}",
                );
            }
        }
    }

    #[test]
    fn lqe_matches_scalar_dual_closed_form() {
        let a = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let c = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let w = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let v = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let solve = lqe_dense(a.as_ref(), c.as_ref(), w.as_ref(), v.as_ref()).unwrap();

        let expected = 1.0 + 2.0f64.sqrt();
        assert!((solve.covariance[(0, 0)] - expected).abs() < 1.0e-10);
        assert!((solve.gain[(0, 0)] - expected).abs() < 1.0e-10);
        assert!((solve.estimator_a[(0, 0)] + 2.0f64.sqrt()).abs() < 1.0e-10);
        assert!(solve.stabilizing);
    }

    #[test]
    fn dlqe_matches_scalar_dual_closed_form() {
        let a = Mat::from_fn(1, 1, |_, _| 1.2f64);
        let c = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let w = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let v = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let solve = dlqe_dense(a.as_ref(), c.as_ref(), w.as_ref(), v.as_ref()).unwrap();

        let p = (1.44 + (1.44f64 * 1.44 + 4.0).sqrt()) / 2.0;
        let expected_l = 1.2 * p / (1.0 + p);
        assert!((solve.covariance[(0, 0)] - p).abs() < 1.0e-10);
        assert!((solve.gain[(0, 0)] - expected_l).abs() < 1.0e-10);
        assert!((solve.estimator_a[(0, 0)] - (1.2 - expected_l)).abs() < 1.0e-10);
        assert!(solve.stabilizing);
    }

    #[test]
    fn lqe_and_dlqe_state_space_methods_match_free_functions() {
        let a = Mat::from_fn(
            2,
            2,
            |row, col| if row == col { 1.0 + row as f64 } else { 0.0 },
        );
        let b = Mat::zeros(2, 1);
        let c = Mat::from_fn(2, 2, |row, col| if row == col { 1.0 } else { 0.0 });
        let d = Mat::zeros(2, 1);
        let w = Mat::from_fn(
            2,
            2,
            |row, col| if row == col { 1.0 + row as f64 } else { 0.0 },
        );
        let v = Mat::from_fn(2, 2, |row, col| if row == col { 1.0 } else { 0.0 });

        let continuous =
            ContinuousStateSpace::new(a.clone(), b.clone(), c.clone(), d.clone()).unwrap();
        let discrete = DiscreteStateSpace::new(a.clone(), b, c.clone(), d, 0.1).unwrap();

        let free_lqe = lqe_dense(a.as_ref(), c.as_ref(), w.as_ref(), v.as_ref()).unwrap();
        let method_lqe = continuous.lqe(w.as_ref(), v.as_ref()).unwrap();
        assert_close(&free_lqe.gain, &method_lqe.gain, 1.0e-12);

        let free_dlqe = dlqe_dense(a.as_ref(), c.as_ref(), w.as_ref(), v.as_ref()).unwrap();
        let method_dlqe = discrete.dlqe(w.as_ref(), v.as_ref()).unwrap();
        assert_close(&free_dlqe.gain, &method_dlqe.gain, 1.0e-12);
    }

    #[test]
    fn discrete_kalman_predict_update_matches_scalar_reference() {
        let a = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let b = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let c = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let d = Mat::from_fn(1, 1, |_, _| 0.0f64);
        let w = Mat::from_fn(1, 1, |_, _| 0.25f64);
        let v = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let x0 = Mat::from_fn(1, 1, |_, _| 0.0f64);
        let p0 = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let filter = DiscreteKalmanFilter::new(a, b, c, d, w, v, x0, p0).unwrap();

        let u = Mat::from_fn(1, 1, |_, _| 2.0f64);
        let pred = filter.predict(u.as_ref()).unwrap();
        assert!((pred.state[(0, 0)] - 2.0).abs() < 1.0e-12);
        assert!((pred.covariance[(0, 0)] - 1.25).abs() < 1.0e-12);
        assert!((pred.output[(0, 0)] - 2.0).abs() < 1.0e-12);

        let y = Mat::from_fn(1, 1, |_, _| 1.5f64);
        let update = filter.update(&pred, u.as_ref(), y.as_ref()).unwrap();
        let expected_k = 1.25 / 2.25;
        let expected_x = 2.0 + expected_k * (1.5 - 2.0);
        let expected_p = 1.25 - expected_k * 2.25 * expected_k;
        assert!((update.gain[(0, 0)] - expected_k).abs() < 1.0e-12);
        assert!((update.state[(0, 0)] - expected_x).abs() < 1.0e-12);
        assert!((update.covariance[(0, 0)] - expected_p).abs() < 1.0e-12);
        assert!((update.predicted_output[(0, 0)] - 2.0).abs() < 1.0e-12);
        assert!((update.output[(0, 0)] - expected_x).abs() < 1.0e-12);
        assert!((update.innovation_norm - 0.5).abs() < 1.0e-12);
        assert!(update.normalized_innovation_norm.is_finite());
    }

    #[test]
    fn discrete_kalman_step_updates_internal_state() {
        let system = DiscreteStateSpace::new(
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            1.0,
        )
        .unwrap();
        let mut filter = DiscreteKalmanFilter::from_state_space(
            &system,
            Mat::from_fn(1, 1, |_, _| 0.1f64),
            Mat::from_fn(1, 1, |_, _| 0.2f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
        )
        .unwrap();

        let u = Mat::from_fn(1, 1, |_, _| 0.0f64);
        let y = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let update = filter.step(u.as_ref(), y.as_ref()).unwrap();
        assert_close(
            &super::clone_mat(filter.state_estimate()),
            &update.state,
            1.0e-12,
        );
        assert_close(
            &super::clone_mat(filter.covariance()),
            &update.covariance,
            1.0e-12,
        );
    }

    #[test]
    fn discrete_kalman_split_update_recomputes_feedthrough_from_update_input() {
        let filter_a = DiscreteKalmanFilter::new(
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 2.0f64),
            Mat::from_fn(1, 1, |_, _| 0.25f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.5f64),
        )
        .unwrap();
        let mut filter_b = DiscreteKalmanFilter::new(
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 2.0f64),
            Mat::from_fn(1, 1, |_, _| 0.25f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.5f64),
        )
        .unwrap();

        let predict_input = Mat::from_fn(1, 1, |_, _| 0.0f64);
        let update_input = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let measurement = Mat::from_fn(1, 1, |_, _| 3.5f64);

        let prediction = filter_a.predict(predict_input.as_ref()).unwrap();
        let split_update = filter_a
            .update(&prediction, update_input.as_ref(), measurement.as_ref())
            .unwrap();
        let step_update = filter_b
            .step(update_input.as_ref(), measurement.as_ref())
            .unwrap();

        assert_close(
            &split_update.predicted_output,
            &step_update.predicted_output,
            1.0e-12,
        );
        assert_close(&split_update.state, &step_update.state, 1.0e-12);
        assert_close(&split_update.covariance, &step_update.covariance, 1.0e-12);
    }

    #[test]
    fn discrete_kalman_rejects_singular_innovation_covariance() {
        let filter = DiscreteKalmanFilter::new(
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
        )
        .unwrap();
        let u = Mat::from_fn(1, 1, |_, _| 0.0f64);
        let pred = filter.predict(u.as_ref()).unwrap();
        let y = Mat::from_fn(1, 1, |_, _| 0.0f64);
        let err = filter.update(&pred, u.as_ref(), y.as_ref()).unwrap_err();
        assert!(matches!(err, EstimatorError::SingularInnovationCovariance));
    }

    #[test]
    fn joseph_and_simple_updates_match_on_scalar_problem() {
        let a = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let b = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let c = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let d = Mat::from_fn(1, 1, |_, _| 0.0f64);
        let w = Mat::from_fn(1, 1, |_, _| 0.25f64);
        let v = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let x0 = Mat::from_fn(1, 1, |_, _| 0.0f64);
        let p0 = Mat::from_fn(1, 1, |_, _| 1.0f64);

        let simple = DiscreteKalmanFilter::new_with_covariance_update(
            a.clone(),
            b.clone(),
            c.clone(),
            d.clone(),
            w.clone(),
            v.clone(),
            x0.clone(),
            p0.clone(),
            CovarianceUpdate::Simple,
        )
        .unwrap();
        let joseph = DiscreteKalmanFilter::new_with_covariance_update(
            a,
            b,
            c,
            d,
            w,
            v,
            x0,
            p0,
            CovarianceUpdate::Joseph,
        )
        .unwrap();

        let u = Mat::from_fn(1, 1, |_, _| 2.0f64);
        let y = Mat::from_fn(1, 1, |_, _| 1.5f64);
        let pred_simple = simple.predict(u.as_ref()).unwrap();
        let pred_joseph = joseph.predict(u.as_ref()).unwrap();
        let update_simple = simple.update(&pred_simple, u.as_ref(), y.as_ref()).unwrap();
        let update_joseph = joseph.update(&pred_joseph, u.as_ref(), y.as_ref()).unwrap();

        assert_close(&update_simple.state, &update_joseph.state, 1.0e-12);
        assert_close(
            &update_simple.covariance,
            &update_joseph.covariance,
            1.0e-12,
        );
    }

    #[test]
    fn steady_state_discrete_filter_matches_dlqe_gain_and_full_filter_init() {
        let system = DiscreteStateSpace::new(
            Mat::from_fn(1, 1, |_, _| 0.8f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            0.1,
        )
        .unwrap();
        let w = Mat::from_fn(1, 1, |_, _| 0.2f64);
        let v = Mat::from_fn(1, 1, |_, _| 0.3f64);
        let x0 = Mat::from_fn(1, 1, |_, _| 0.0f64);
        let solve = system.dlqe(w.as_ref(), v.as_ref()).unwrap();

        let steady =
            SteadyStateKalmanFilter::from_dlqe(&system, w.as_ref(), v.as_ref(), x0.clone())
                .unwrap();
        let full = DiscreteKalmanFilter::from_dlqe(&system, w.as_ref(), v.as_ref(), x0).unwrap();

        assert_close(
            steady.steady_state_covariance.as_ref().unwrap(),
            &solve.covariance,
            1.0e-12,
        );

        let u = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let y = Mat::from_fn(1, 1, |_, _| 0.25f64);
        let full_prediction = full.predict(u.as_ref()).unwrap();
        assert_close(&full_prediction.covariance, &solve.covariance, 1.0e-12);

        let steady_update = {
            let pred = steady.predict(u.as_ref()).unwrap();
            steady.update(&pred, u.as_ref(), y.as_ref()).unwrap()
        };
        let full_update = full
            .update(&full_prediction, u.as_ref(), y.as_ref())
            .unwrap();

        assert_close(&super::clone_mat(steady.gain()), &full_update.gain, 1.0e-10);
        assert_close(
            &super::clone_mat(full.covariance()),
            &full_update.covariance,
            1.0e-10,
        );
        assert_close(&steady_update.state, &full_update.state, 1.0e-10);
        assert_close(&steady_update.output, &full_update.output, 1.0e-10);
    }

    #[test]
    fn steady_state_from_filter_gain_matches_dlqe_when_given_filter_gain() {
        let system = DiscreteStateSpace::new(
            Mat::from_fn(1, 1, |_, _| 0.8f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            0.1,
        )
        .unwrap();
        let w = Mat::from_fn(1, 1, |_, _| 0.2f64);
        let v = Mat::from_fn(1, 1, |_, _| 0.3f64);
        let x0 = Mat::from_fn(1, 1, |_, _| 0.0f64);
        let solve = system.dlqe(w.as_ref(), v.as_ref()).unwrap();
        let (filter_gain, _) =
            super::steady_state_filter_gain(system.c(), v.as_ref(), solve.covariance.as_ref())
                .unwrap();

        let from_filter_gain = SteadyStateKalmanFilter::from_filter_gain(
            &system,
            filter_gain,
            x0.clone(),
            Some(solve.covariance.clone()),
        )
        .unwrap();
        let from_dlqe =
            SteadyStateKalmanFilter::from_dlqe(&system, w.as_ref(), v.as_ref(), x0).unwrap();

        let u = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let y = Mat::from_fn(1, 1, |_, _| 0.25f64);
        let update_from_filter_gain = {
            let prediction = from_filter_gain.predict(u.as_ref()).unwrap();
            from_filter_gain
                .update(&prediction, u.as_ref(), y.as_ref())
                .unwrap()
        };
        let update_from_dlqe = {
            let prediction = from_dlqe.predict(u.as_ref()).unwrap();
            from_dlqe
                .update(&prediction, u.as_ref(), y.as_ref())
                .unwrap()
        };

        assert_close(
            &update_from_filter_gain.state,
            &update_from_dlqe.state,
            1.0e-10,
        );
        assert_close(
            &update_from_filter_gain.output,
            &update_from_dlqe.output,
            1.0e-10,
        );
    }

    #[test]
    fn steady_state_split_update_recomputes_feedthrough_from_update_input() {
        let system = DiscreteStateSpace::new(
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 2.0f64),
            1.0,
        )
        .unwrap();
        let filter_a = SteadyStateKalmanFilter::from_filter_gain(
            &system,
            Mat::from_fn(1, 1, |_, _| 0.25f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            None,
        )
        .unwrap();
        let mut filter_b = SteadyStateKalmanFilter::from_filter_gain(
            &system,
            Mat::from_fn(1, 1, |_, _| 0.25f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            None,
        )
        .unwrap();

        let predict_input = Mat::from_fn(1, 1, |_, _| 0.0f64);
        let update_input = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let measurement = Mat::from_fn(1, 1, |_, _| 3.5f64);

        let prediction = filter_a.predict(predict_input.as_ref()).unwrap();
        let split_update = filter_a
            .update(&prediction, update_input.as_ref(), measurement.as_ref())
            .unwrap();
        let step_update = filter_b
            .step(update_input.as_ref(), measurement.as_ref())
            .unwrap();

        assert_close(&split_update.state, &step_update.state, 1.0e-12);
        assert_close(&split_update.output, &step_update.output, 1.0e-12);
    }

    #[test]
    fn continuous_observer_derivative_matches_manual_scalar_formula() {
        let system = ContinuousStateSpace::new(
            Mat::from_fn(1, 1, |_, _| 2.0f64),
            Mat::from_fn(1, 1, |_, _| 3.0f64),
            Mat::from_fn(1, 1, |_, _| 4.0f64),
            Mat::from_fn(1, 1, |_, _| 5.0f64),
        )
        .unwrap();
        let observer = ContinuousObserver::from_gain(
            &system,
            Mat::from_fn(1, 1, |_, _| 6.0f64),
            Mat::from_fn(1, 1, |_, _| 7.0f64),
        )
        .unwrap();

        let u = Mat::from_fn(1, 1, |_, _| 0.5f64);
        let y = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let deriv = observer.derivative(u.as_ref(), y.as_ref()).unwrap();

        let expected_output = 4.0 * 7.0 + 5.0 * 0.5;
        let expected_innovation = 1.0 - expected_output;
        let expected_xdot = 2.0 * 7.0 + 3.0 * 0.5 + 6.0 * expected_innovation;

        assert!((deriv.output[(0, 0)] - expected_output).abs() < 1.0e-12);
        assert!((deriv.innovation[(0, 0)] - expected_innovation).abs() < 1.0e-12);
        assert!((deriv.state_derivative[(0, 0)] - expected_xdot).abs() < 1.0e-12);
        assert!((deriv.innovation_norm - expected_innovation.abs()).abs() < 1.0e-12);
    }

    #[test]
    fn state_space_observer_and_steady_state_methods_build_wrappers() {
        let continuous = ContinuousStateSpace::new(
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
        )
        .unwrap();
        let discrete = DiscreteStateSpace::new(
            Mat::from_fn(1, 1, |_, _| 0.8f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            0.1,
        )
        .unwrap();
        let w = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let v = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let x0 = Mat::from_fn(1, 1, |_, _| 0.0f64);

        let observer = continuous
            .observer(w.as_ref(), v.as_ref(), x0.clone())
            .unwrap();
        let steady = discrete
            .steady_state_kalman(w.as_ref(), v.as_ref(), x0)
            .unwrap();

        assert_eq!(observer.gain.nrows(), 1);
        assert_eq!(steady.gain.nrows(), 1);
        assert!(steady.steady_state_covariance.is_some());
    }
}
