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

#[path = "ekf.rs"]
mod ekf;
#[path = "ukf.rs"]
mod ukf;

pub use ekf::ExtendedKalmanFilter;
pub use ukf::UnscentedKalmanFilter;

use alloc::{boxed::Box, vec::Vec};
use core::fmt;
use faer::{Mat, MatRef};
use faer_traits::ComplexField;
use num_traits::{Float, NumCast};

use crate::sparse::compensated::CompensatedField;

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

impl core::error::Error for NonlinearEstimatorError {}

pub(super) fn validate_nonlinear_filter_model<R>(
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
    validate_column_vector("x_hat", x_hat, nstates)?;
    validate_square("p", p, nstates)?;
    let _ = ninputs;
    Ok(())
}

pub(super) fn validate_prediction<R>(
    prediction: &NonlinearKalmanPrediction<R>,
    nstates: usize,
    noutputs: usize,
    which: &'static str,
) -> Result<(), NonlinearEstimatorError>
where
    R: CompensatedField,
    R::Real: Float + Copy,
{
    validate_column_vector(which, prediction.state.as_ref(), nstates)?;
    validate_square(which, prediction.covariance.as_ref(), nstates)?;
    validate_column_vector(which, prediction.output.as_ref(), noutputs)
}

pub(super) fn validate_square<R>(
    which: &'static str,
    matrix: MatRef<'_, R>,
    expected_nrows: usize,
) -> Result<(), NonlinearEstimatorError> {
    if matrix.nrows() == expected_nrows && matrix.ncols() == expected_nrows {
        Ok(())
    } else {
        Err(NonlinearEstimatorError::DimensionMismatch {
            which,
            expected_nrows,
            expected_ncols: expected_nrows,
            actual_nrows: matrix.nrows(),
            actual_ncols: matrix.ncols(),
        })
    }
}

pub(super) fn validate_rect<R>(
    which: &'static str,
    matrix: MatRef<'_, R>,
    expected_nrows: usize,
    expected_ncols: usize,
) -> Result<(), NonlinearEstimatorError> {
    if matrix.nrows() == expected_nrows && matrix.ncols() == expected_ncols {
        Ok(())
    } else {
        Err(NonlinearEstimatorError::DimensionMismatch {
            which,
            expected_nrows,
            expected_ncols,
            actual_nrows: matrix.nrows(),
            actual_ncols: matrix.ncols(),
        })
    }
}

pub(super) fn validate_column_vector<R>(
    which: &'static str,
    vector: MatRef<'_, R>,
    expected_nrows: usize,
) -> Result<(), NonlinearEstimatorError> {
    validate_rect(which, vector, expected_nrows, 1)
}

pub(super) fn validate_model_output<R>(
    which: &'static str,
    matrix: MatRef<'_, R>,
    expected_nrows: usize,
    expected_ncols: usize,
) -> Result<(), NonlinearEstimatorError> {
    if matrix.nrows() == expected_nrows && matrix.ncols() == expected_ncols {
        Ok(())
    } else {
        Err(NonlinearEstimatorError::InvalidModelOutput {
            which,
            expected_nrows,
            expected_ncols,
            actual_nrows: matrix.nrows(),
            actual_ncols: matrix.ncols(),
        })
    }
}

pub(super) fn validate_finite<R>(
    which: &'static str,
    matrix: MatRef<'_, R>,
) -> Result<(), NonlinearEstimatorError>
where
    R: ComplexField,
{
    if matrix.is_all_finite() {
        Ok(())
    } else {
        Err(NonlinearEstimatorError::NonFiniteResult { which })
    }
}
