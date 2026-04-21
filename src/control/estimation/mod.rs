//! Estimator design and runtime filtering.
//!
//! This subsystem groups the crate's linear and nonlinear estimator features:
//!
//! - [`linear`] covers `LQE` / `DLQE`, discrete linear Kalman filtering, and
//!   fixed-gain observer runtimes
//! - [`nonlinear`] covers discrete-time `EKF` / `UKF` with additive-noise
//!   models and customizable UKF sigma-point placement
//!
//! The shared [`crate::control::estimation`] namespace re-exports the main
//! types and entry points from both layers so most callers do not need to care
//! which submodule owns a specific implementation.
//!
//! # Two Intuitions
//!
//! 1. **Information-flow view.** Estimation turns noisy outputs and known
//!    inputs into a best current guess of the hidden system state.
//! 2. **Dual-control view.** Linear estimator design mirrors controller
//!    synthesis on the transposed system: `LQE` is the observer-side dual of
//!    `LQR`, while EKF and UKF extend the same predict/update structure to
//!    nonlinear models.
//!
//! # Glossary
//!
//! - **Observer gain `L`:** Injection gain in `A - L C`.
//! - **Innovation:** Measurement residual `y - y^-`.
//! - **Joseph update:** Numerically robust covariance update formula.
//! - **Steady-state filter:** Fixed-gain observer obtained after covariance
//!   convergence.
//! - **Sigma points:** Deterministic sample set used by the UKF.
//!
//! # Mathematical Formulation
//!
//! Linear estimators use:
//!
//! - prediction: `x^- = A x + B u`
//! - update: `x^+ = x^- + L (y - C x^- - D u)`
//!
//! Nonlinear estimators replace the linear maps by:
//!
//! - `x^- = f(x, u)`
//! - `y^- = h(x, u)`
//!
//! and then compute an EKF linearization or UKF sigma-point approximation for
//! the covariance update.
//!
//! # Implementation Notes
//!
//! - The nonlinear surface is discrete-time and additive-noise only.
//! - Linear design and runtime live together so the returned gains can be used
//!   directly by convenience observers.
//! - UKF sigma-point placement is pluggable for users who need to avoid model
//!   discontinuities.
//!
//! # Feature Matrix
//!
//! | Feature | Continuous | Discrete | Linear | Nonlinear |
//! | --- | --- | --- | --- | --- |
//! | `LQE` / `DLQE` design | yes | yes | yes | no |
//! | Time-varying Kalman runtime | no | yes | yes | no |
//! | Fixed-gain observer runtime | yes | yes | yes | no |
//! | EKF | no | yes | no | yes |
//! | UKF | no | yes | no | yes |

mod dense;
pub mod linear;
pub mod nonlinear;
pub(crate) mod nonlinear_core;

pub use linear::{
    ContinuousObserver, ContinuousObserverDerivative, CovarianceUpdate, DiscreteKalmanFilter,
    EstimatorError, KalmanPrediction, KalmanUpdate, LqeSolve, SteadyStateKalmanFilter,
    SteadyStateKalmanPrediction, SteadyStateKalmanUpdate, dlqe_dense, lqe_dense,
    steady_state_filter_gain_dense,
};
pub use nonlinear::{
    DiscreteExtendedKalmanModel, DiscreteNonlinearModel, ExtendedKalmanFilter,
    NonlinearEstimatorError, NonlinearKalmanPrediction, NonlinearKalmanUpdate, SigmaPointProvider,
    SigmaPointSet, SigmaPointStrategy, UkfStage, UnscentedKalmanFilter, UnscentedParams,
};
