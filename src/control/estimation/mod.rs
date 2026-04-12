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

pub mod linear;
pub mod nonlinear;

pub use linear::{
    ContinuousObserver, ContinuousObserverDerivative, CovarianceUpdate, DiscreteKalmanFilter,
    EstimatorError, KalmanPrediction, KalmanUpdate, LqeSolve, SteadyStateKalmanFilter,
    SteadyStateKalmanPrediction, SteadyStateKalmanUpdate, dlqe_dense, lqe_dense,
};
pub use nonlinear::{
    DiscreteExtendedKalmanModel, DiscreteNonlinearModel, ExtendedKalmanFilter,
    NonlinearEstimatorError, NonlinearKalmanPrediction, NonlinearKalmanUpdate, SigmaPointProvider,
    SigmaPointSet, SigmaPointStrategy, UkfStage, UnscentedKalmanFilter, UnscentedParams,
};
