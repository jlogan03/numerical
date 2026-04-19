//! Dynamic-size nonlinear estimators for `embedded::alloc`.

mod ekf;
mod ukf;

pub use ekf::{
    DiscreteExtendedKalmanModel, DiscreteNonlinearModel, ExtendedKalmanFilter,
    ExtendedKalmanPrediction, ExtendedKalmanUpdate,
};
pub use ukf::{UnscentedKalmanFilter, UnscentedKalmanPrediction, UnscentedKalmanUpdate};
