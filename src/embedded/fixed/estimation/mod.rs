//! Fixed-size embedded estimation kernels.

mod kalman;

pub use kalman::{
    CovarianceUpdate, DiscreteKalmanFilter, KalmanPrediction, KalmanUpdate, SteadyStateKalmanFilter,
};
