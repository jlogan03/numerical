//! Dynamic-size nonlinear estimators for `embedded::alloc`.

use crate::control::estimation::NonlinearEstimatorError;
use crate::embedded::EmbeddedError;

mod dense;
mod ekf;
mod ukf;

fn map_nonlinear_error(error: NonlinearEstimatorError) -> EmbeddedError {
    match error {
        NonlinearEstimatorError::DimensionMismatch {
            which,
            expected_nrows,
            expected_ncols,
            actual_nrows,
            actual_ncols,
        }
        | NonlinearEstimatorError::InvalidModelOutput {
            which,
            expected_nrows,
            expected_ncols,
            actual_nrows,
            actual_ncols,
        } => EmbeddedError::DimensionMismatch {
            which,
            expected_rows: expected_nrows,
            expected_cols: expected_ncols,
            actual_rows: actual_nrows,
            actual_cols: actual_ncols,
        },
        NonlinearEstimatorError::SingularInnovationCovariance => EmbeddedError::SingularMatrix {
            which: "embedded.alloc.estimation.innovation_covariance",
        },
        NonlinearEstimatorError::SingularPredictedCovariance => EmbeddedError::SingularMatrix {
            which: "embedded.alloc.estimation.predicted_covariance",
        },
        NonlinearEstimatorError::NonPositiveDefiniteCovariance { which } => {
            EmbeddedError::NonPositiveDefinite { which }
        }
        NonlinearEstimatorError::InvalidUnscentedParams { which }
        | NonlinearEstimatorError::InvalidSigmaPointSet { which } => {
            EmbeddedError::InvalidParameter { which }
        }
        NonlinearEstimatorError::NonFiniteResult { which } => {
            EmbeddedError::NonFiniteValue { which }
        }
    }
}

pub use ekf::{
    DiscreteExtendedKalmanModel, DiscreteNonlinearModel, ExtendedKalmanFilter,
    ExtendedKalmanPrediction, ExtendedKalmanUpdate,
};
pub use ukf::{UnscentedKalmanFilter, UnscentedKalmanPrediction, UnscentedKalmanUpdate};
