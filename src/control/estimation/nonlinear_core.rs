//! Shared alloc-only EKF/UKF algebra helpers.
//!
//! This module centralizes the standard predict/update matrix algebra used by
//! both the rich control-side nonlinear estimators and the simplified
//! `embedded::alloc` wrappers.

use super::CovarianceUpdate;
use super::dense::{default_tolerance, solve_left_checked};
use super::nonlinear::{NonlinearEstimatorError, UnscentedParams};
use crate::control::dense_ops::{
    dense_mul, dense_mul_adjoint_rhs, hermitian_project_in_place, inner_product_real,
};
use crate::sparse::compensated::{CompensatedField, CompensatedSum};
use faer::{Mat, MatRef};
use faer_traits::RealField;
use faer_traits::ext::ComplexFieldExt;
use num_traits::{Float, NumCast, Zero};

/// Validates the standard scaled-unscented-transform parameters.
pub(crate) fn validate_unscented_params<R>(
    params: UnscentedParams<R>,
    nstates: usize,
) -> Result<(), NonlinearEstimatorError>
where
    R: Float + Copy + NumCast + RealField,
{
    if nstates == 0 {
        return Err(NonlinearEstimatorError::InvalidUnscentedParams { which: "nstates" });
    }
    if !params.alpha.is_finite() || params.alpha <= R::zero() {
        return Err(NonlinearEstimatorError::InvalidUnscentedParams { which: "alpha" });
    }
    if !params.beta.is_finite() || params.beta < R::zero() {
        return Err(NonlinearEstimatorError::InvalidUnscentedParams { which: "beta" });
    }
    if !params.kappa.is_finite() {
        return Err(NonlinearEstimatorError::InvalidUnscentedParams { which: "kappa" });
    }
    let n: R = NumCast::from(nstates).unwrap();
    let lambda = params.alpha * params.alpha * (n + params.kappa) - n;
    let scaling = n + lambda;
    if !scaling.is_finite() || scaling <= R::zero() {
        return Err(NonlinearEstimatorError::InvalidUnscentedParams {
            which: "n_plus_lambda",
        });
    }
    Ok(())
}

/// Propagates covariance through one explicit linearization.
pub(crate) fn predict_covariance<R>(f: MatRef<'_, R>, p: MatRef<'_, R>, q: MatRef<'_, R>) -> Mat<R>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    let mut covariance = dense_mul_adjoint_rhs(dense_mul(f, p).as_ref(), f) + q;
    hermitian_project_in_place(&mut covariance);
    covariance
}

/// Reconstructs a weighted mean from sigma points stored columnwise.
pub(crate) fn weighted_mean<R>(points: MatRef<'_, R>, weights: &[R]) -> Mat<R>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    Mat::from_fn(points.nrows(), 1, |row, _| {
        let mut acc = CompensatedSum::<R>::default();
        for col in 0..points.ncols() {
            acc.add(points[(row, col)] * weights[col]);
        }
        acc.finish()
    })
}

/// Reconstructs a covariance-like second moment from centered sigma points.
pub(crate) fn weighted_covariance<R>(
    points: MatRef<'_, R>,
    mean: MatRef<'_, R>,
    weights: &[R],
) -> Mat<R>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    Mat::from_fn(points.nrows(), points.nrows(), |row, col| {
        let mut acc = CompensatedSum::<R>::default();
        for k in 0..points.ncols() {
            let dr = points[(row, k)] - mean[(row, 0)];
            let dc = points[(col, k)] - mean[(col, 0)];
            acc.add(dr * dc.conj() * weights[k]);
        }
        acc.finish()
    })
}

/// Reconstructs the state/measurement cross covariance used in the UKF gain.
pub(crate) fn weighted_cross_covariance<R>(
    lhs_points: MatRef<'_, R>,
    lhs_mean: MatRef<'_, R>,
    rhs_points: MatRef<'_, R>,
    rhs_mean: MatRef<'_, R>,
    weights: &[R],
) -> Mat<R>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    Mat::from_fn(lhs_points.nrows(), rhs_points.nrows(), |row, col| {
        let mut acc = CompensatedSum::<R>::default();
        for k in 0..lhs_points.ncols() {
            let dl = lhs_points[(row, k)] - lhs_mean[(row, 0)];
            let dr = rhs_points[(col, k)] - rhs_mean[(col, 0)];
            acc.add(dl * dr.conj() * weights[k]);
        }
        acc.finish()
    })
}

/// Applies the EKF covariance update using either the simple or Joseph form.
pub(crate) fn updated_covariance<R>(
    covariance_update: CovarianceUpdate,
    predicted_covariance: MatRef<'_, R>,
    gain: MatRef<'_, R>,
    h: MatRef<'_, R>,
    r: MatRef<'_, R>,
    innovation_covariance: MatRef<'_, R>,
) -> Mat<R>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    match covariance_update {
        CovarianceUpdate::Simple => {
            predicted_covariance.to_owned()
                - dense_mul_adjoint_rhs(dense_mul(gain, innovation_covariance).as_ref(), gain)
                    .as_ref()
        }
        CovarianceUpdate::Joseph => {
            let identity =
                Mat::<R>::identity(predicted_covariance.nrows(), predicted_covariance.nrows());
            let kh = dense_mul(gain, h);
            let i_minus_kh = identity - kh.as_ref();
            let first = dense_mul_adjoint_rhs(
                dense_mul(i_minus_kh.as_ref(), predicted_covariance).as_ref(),
                i_minus_kh.as_ref(),
            );
            let second = dense_mul_adjoint_rhs(dense_mul(gain, r).as_ref(), gain);
            first + second.as_ref()
        }
    }
}

/// Applies the UKF covariance update.
pub(crate) fn updated_covariance_ukf<R>(
    covariance_update: CovarianceUpdate,
    predicted_covariance: MatRef<'_, R>,
    gain: MatRef<'_, R>,
    cross: MatRef<'_, R>,
    r: MatRef<'_, R>,
    innovation_covariance: MatRef<'_, R>,
) -> Result<Mat<R>, NonlinearEstimatorError>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    match covariance_update {
        CovarianceUpdate::Simple => Ok(predicted_covariance.to_owned()
            - dense_mul_adjoint_rhs(dense_mul(gain, innovation_covariance).as_ref(), gain)
                .as_ref()),
        CovarianceUpdate::Joseph => {
            let h_t = solve_left_checked(
                predicted_covariance,
                cross,
                default_tolerance::<R>(),
                || NonlinearEstimatorError::SingularPredictedCovariance,
            )?;
            Ok(updated_covariance(
                CovarianceUpdate::Joseph,
                predicted_covariance,
                gain,
                h_t.transpose().to_owned().as_ref(),
                r,
                innovation_covariance,
            ))
        }
    }
}

/// Returns the square root of the normalized innovation energy.
pub(crate) fn normalized_innovation_norm<R>(
    innovation: MatRef<'_, R>,
    innovation_covariance: MatRef<'_, R>,
) -> Result<R::Real, NonlinearEstimatorError>
where
    R: CompensatedField + RealField,
    R::Real: Float + Copy,
{
    let whitened = solve_left_checked(
        innovation_covariance,
        innovation,
        default_tolerance::<R>(),
        || NonlinearEstimatorError::SingularInnovationCovariance,
    )?;
    Ok(inner_product_real(innovation, whitened.as_ref())
        .max(<R::Real as Zero>::zero())
        .sqrt())
}
