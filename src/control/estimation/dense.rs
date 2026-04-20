//! Shared dense solver helpers for estimator runtimes.

use crate::control::dense_ops::{dense_mul_plain, frobenius_norm_plain};
use crate::sparse::compensated::CompensatedField;
use faer::prelude::Solve;
use faer::{Mat, MatRef};
use faer_traits::ComplexField;
use num_traits::{Float, One};

/// Returns the default dense-solve tolerance used by estimator internals.
pub(crate) fn default_tolerance<T>() -> T::Real
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    T::Real::epsilon().sqrt()
}

/// Solves `lhs * X = rhs` and rejects numerically unusable results.
pub(crate) fn solve_left_checked<T, E, F>(
    lhs: MatRef<'_, T>,
    rhs: MatRef<'_, T>,
    tol: T::Real,
    err: F,
) -> Result<Mat<T>, E>
where
    T: ComplexField + Copy,
    T::Real: Float + Copy,
    F: Fn() -> E,
{
    let solution = lhs.full_piv_lu().solve(rhs);
    if !solution.as_ref().is_all_finite() {
        return Err(err());
    }

    let residual = dense_mul_plain(lhs, solution.as_ref()) - rhs;
    let residual_norm = frobenius_norm_plain(residual.as_ref());
    let scale = frobenius_norm_plain(lhs) * frobenius_norm_plain(solution.as_ref())
        + frobenius_norm_plain(rhs);
    let one = <T::Real as One>::one();
    let threshold = scale.max(one) * tol * (one + one);
    if !residual_norm.is_finite() || residual_norm > threshold {
        return Err(err());
    }

    Ok(solution)
}

/// Solves `X * lhs = rhs` by transposing into [`solve_left_checked`].
pub(crate) fn solve_right_checked<T, E, F>(
    rhs_left: MatRef<'_, T>,
    lhs_right: MatRef<'_, T>,
    tol: T::Real,
    err: F,
) -> Result<Mat<T>, E>
where
    T: ComplexField + Copy,
    T::Real: Float + Copy,
    F: Fn() -> E + Copy,
{
    let solved_t = solve_left_checked(lhs_right.transpose(), rhs_left.transpose(), tol, err)?;
    Ok(solved_t.transpose().to_owned())
}
