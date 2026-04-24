//! Shared dense-matrix helpers for control algorithms.
//!
//! These helpers centralize the crate's repeated dense kernels so control
//! submodules can share one implementation and one accumulation policy.
//!
//! # Glossary
//!
//! - **Adjoint:** Conjugate transpose of a matrix.
//! - **Compensated accumulation:** Summation that keeps a running correction
//!   term to reduce floating-point loss.
//! - **Hermitian projection:** Explicit symmetrization step used to pull a
//!   nearly Hermitian matrix back onto the Hermitian manifold.
//! - **Frobenius norm:** Square-root of the sum of squared entry magnitudes.

use crate::sparse::compensated::{CompensatedField, CompensatedSum};
use faer::{Mat, MatRef};
use faer_traits::ext::ComplexFieldExt;
use num_traits::{Float, One, Zero};

/// Dense matrix multiply using compensated accumulation per output entry.
pub(crate) fn dense_mul<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
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

/// Dense multiply with an adjoint on the right-hand factor.
pub(crate) fn dense_mul_adjoint_rhs<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
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

/// Dense multiply with an adjoint on the left-hand factor.
pub(crate) fn dense_mul_adjoint_lhs<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    Mat::from_fn(lhs.ncols(), rhs.ncols(), |row, col| {
        let mut acc = CompensatedSum::<T>::default();
        for k in 0..lhs.nrows() {
            acc.add(lhs[(k, row)].conj() * rhs[(k, col)]);
        }
        acc.finish()
    })
}

/// Plain dense matrix multiply used only in residual checks or lightweight analysis.
pub(crate) fn dense_mul_plain<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: faer_traits::ComplexField + Copy,
{
    Mat::from_fn(lhs.nrows(), rhs.ncols(), |row, col| {
        let mut acc = T::zero();
        for k in 0..lhs.ncols() {
            acc += lhs[(row, k)] * rhs[(k, col)];
        }
        acc
    })
}

/// Returns the real part of the Hermitian inner product between two vectors.
pub(crate) fn inner_product_real<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> T::Real
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let mut acc = CompensatedSum::<T>::default();
    for row in 0..lhs.nrows() {
        acc.add(lhs[(row, 0)].conj() * rhs[(row, 0)]);
    }
    acc.finish().real()
}

/// Returns the Euclidean norm of a dense column vector.
pub(crate) fn column_vector_norm<T>(vector: MatRef<'_, T>) -> T::Real
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let mut acc = <T::Real as Zero>::zero();
    for row in 0..vector.nrows() {
        acc += vector[(row, 0)].abs2();
    }
    acc.sqrt()
}

/// Projects a dense matrix onto the Hermitian subspace in place.
pub(crate) fn hermitian_project_in_place<T>(matrix: &mut Mat<T>)
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

/// Returns the Frobenius norm of a dense matrix without compensation.
pub(crate) fn frobenius_norm_plain<T>(matrix: MatRef<'_, T>) -> T::Real
where
    T: faer_traits::ComplexField + Copy,
    T::Real: Float + Copy,
{
    let mut acc = <T::Real as Zero>::zero();
    for col in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            acc += matrix[(row, col)].abs2();
        }
    }
    acc.sqrt()
}
