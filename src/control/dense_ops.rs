//! Shared dense-matrix helpers for control algorithms.
//!
//! These helpers centralize the crate's repeated dense kernels so control
//! submodules can share one implementation and one accumulation policy.

use crate::sparse::compensated::{CompensatedField, CompensatedSum};
use faer::{Mat, MatRef};
use faer_traits::ComplexField;
use faer_traits::ext::ComplexFieldExt;
use num_traits::{Float, One, Zero};

/// Clones a dense matrix reference into an owned matrix.
pub(crate) fn clone_mat<T: Copy>(matrix: MatRef<'_, T>) -> Mat<T> {
    Mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        matrix[(row, col)]
    })
}

/// Returns a dense identity matrix of the requested dimension.
pub(crate) fn identity<T>(dim: usize) -> Mat<T>
where
    T: ComplexField + Copy,
{
    Mat::identity(dim, dim)
}

/// Returns the plain transpose of a dense matrix.
pub(crate) fn dense_transpose<T: Copy>(matrix: MatRef<'_, T>) -> Mat<T> {
    Mat::from_fn(matrix.ncols(), matrix.nrows(), |row, col| {
        matrix[(col, row)]
    })
}

/// Returns the dense adjoint of a matrix.
pub(crate) fn dense_adjoint<T>(matrix: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    matrix.adjoint().to_owned()
}

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

/// Dense matrix addition using compensated accumulation per entry.
pub(crate) fn dense_add<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    dense_add_plain(lhs, rhs)
}

/// Dense matrix subtraction using compensated accumulation per entry.
pub(crate) fn dense_sub<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    dense_sub_plain(lhs, rhs)
}

/// Plain dense matrix addition used where compensation is unnecessary.
pub(crate) fn dense_add_plain<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| {
        lhs[(row, col)] + rhs[(row, col)]
    })
}

/// Plain dense matrix multiply used only in residual checks or lightweight analysis.
pub(crate) fn dense_mul_plain<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
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

/// Plain dense subtraction used only in residual checks.
pub(crate) fn dense_sub_plain<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| {
        lhs[(row, col)] - rhs[(row, col)]
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
        acc = acc + vector[(row, 0)].abs2();
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
