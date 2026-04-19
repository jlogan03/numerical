//! Small heap-backed dense matrix helpers for `embedded::alloc`.

use crate::embedded::error::EmbeddedError;
use crate::embedded::math::ensure_finite;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Index, IndexMut};
use num_traits::Float;

/// Row-major dense matrix used by the `embedded::alloc` estimators.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T> {
    nrows: usize,
    ncols: usize,
    data: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Float + Copy,
{
    /// Returns the all-zero matrix with the requested shape.
    #[must_use]
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            data: vec![T::zero(); nrows * ncols],
        }
    }

    /// Returns the identity matrix of order `n`.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        let mut out = Self::zeros(n, n);
        let mut idx = 0usize;
        while idx < n {
            out[(idx, idx)] = T::one();
            idx += 1;
        }
        out
    }

    /// Builds one matrix from a flat row-major buffer.
    pub fn from_vec(nrows: usize, ncols: usize, data: Vec<T>) -> Result<Self, EmbeddedError> {
        if data.len() != nrows * ncols {
            return Err(EmbeddedError::LengthMismatch {
                which: "embedded.alloc.matrix.from_vec",
                expected: nrows * ncols,
                actual: data.len(),
            });
        }
        Ok(Self { nrows, ncols, data })
    }

    /// Builds one matrix from an element generator.
    #[must_use]
    pub fn from_fn(nrows: usize, ncols: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut out = Self::zeros(nrows, ncols);
        let mut row = 0usize;
        while row < nrows {
            let mut col = 0usize;
            while col < ncols {
                out[(row, col)] = f(row, col);
                col += 1;
            }
            row += 1;
        }
        out
    }

    /// Returns the row count.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns the column count.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns the backing row-major storage.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns the transpose of this matrix.
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self::from_fn(self.ncols, self.nrows, |row, col| self[(col, row)])
    }

    /// Returns the matrix sum `self + rhs`.
    pub fn add(&self, rhs: &Self) -> Result<Self, EmbeddedError> {
        validate_same_shape(self, rhs, "embedded.alloc.matrix.add")?;
        Ok(Self::from_fn(self.nrows, self.ncols, |row, col| {
            self[(row, col)] + rhs[(row, col)]
        }))
    }

    /// Returns the matrix difference `self - rhs`.
    pub fn sub(&self, rhs: &Self) -> Result<Self, EmbeddedError> {
        validate_same_shape(self, rhs, "embedded.alloc.matrix.sub")?;
        Ok(Self::from_fn(self.nrows, self.ncols, |row, col| {
            self[(row, col)] - rhs[(row, col)]
        }))
    }

    /// Returns the matrix product `self * rhs`.
    pub fn mul(&self, rhs: &Self) -> Result<Self, EmbeddedError> {
        if self.ncols != rhs.nrows {
            return Err(EmbeddedError::DimensionMismatch {
                which: "embedded.alloc.matrix.mul",
                expected_rows: self.ncols,
                expected_cols: 1,
                actual_rows: rhs.nrows,
                actual_cols: 1,
            });
        }
        Ok(Self::from_fn(self.nrows, rhs.ncols, |row, col| {
            let mut acc = T::zero();
            let mut idx = 0usize;
            while idx < self.ncols {
                acc = acc + self[(row, idx)] * rhs[(idx, col)];
                idx += 1;
            }
            acc
        }))
    }

    /// Returns the matrix-vector product `self * x`.
    pub fn mul_vec(&self, x: &[T]) -> Result<Vec<T>, EmbeddedError> {
        if self.ncols != x.len() {
            return Err(EmbeddedError::LengthMismatch {
                which: "embedded.alloc.matrix.mul_vec",
                expected: self.ncols,
                actual: x.len(),
            });
        }
        let mut out = vec![T::zero(); self.nrows];
        let mut row = 0usize;
        while row < self.nrows {
            let mut acc = T::zero();
            let mut col = 0usize;
            while col < self.ncols {
                acc = acc + self[(row, col)] * x[col];
                col += 1;
            }
            out[row] = acc;
            row += 1;
        }
        Ok(out)
    }

    /// Returns the scalar-scaled matrix `alpha * self`.
    #[must_use]
    pub fn scale(&self, alpha: T) -> Self {
        Self::from_fn(self.nrows, self.ncols, |row, col| alpha * self[(row, col)])
    }

    /// Solves `self * X = rhs`.
    pub fn solve(&self, rhs: &Self, which: &'static str) -> Result<Self, EmbeddedError> {
        if self.nrows != self.ncols {
            return Err(EmbeddedError::DimensionMismatch {
                which,
                expected_rows: self.nrows,
                expected_cols: self.nrows,
                actual_rows: self.nrows,
                actual_cols: self.ncols,
            });
        }
        if rhs.nrows != self.nrows {
            return Err(EmbeddedError::DimensionMismatch {
                which,
                expected_rows: self.nrows,
                expected_cols: rhs.ncols,
                actual_rows: rhs.nrows,
                actual_cols: rhs.ncols,
            });
        }

        let n = self.nrows;
        let m = rhs.ncols;
        let mut a = self.clone();
        let mut b = rhs.clone();
        let epsilon = T::epsilon().sqrt();

        let mut k = 0usize;
        while k < n {
            let mut pivot_row = k;
            let mut pivot_abs = a[(k, k)].abs();
            let mut row = k + 1;
            while row < n {
                let candidate = a[(row, k)].abs();
                if candidate > pivot_abs {
                    pivot_abs = candidate;
                    pivot_row = row;
                }
                row += 1;
            }

            if pivot_abs <= epsilon {
                return Err(EmbeddedError::SingularMatrix { which });
            }

            if pivot_row != k {
                swap_rows(&mut a, k, pivot_row);
                swap_rows(&mut b, k, pivot_row);
            }

            let diag = a[(k, k)];
            let mut col = k;
            while col < n {
                a[(k, col)] = a[(k, col)] / diag;
                col += 1;
            }
            let mut col = 0usize;
            while col < m {
                b[(k, col)] = b[(k, col)] / diag;
                col += 1;
            }

            let mut row = 0usize;
            while row < n {
                if row != k {
                    let factor = a[(row, k)];
                    if factor != T::zero() {
                        let mut col = k;
                        while col < n {
                            a[(row, col)] = a[(row, col)] - factor * a[(k, col)];
                            col += 1;
                        }
                        let mut col = 0usize;
                        while col < m {
                            b[(row, col)] = b[(row, col)] - factor * b[(k, col)];
                            col += 1;
                        }
                    }
                }
                row += 1;
            }

            k += 1;
        }

        let mut idx = 0usize;
        while idx < b.data.len() {
            b.data[idx] = ensure_finite(b.data[idx], which)?;
            idx += 1;
        }

        Ok(b)
    }

    /// Returns the matrix inverse.
    pub fn inverse(&self, which: &'static str) -> Result<Self, EmbeddedError> {
        self.solve(&Self::identity(self.nrows), which)
    }

    /// Computes the lower-triangular Cholesky factor `L` such that
    /// `self = L L^T`.
    pub fn cholesky_lower(&self, which: &'static str) -> Result<Self, EmbeddedError> {
        if self.nrows != self.ncols {
            return Err(EmbeddedError::DimensionMismatch {
                which,
                expected_rows: self.nrows,
                expected_cols: self.nrows,
                actual_rows: self.nrows,
                actual_cols: self.ncols,
            });
        }

        let n = self.nrows;
        let mut out = Self::zeros(n, n);
        let epsilon = T::epsilon().sqrt();
        let mut i = 0usize;
        while i < n {
            let mut j = 0usize;
            while j <= i {
                let mut sum = self[(i, j)];
                let mut k = 0usize;
                while k < j {
                    sum = sum - out[(i, k)] * out[(j, k)];
                    k += 1;
                }

                if i == j {
                    if sum <= epsilon {
                        return Err(EmbeddedError::NonPositiveDefinite { which });
                    }
                    out[(i, j)] = sum.sqrt();
                } else {
                    out[(i, j)] = sum / out[(j, j)];
                }
                j += 1;
            }
            i += 1;
        }
        Ok(out)
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.ncols + index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 * self.ncols + index.1]
    }
}

/// Returns the outer product `x y^T`.
pub(crate) fn outer_product<T>(x: &[T], y: &[T]) -> Matrix<T>
where
    T: Float + Copy,
{
    Matrix::from_fn(x.len(), y.len(), |row, col| x[row] * y[col])
}

/// Returns the Euclidean norm of one dynamic vector.
pub(crate) fn vec_norm<T>(x: &[T]) -> T
where
    T: Float + Copy,
{
    let mut sum = T::zero();
    let mut idx = 0usize;
    while idx < x.len() {
        sum = sum + x[idx] * x[idx];
        idx += 1;
    }
    sum.sqrt()
}

/// Returns the weighted quadratic form `x^T A x`.
pub(crate) fn quadratic_form<T>(a: &Matrix<T>, x: &[T]) -> Result<T, EmbeddedError>
where
    T: Float + Copy,
{
    let weighted = a.mul_vec(x)?;
    let mut acc = T::zero();
    let mut idx = 0usize;
    while idx < x.len() {
        acc = acc + x[idx] * weighted[idx];
        idx += 1;
    }
    Ok(acc)
}

/// Returns `x + y`.
pub(crate) fn vec_add<T>(x: &[T], y: &[T]) -> Result<Vec<T>, EmbeddedError>
where
    T: Float + Copy,
{
    validate_same_len(x, y, "embedded.alloc.vec_add")?;
    let mut out = vec![T::zero(); x.len()];
    let mut idx = 0usize;
    while idx < x.len() {
        out[idx] = x[idx] + y[idx];
        idx += 1;
    }
    Ok(out)
}

/// Returns `x - y`.
pub(crate) fn vec_sub<T>(x: &[T], y: &[T]) -> Result<Vec<T>, EmbeddedError>
where
    T: Float + Copy,
{
    validate_same_len(x, y, "embedded.alloc.vec_sub")?;
    let mut out = vec![T::zero(); x.len()];
    let mut idx = 0usize;
    while idx < x.len() {
        out[idx] = x[idx] - y[idx];
        idx += 1;
    }
    Ok(out)
}

/// Validates that two matrices have the same shape.
fn validate_same_shape<T>(
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
    which: &'static str,
) -> Result<(), EmbeddedError> {
    if lhs.nrows == rhs.nrows && lhs.ncols == rhs.ncols {
        Ok(())
    } else {
        Err(EmbeddedError::DimensionMismatch {
            which,
            expected_rows: lhs.nrows,
            expected_cols: lhs.ncols,
            actual_rows: rhs.nrows,
            actual_cols: rhs.ncols,
        })
    }
}

/// Validates that two vectors have the same length.
fn validate_same_len<T>(lhs: &[T], rhs: &[T], which: &'static str) -> Result<(), EmbeddedError> {
    if lhs.len() == rhs.len() {
        Ok(())
    } else {
        Err(EmbeddedError::LengthMismatch {
            which,
            expected: lhs.len(),
            actual: rhs.len(),
        })
    }
}

/// Swaps two rows in place.
fn swap_rows<T>(matrix: &mut Matrix<T>, lhs: usize, rhs: usize) {
    if lhs == rhs {
        return;
    }
    let ncols = matrix.ncols;
    let mut col = 0usize;
    while col < ncols {
        matrix.data.swap(lhs * ncols + col, rhs * ncols + col);
        col += 1;
    }
}
