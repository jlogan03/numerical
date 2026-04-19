//! Small dense helpers for `embedded::alloc`, built on `faer`.

use crate::embedded::error::EmbeddedError;
use crate::embedded::math::ensure_finite;
use faer::{Col, Mat, Unbind};
use num_traits::Float;

/// Heap-backed dense matrix used by the `embedded::alloc` estimators.
pub type Matrix<T> = Mat<T>;

/// Heap-backed dense column vector used by the `embedded::alloc` estimators.
pub type Vector<T> = Col<T>;

/// Returns the all-zero dense matrix with the requested shape.
pub fn zero_matrix<T>(nrows: usize, ncols: usize) -> Matrix<T>
where
    T: Float + Copy,
{
    Mat::from_fn(nrows, ncols, |_, _| T::zero())
}

/// Returns the identity matrix of order `n`.
pub fn identity_matrix<T>(n: usize) -> Matrix<T>
where
    T: Float + Copy,
{
    Mat::from_fn(
        n,
        n,
        |row, col| if row == col { T::one() } else { T::zero() },
    )
}

/// Returns the all-zero dense vector of length `n`.
pub fn zero_vector<T>(n: usize) -> Vector<T>
where
    T: Float + Copy,
{
    Col::from_fn(n, |_| T::zero())
}

/// Copies a dense vector from a slice.
pub fn vector_from_slice<T>(values: &[T]) -> Vector<T>
where
    T: Copy,
{
    Col::from_fn(values.len(), |i| values[i.unbound()])
}

/// Returns an immutable slice view of one dense vector.
pub fn vec_as_slice<T>(vector: &Vector<T>) -> &[T] {
    vector.try_as_col_major().unwrap().as_slice()
}

/// Returns a mutable slice view of one dense vector.
pub fn vec_as_slice_mut<T>(vector: &mut Vector<T>) -> &mut [T] {
    vector.try_as_col_major_mut().unwrap().as_slice_mut()
}

/// Returns the transpose of one dense matrix.
pub fn transpose<T>(matrix: &Matrix<T>) -> Matrix<T>
where
    T: Float + Copy,
{
    Mat::from_fn(matrix.ncols(), matrix.nrows(), |row, col| {
        matrix[(col, row)]
    })
}

/// Returns the matrix sum `lhs + rhs`.
pub fn mat_add<T>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Result<Matrix<T>, EmbeddedError>
where
    T: Float + Copy,
{
    validate_same_shape(lhs, rhs, "embedded.alloc.matrix.add")?;
    Ok(Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| {
        lhs[(row, col)] + rhs[(row, col)]
    }))
}

/// Returns the matrix difference `lhs - rhs`.
pub fn mat_sub<T>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Result<Matrix<T>, EmbeddedError>
where
    T: Float + Copy,
{
    validate_same_shape(lhs, rhs, "embedded.alloc.matrix.sub")?;
    Ok(Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| {
        lhs[(row, col)] - rhs[(row, col)]
    }))
}

/// Returns the scalar-scaled matrix `alpha * matrix`.
pub fn scale_matrix<T>(matrix: &Matrix<T>, alpha: T) -> Matrix<T>
where
    T: Float + Copy,
{
    Mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        alpha * matrix[(row, col)]
    })
}

/// Returns the matrix product `lhs * rhs`.
pub fn mat_mul<T>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Result<Matrix<T>, EmbeddedError>
where
    T: Float + Copy,
{
    if lhs.ncols() != rhs.nrows() {
        return Err(EmbeddedError::DimensionMismatch {
            which: "embedded.alloc.matrix.mul",
            expected_rows: lhs.ncols(),
            expected_cols: 1,
            actual_rows: rhs.nrows(),
            actual_cols: 1,
        });
    }

    Ok(Mat::from_fn(lhs.nrows(), rhs.ncols(), |row, col| {
        let mut acc = T::zero();
        let mut idx = 0usize;
        while idx < lhs.ncols() {
            acc = acc + lhs[(row, idx)] * rhs[(idx, col)];
            idx += 1;
        }
        acc
    }))
}

/// Returns the dense vector product `matrix * vector`.
pub fn mat_mul_vec<T>(matrix: &Matrix<T>, vector: &Vector<T>) -> Result<Vector<T>, EmbeddedError>
where
    T: Float + Copy,
{
    if matrix.ncols() != vector.nrows() {
        return Err(EmbeddedError::LengthMismatch {
            which: "embedded.alloc.matrix.mul_vec",
            expected: matrix.ncols(),
            actual: vector.nrows(),
        });
    }

    Ok(Col::from_fn(matrix.nrows(), |row| {
        let mut acc = T::zero();
        let mut col = 0usize;
        while col < matrix.ncols() {
            acc = acc + matrix[(row.unbound(), col)] * vector[col];
            col += 1;
        }
        acc
    }))
}

/// Solves `matrix * X = rhs`.
pub fn solve_linear_system<T>(
    matrix: &Matrix<T>,
    rhs: &Matrix<T>,
    which: &'static str,
) -> Result<Matrix<T>, EmbeddedError>
where
    T: Float + Copy,
{
    if matrix.nrows() != matrix.ncols() {
        return Err(EmbeddedError::DimensionMismatch {
            which,
            expected_rows: matrix.nrows(),
            expected_cols: matrix.nrows(),
            actual_rows: matrix.nrows(),
            actual_cols: matrix.ncols(),
        });
    }
    if rhs.nrows() != matrix.nrows() {
        return Err(EmbeddedError::DimensionMismatch {
            which,
            expected_rows: matrix.nrows(),
            expected_cols: rhs.ncols(),
            actual_rows: rhs.nrows(),
            actual_cols: rhs.ncols(),
        });
    }

    let n = matrix.nrows();
    let m = rhs.ncols();
    let mut a = matrix.clone();
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
                row += 1;
            } else {
                row += 1;
            }
        }

        k += 1;
    }

    let mut row = 0usize;
    while row < b.nrows() {
        let mut col = 0usize;
        while col < b.ncols() {
            b[(row, col)] = ensure_finite(b[(row, col)], which)?;
            col += 1;
        }
        row += 1;
    }

    Ok(b)
}

/// Returns the inverse of one dense square matrix.
pub fn invert_matrix<T>(matrix: &Matrix<T>, which: &'static str) -> Result<Matrix<T>, EmbeddedError>
where
    T: Float + Copy,
{
    solve_linear_system(matrix, &identity_matrix(matrix.nrows()), which)
}

/// Returns the lower-triangular Cholesky factor `L` such that `A = L L^T`.
pub fn cholesky_lower<T>(
    matrix: &Matrix<T>,
    which: &'static str,
) -> Result<Matrix<T>, EmbeddedError>
where
    T: Float + Copy,
{
    if matrix.nrows() != matrix.ncols() {
        return Err(EmbeddedError::DimensionMismatch {
            which,
            expected_rows: matrix.nrows(),
            expected_cols: matrix.nrows(),
            actual_rows: matrix.nrows(),
            actual_cols: matrix.ncols(),
        });
    }

    let n = matrix.nrows();
    let mut out = zero_matrix(n, n);
    let mut i = 0usize;
    while i < n {
        let mut j = 0usize;
        while j <= i {
            let mut sum = matrix[(i, j)];
            let mut k = 0usize;
            while k < j {
                sum = sum - out[(i, k)] * out[(j, k)];
                k += 1;
            }

            if i == j {
                if !sum.is_finite() || sum <= T::zero() {
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

/// Returns the outer product `x y^T`.
pub fn outer_product<T>(x: &Vector<T>, y: &Vector<T>) -> Matrix<T>
where
    T: Float + Copy,
{
    Mat::from_fn(x.nrows(), y.nrows(), |row, col| x[row] * y[col])
}

/// Returns the Euclidean norm of one dense vector.
pub fn vec_norm<T>(vector: &Vector<T>) -> T
where
    T: Float + Copy,
{
    let mut sum = T::zero();
    let mut idx = 0usize;
    while idx < vector.nrows() {
        sum = sum + vector[idx] * vector[idx];
        idx += 1;
    }
    sum.sqrt()
}

/// Returns the quadratic form `x^T A x`.
pub fn quadratic_form<T>(matrix: &Matrix<T>, vector: &Vector<T>) -> Result<T, EmbeddedError>
where
    T: Float + Copy,
{
    let weighted = mat_mul_vec(matrix, vector)?;
    let mut acc = T::zero();
    let mut idx = 0usize;
    while idx < vector.nrows() {
        acc = acc + vector[idx] * weighted[idx];
        idx += 1;
    }
    Ok(acc)
}

/// Returns `lhs + rhs` for dense vectors.
pub fn vec_add<T>(lhs: &Vector<T>, rhs: &Vector<T>) -> Result<Vector<T>, EmbeddedError>
where
    T: Float + Copy,
{
    validate_same_len(lhs, rhs, "embedded.alloc.vec_add")?;
    Ok(Col::from_fn(lhs.nrows(), |i| {
        lhs[i.unbound()] + rhs[i.unbound()]
    }))
}

/// Returns `lhs - rhs` for dense vectors.
pub fn vec_sub<T>(lhs: &Vector<T>, rhs: &Vector<T>) -> Result<Vector<T>, EmbeddedError>
where
    T: Float + Copy,
{
    validate_same_len(lhs, rhs, "embedded.alloc.vec_sub")?;
    Ok(Col::from_fn(lhs.nrows(), |i| {
        lhs[i.unbound()] - rhs[i.unbound()]
    }))
}

/// Validates that two dense matrices have the same shape.
fn validate_same_shape<T>(
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
    which: &'static str,
) -> Result<(), EmbeddedError> {
    if lhs.nrows() == rhs.nrows() && lhs.ncols() == rhs.ncols() {
        Ok(())
    } else {
        Err(EmbeddedError::DimensionMismatch {
            which,
            expected_rows: lhs.nrows(),
            expected_cols: lhs.ncols(),
            actual_rows: rhs.nrows(),
            actual_cols: rhs.ncols(),
        })
    }
}

/// Validates that two dense vectors have the same length.
fn validate_same_len<T>(
    lhs: &Vector<T>,
    rhs: &Vector<T>,
    which: &'static str,
) -> Result<(), EmbeddedError> {
    if lhs.nrows() == rhs.nrows() {
        Ok(())
    } else {
        Err(EmbeddedError::LengthMismatch {
            which,
            expected: lhs.nrows(),
            actual: rhs.nrows(),
        })
    }
}

/// Swaps two rows in one dense matrix.
fn swap_rows<T>(matrix: &mut Matrix<T>, lhs: usize, rhs: usize)
where
    T: Float + Copy,
{
    if lhs == rhs {
        return;
    }
    let mut col = 0usize;
    while col < matrix.ncols() {
        let tmp = matrix[(lhs, col)];
        matrix[(lhs, col)] = matrix[(rhs, col)];
        matrix[(rhs, col)] = tmp;
        col += 1;
    }
}
