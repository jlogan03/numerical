//! Small dense helpers for `embedded::alloc`, built on `faer`.

use crate::embedded::error::EmbeddedError;
use crate::embedded::math::ensure_finite;
use faer::linalg::solvers::Solve;
use faer::{Col, Mat, Side, Unbind};
use faer_traits::ComplexField;
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
        for idx in 0..lhs.ncols() {
            acc = acc + lhs[(row, idx)] * rhs[(idx, col)];
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
        for col in 0..matrix.ncols() {
            acc = acc + matrix[(row.unbound(), col)] * vector[col];
        }
        acc
    }))
}

/// Solves `matrix * X = rhs` using faer's dense `LL^T` factorization.
pub fn llt_solve<T>(
    matrix: &Matrix<T>,
    rhs: &Matrix<T>,
    which: &'static str,
) -> Result<Matrix<T>, EmbeddedError>
where
    T: ComplexField<Real = T> + Float + Copy,
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

    let factor = matrix
        .as_ref()
        .llt(Side::Lower)
        .map_err(|_| EmbeddedError::NonPositiveDefinite { which })?;
    let mut solution = factor.solve(rhs.as_ref());
    for row in 0..solution.nrows() {
        for col in 0..solution.ncols() {
            solution[(row, col)] = ensure_finite(solution[(row, col)], which)?;
        }
    }
    Ok(solution)
}

/// Solves `matrix * x = rhs` for one dense vector using faer's dense `LL^T` factorization.
pub fn llt_solve_vector<T>(
    matrix: &Matrix<T>,
    rhs: &Vector<T>,
    which: &'static str,
) -> Result<Vector<T>, EmbeddedError>
where
    T: ComplexField<Real = T> + Float + Copy,
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
        return Err(EmbeddedError::LengthMismatch {
            which,
            expected: matrix.nrows(),
            actual: rhs.nrows(),
        });
    }

    let factor = matrix
        .as_ref()
        .llt(Side::Lower)
        .map_err(|_| EmbeddedError::NonPositiveDefinite { which })?;
    let mut solution = factor.solve(rhs.as_ref());
    for idx in 0..solution.nrows() {
        solution[idx] = ensure_finite(solution[idx], which)?;
    }
    Ok(solution)
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
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[(i, j)];
            for k in 0..j {
                sum = sum - out[(i, k)] * out[(j, k)];
            }

            if i == j {
                if !sum.is_finite() || sum <= T::zero() {
                    return Err(EmbeddedError::NonPositiveDefinite { which });
                }
                out[(i, j)] = sum.sqrt();
            } else {
                out[(i, j)] = sum / out[(j, j)];
            }
        }
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
    for idx in 0..vector.nrows() {
        sum = sum + vector[idx] * vector[idx];
    }
    sum.sqrt()
}

/// Returns the dot product `lhs^T rhs`.
pub fn vec_dot<T>(lhs: &Vector<T>, rhs: &Vector<T>) -> Result<T, EmbeddedError>
where
    T: Float + Copy,
{
    validate_same_len(lhs, rhs, "embedded.alloc.vec_dot")?;
    let mut acc = T::zero();
    for idx in 0..lhs.nrows() {
        acc = acc + lhs[idx] * rhs[idx];
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
