//! Fixed-size linear algebra helpers for the embedded runtime lane.

use crate::embedded::error::EmbeddedError;
use crate::embedded::math::ensure_finite;
use num_traits::Float;

/// Fixed-size row-major matrix storage.
pub type Matrix<T, const R: usize, const C: usize> = [[T; C]; R];

/// Fixed-size column vector storage.
pub type Vector<T, const N: usize> = [T; N];

/// Returns the all-zero fixed-size matrix.
pub(crate) fn zero_matrix<T, const R: usize, const C: usize>() -> Matrix<T, R, C>
where
    T: Float + Copy,
{
    [[T::zero(); C]; R]
}

/// Returns the all-zero fixed-size vector.
pub(crate) fn zero_vector<T, const N: usize>() -> Vector<T, N>
where
    T: Float + Copy,
{
    [T::zero(); N]
}

/// Returns the identity matrix of order `N`.
pub(crate) fn identity_matrix<T, const N: usize>() -> Matrix<T, N, N>
where
    T: Float + Copy,
{
    let mut out = zero_matrix::<T, N, N>();
    let mut i = 0usize;
    while i < N {
        out[i][i] = T::one();
        i += 1;
    }
    out
}

/// Multiplies one fixed-size matrix by one fixed-size vector.
pub(crate) fn mat_vec_mul<T, const R: usize, const C: usize>(
    a: &Matrix<T, R, C>,
    x: &Vector<T, C>,
) -> Vector<T, R>
where
    T: Float + Copy,
{
    let mut out = zero_vector::<T, R>();
    let mut i = 0usize;
    while i < R {
        let mut acc = T::zero();
        let mut j = 0usize;
        while j < C {
            acc = acc + a[i][j] * x[j];
            j += 1;
        }
        out[i] = acc;
        i += 1;
    }
    out
}

/// Multiplies two fixed-size matrices.
pub(crate) fn mat_mul<T, const R: usize, const K: usize, const C: usize>(
    a: &Matrix<T, R, K>,
    b: &Matrix<T, K, C>,
) -> Matrix<T, R, C>
where
    T: Float + Copy,
{
    let mut out = zero_matrix::<T, R, C>();
    let mut i = 0usize;
    while i < R {
        let mut j = 0usize;
        while j < C {
            let mut acc = T::zero();
            let mut k = 0usize;
            while k < K {
                acc = acc + a[i][k] * b[k][j];
                k += 1;
            }
            out[i][j] = acc;
            j += 1;
        }
        i += 1;
    }
    out
}

/// Transposes one fixed-size matrix.
pub(crate) fn transpose<T, const R: usize, const C: usize>(a: &Matrix<T, R, C>) -> Matrix<T, C, R>
where
    T: Float + Copy,
{
    let mut out = zero_matrix::<T, C, R>();
    let mut i = 0usize;
    while i < R {
        let mut j = 0usize;
        while j < C {
            out[j][i] = a[i][j];
            j += 1;
        }
        i += 1;
    }
    out
}

/// Adds two fixed-size matrices elementwise.
pub(crate) fn mat_add<T, const R: usize, const C: usize>(
    a: &Matrix<T, R, C>,
    b: &Matrix<T, R, C>,
) -> Matrix<T, R, C>
where
    T: Float + Copy,
{
    let mut out = zero_matrix::<T, R, C>();
    let mut i = 0usize;
    while i < R {
        let mut j = 0usize;
        while j < C {
            out[i][j] = a[i][j] + b[i][j];
            j += 1;
        }
        i += 1;
    }
    out
}

/// Subtracts two fixed-size matrices elementwise.
pub(crate) fn mat_sub<T, const R: usize, const C: usize>(
    a: &Matrix<T, R, C>,
    b: &Matrix<T, R, C>,
) -> Matrix<T, R, C>
where
    T: Float + Copy,
{
    let mut out = zero_matrix::<T, R, C>();
    let mut i = 0usize;
    while i < R {
        let mut j = 0usize;
        while j < C {
            out[i][j] = a[i][j] - b[i][j];
            j += 1;
        }
        i += 1;
    }
    out
}

/// Adds two fixed-size vectors elementwise.
pub(crate) fn vec_add<T, const N: usize>(a: &Vector<T, N>, b: &Vector<T, N>) -> Vector<T, N>
where
    T: Float + Copy,
{
    let mut out = zero_vector::<T, N>();
    let mut i = 0usize;
    while i < N {
        out[i] = a[i] + b[i];
        i += 1;
    }
    out
}

/// Subtracts two fixed-size vectors elementwise.
pub(crate) fn vec_sub<T, const N: usize>(a: &Vector<T, N>, b: &Vector<T, N>) -> Vector<T, N>
where
    T: Float + Copy,
{
    let mut out = zero_vector::<T, N>();
    let mut i = 0usize;
    while i < N {
        out[i] = a[i] - b[i];
        i += 1;
    }
    out
}

/// Computes the Euclidean norm of one fixed-size vector.
pub(crate) fn vec_norm<T, const N: usize>(x: &Vector<T, N>) -> T
where
    T: Float + Copy,
{
    let mut sum = T::zero();
    let mut i = 0usize;
    while i < N {
        sum = sum + x[i] * x[i];
        i += 1;
    }
    sum.sqrt()
}

/// Solves `A X = B` for one fixed-size right-hand side block.
pub(crate) fn solve_linear_system<T, const N: usize, const M: usize>(
    a: &Matrix<T, N, N>,
    b: &Matrix<T, N, M>,
    which: &'static str,
) -> Result<Matrix<T, N, M>, EmbeddedError>
where
    T: Float + Copy,
{
    let mut a = *a;
    let mut b = *b;
    let epsilon = T::epsilon().sqrt();

    let mut k = 0usize;
    while k < N {
        let mut pivot_row = k;
        let mut pivot_abs = a[k][k].abs();
        let mut row = k + 1;
        while row < N {
            let candidate = a[row][k].abs();
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
            a.swap(k, pivot_row);
            b.swap(k, pivot_row);
        }

        let diag = a[k][k];
        let mut j = k;
        while j < N {
            a[k][j] = a[k][j] / diag;
            j += 1;
        }
        let mut rhs_col = 0usize;
        while rhs_col < M {
            b[k][rhs_col] = b[k][rhs_col] / diag;
            rhs_col += 1;
        }

        let mut i = 0usize;
        while i < N {
            if i != k {
                let factor = a[i][k];
                if factor != T::zero() {
                    let mut j = k;
                    while j < N {
                        a[i][j] = a[i][j] - factor * a[k][j];
                        j += 1;
                    }
                    let mut rhs_col = 0usize;
                    while rhs_col < M {
                        b[i][rhs_col] = b[i][rhs_col] - factor * b[k][rhs_col];
                        rhs_col += 1;
                    }
                }
            }
            i += 1;
        }

        k += 1;
    }

    let mut row = 0usize;
    while row < N {
        let mut col = 0usize;
        while col < M {
            b[row][col] = ensure_finite(b[row][col], which)?;
            col += 1;
        }
        row += 1;
    }

    Ok(b)
}

/// Inverts one fixed-size square matrix.
pub(crate) fn invert_matrix<T, const N: usize>(
    a: &Matrix<T, N, N>,
    which: &'static str,
) -> Result<Matrix<T, N, N>, EmbeddedError>
where
    T: Float + Copy,
{
    solve_linear_system(a, &identity_matrix::<T, N>(), which)
}
