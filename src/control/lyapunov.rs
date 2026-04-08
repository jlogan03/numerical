//! Dense continuous-time Lyapunov solves and Gramian helpers.
//!
//! The first Lyapunov path in this crate is intentionally a direct dense
//! reference implementation for modest problem sizes. It solves
//!
//! `A X + X A^H + Q = 0`
//!
//! by vectorizing the matrix equation into one dense linear system and solving
//! that system with `faer`'s full-pivoting dense LU factorization.
//!
//! This is not the long-term sparse/control path for large systems. Its role is:
//!
//! - provide immediately usable controllability and observability Gramians
//! - act as a numerically careful reference for later low-rank sparse solvers
//! - reuse direct solves where they are the right inner kernel
//!
//! The dense path still uses compensated accumulation where that is most useful:
//!
//! - forming `B B^H` and `C^H C`
//! - recomputing the final Lyapunov residual
//! - evaluating the Frobenius residual norm

use crate::sparse::compensated::{CompensatedField, CompensatedSum};
use crate::sum::twosum::TwoSum;
use core::fmt;
use faer::linalg::solvers::Solve;
use faer::{Mat, MatRef};
use faer_traits::ComplexField;
use faer_traits::ext::ComplexFieldExt;
use num_traits::Float;

/// Result of a dense continuous-time Lyapunov solve.
///
/// `solution` is the dense matrix `X` satisfying `A X + X A^H + Q = 0`.
/// `residual_norm` is the compensated Frobenius norm of the final residual,
/// which is useful when the caller wants a numerical sanity check without
/// re-forming the residual themselves.
#[derive(Clone, Debug)]
pub struct DenseLyapunovSolve<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    /// Dense Lyapunov solution matrix.
    pub solution: Mat<T>,
    /// Compensated Frobenius norm of `A X + X A^H + Q`.
    pub residual_norm: T::Real,
}

/// Errors that can occur while building or solving a dense Lyapunov system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LyapunovError {
    /// The state matrix must be square.
    NonSquare { nrows: usize, ncols: usize },
    /// A supplied matrix has incompatible dimensions.
    DimensionMismatch {
        which: &'static str,
        expected_nrows: usize,
        expected_ncols: usize,
        actual_nrows: usize,
        actual_ncols: usize,
    },
    /// The dense solve produced a non-finite result or residual.
    SolveFailed,
}

impl fmt::Display for LyapunovError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for LyapunovError {}

/// Solves the dense continuous-time Lyapunov equation `A X + X A^H + Q = 0`.
///
/// This implementation is intended for modest problem sizes where explicitly
/// forming the dense `n^2 × n^2` Kronecker-sum system is acceptable. It uses
/// `faer`'s full-pivoting LU as a stability-oriented direct reference solve.
///
/// The returned matrix is projected back onto the Hermitian subspace with
/// `(X + X^H) / 2`, since exact Gramians are Hermitian and the direct solve can
/// pick up small asymmetry from finite precision.
pub fn solve_continuous_lyapunov_dense<T>(
    a: MatRef<'_, T>,
    q: MatRef<'_, T>,
) -> Result<DenseLyapunovSolve<T>, LyapunovError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    validate_square("a", a.nrows(), a.ncols())?;
    validate_dims("q", q.nrows(), q.ncols(), a.nrows(), a.ncols())?;

    let n = a.nrows();
    if n == 0 {
        return Ok(DenseLyapunovSolve {
            solution: Mat::zeros(0, 0),
            residual_norm: T::Real::zero(),
        });
    }

    let operator = build_continuous_operator(a);
    let rhs = Mat::from_fn(n * n, 1, |index, _| {
        let row = index % n;
        let col = index / n;
        -q[(row, col)]
    });

    let vectorized = operator.full_piv_lu().solve(rhs.as_ref());
    if !vectorized.as_ref().is_all_finite() {
        return Err(LyapunovError::SolveFailed);
    }

    let mut solution = unvectorize_square(vectorized.as_ref(), n);
    hermitianize_in_place(&mut solution);

    let residual = continuous_residual(a, solution.as_ref(), q);
    let residual_norm = frobenius_norm(residual.as_ref());
    if !residual_norm.is_finite() {
        return Err(LyapunovError::SolveFailed);
    }

    Ok(DenseLyapunovSolve {
        solution,
        residual_norm,
    })
}

/// Computes the dense continuous-time controllability Gramian.
///
/// For a stable system `x' = A x + B u`, the controllability Gramian solves
///
/// `A Wc + Wc A^H + B B^H = 0`
///
/// Intuitively, `B B^H` measures how strongly the inputs drive the state
/// variables, and the Gramian accumulates that effect through the stable
/// dynamics in `A`.
pub fn controllability_gramian_dense<T>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Result<DenseLyapunovSolve<T>, LyapunovError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    validate_square("a", a.nrows(), a.ncols())?;
    validate_dims("b", b.nrows(), b.ncols(), a.nrows(), b.ncols())?;

    let q = dense_mul_with_adjoint_rhs(b);
    solve_continuous_lyapunov_dense(a, q.as_ref())
}

/// Computes the dense continuous-time observability Gramian.
///
/// For a stable system `y = C x`, the observability Gramian solves
///
/// `A^H Wo + Wo A + C^H C = 0`
///
/// This is the dual controllability problem. The implementation uses the same
/// Lyapunov core by solving
///
/// `A_obs X + X A_obs^H + Q = 0`
///
/// with `A_obs = A^H` and `Q = C^H C`.
pub fn observability_gramian_dense<T>(
    a: MatRef<'_, T>,
    c: MatRef<'_, T>,
) -> Result<DenseLyapunovSolve<T>, LyapunovError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    validate_square("a", a.nrows(), a.ncols())?;
    validate_dims("c", c.nrows(), c.ncols(), c.nrows(), a.ncols())?;

    let q = dense_mul_adjoint_lhs(c);
    let a_adjoint = a.adjoint().to_owned();
    solve_continuous_lyapunov_dense(a_adjoint.as_ref(), q.as_ref())
}

fn validate_square(which: &'static str, nrows: usize, ncols: usize) -> Result<(), LyapunovError> {
    if nrows != ncols {
        if which == "a" {
            return Err(LyapunovError::NonSquare { nrows, ncols });
        }
        return Err(LyapunovError::DimensionMismatch {
            which,
            expected_nrows: ncols,
            expected_ncols: ncols,
            actual_nrows: nrows,
            actual_ncols: ncols,
        });
    }
    Ok(())
}

fn validate_dims(
    which: &'static str,
    actual_nrows: usize,
    actual_ncols: usize,
    expected_nrows: usize,
    expected_ncols: usize,
) -> Result<(), LyapunovError> {
    if actual_nrows != expected_nrows || actual_ncols != expected_ncols {
        return Err(LyapunovError::DimensionMismatch {
            which,
            expected_nrows,
            expected_ncols,
            actual_nrows,
            actual_ncols,
        });
    }
    Ok(())
}

fn build_continuous_operator<T>(a: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    let n = a.nrows();
    let mut operator = Mat::<T>::zeros(n * n, n * n);

    // Column-major vectorization turns `A X + X A^H` into
    // `(I ⊗ A + conj(A) ⊗ I) vec(X)`. Building the dense operator explicitly
    // keeps the first implementation simple and makes the direct solve path a
    // reliable reference for later sparse methods.
    for col in 0..n {
        for row in 0..n {
            let eq = vec_index(row, col, n);
            for k in 0..n {
                operator[(eq, vec_index(k, col, n))] =
                    operator[(eq, vec_index(k, col, n))] + a[(row, k)];
                operator[(eq, vec_index(row, k, n))] =
                    operator[(eq, vec_index(row, k, n))] + a[(col, k)].conj();
            }
        }
    }

    operator
}

fn unvectorize_square<T: ComplexField + Copy>(values: MatRef<'_, T>, n: usize) -> Mat<T> {
    Mat::from_fn(n, n, |row, col| values[(vec_index(row, col, n), 0)])
}

fn vec_index(row: usize, col: usize, nrows: usize) -> usize {
    row + nrows * col
}

fn hermitianize_in_place<T>(matrix: &mut Mat<T>)
where
    T: ComplexField + Copy,
    T::Real: Float + Copy,
{
    let half = T::Real::one() / (T::Real::one() + T::Real::one());
    for col in 0..matrix.ncols() {
        for row in 0..=col.min(matrix.nrows().saturating_sub(1)) {
            let avg = (matrix[(row, col)] + matrix[(col, row)].conj()).mul_real(half);
            matrix[(row, col)] = avg;
            matrix[(col, row)] = avg.conj();
        }
    }
}

fn dense_mul_with_adjoint_rhs<T>(lhs: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let nrows = lhs.nrows();
    let ncols = lhs.nrows();
    Mat::from_fn(nrows, ncols, |row, col| {
        let mut acc = CompensatedSum::<T>::default();
        for k in 0..lhs.ncols() {
            acc.add(lhs[(row, k)] * lhs[(col, k)].conj());
        }
        acc.finish()
    })
}

fn dense_mul_adjoint_lhs<T>(rhs: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let nrows = rhs.ncols();
    let ncols = rhs.ncols();
    Mat::from_fn(nrows, ncols, |row, col| {
        let mut acc = CompensatedSum::<T>::default();
        for k in 0..rhs.nrows() {
            acc.add(rhs[(k, row)].conj() * rhs[(k, col)]);
        }
        acc.finish()
    })
}

fn continuous_residual<T>(a: MatRef<'_, T>, x: MatRef<'_, T>, q: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let n = a.nrows();
    Mat::from_fn(n, n, |row, col| {
        let mut acc = CompensatedSum::<T>::default();
        for k in 0..n {
            acc.add(a[(row, k)] * x[(k, col)]);
        }
        for k in 0..n {
            acc.add(x[(row, k)] * a[(col, k)].conj());
        }
        acc.add(q[(row, col)]);
        acc.finish()
    })
}

fn frobenius_norm<T>(matrix: MatRef<'_, T>) -> T::Real
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let mut acc: Option<TwoSum<T::Real>> = None;
    for col in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            let value = matrix[(row, col)].abs2();
            match acc.as_mut() {
                Some(acc) => acc.add(value),
                None => acc = Some(TwoSum::new(value)),
            }
        }
    }

    match acc {
        Some(acc) => {
            let (sum, residual) = acc.finish();
            (sum + residual).sqrt()
        }
        None => T::Real::zero(),
    }
}

#[cfg(test)]
mod test {
    use super::{
        LyapunovError, continuous_residual, controllability_gramian_dense, dense_mul_adjoint_lhs,
        dense_mul_with_adjoint_rhs, frobenius_norm, observability_gramian_dense,
        solve_continuous_lyapunov_dense,
    };
    use faer::{Mat, c64};
    use faer_traits::ext::ComplexFieldExt;

    fn diagonal_solution_from_q<T>(diag: &[T], q: &Mat<T>) -> Mat<T>
    where
        T: super::CompensatedField,
        T::Real: num_traits::Float + Copy,
    {
        Mat::from_fn(diag.len(), diag.len(), |row, col| {
            -q[(row, col)] / (diag[row] + diag[col].conj())
        })
    }

    fn assert_close<T>(lhs: &Mat<T>, rhs: &Mat<T>, tol: T::Real)
    where
        T: super::CompensatedField,
        T::Real: num_traits::Float + Copy,
    {
        assert_eq!(lhs.nrows(), rhs.nrows());
        assert_eq!(lhs.ncols(), rhs.ncols());
        for col in 0..lhs.ncols() {
            for row in 0..lhs.nrows() {
                let err = (lhs[(row, col)] - rhs[(row, col)]).abs1();
                assert!(
                    err <= tol,
                    "entry ({row}, {col}) mismatch: err={err:?}, tol={tol:?}",
                );
            }
        }
    }

    #[test]
    fn dense_continuous_lyapunov_matches_closed_form_real_diagonal_case() {
        let diag = [-1.0f64, -2.0];
        let a = Mat::from_fn(2, 2, |row, col| if row == col { diag[row] } else { 0.0 });
        let q = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => 2.0,
            (0, 1) => -1.0,
            (1, 0) => -1.0,
            _ => 4.0,
        });

        let solve = solve_continuous_lyapunov_dense(a.as_ref(), q.as_ref()).unwrap();
        let expected = diagonal_solution_from_q(&diag, &q);
        assert_close(&solve.solution, &expected, 1.0e-12);
        assert!(solve.residual_norm <= 1.0e-12);
    }

    #[test]
    fn controllability_gramian_matches_closed_form_complex_diagonal_case() {
        let diag = [c64::new(-1.0, 2.0), c64::new(-3.0, -1.0)];
        let a = Mat::from_fn(2, 2, |row, col| {
            if row == col {
                diag[row]
            } else {
                c64::new(0.0, 0.0)
            }
        });
        let b = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => c64::new(1.0, -1.0),
            (0, 1) => c64::new(2.0, 0.5),
            (1, 0) => c64::new(-0.5, 1.0),
            _ => c64::new(1.5, -2.0),
        });

        let q = dense_mul_with_adjoint_rhs(b.as_ref());
        let expected = diagonal_solution_from_q(&diag, &q);
        let solve = controllability_gramian_dense(a.as_ref(), b.as_ref()).unwrap();

        assert_close(&solve.solution, &expected, 1.0e-11);
        assert!(solve.residual_norm <= 1.0e-11);
        for col in 0..solve.solution.ncols() {
            for row in 0..solve.solution.nrows() {
                assert!(
                    (solve.solution[(row, col)] - solve.solution[(col, row)].conj()).abs1()
                        <= 1.0e-11
                );
            }
        }
    }

    #[test]
    fn observability_gramian_matches_closed_form_complex_diagonal_case() {
        let diag = [c64::new(-0.5, 0.75), c64::new(-2.0, -1.5)];
        let a = Mat::from_fn(2, 2, |row, col| {
            if row == col {
                diag[row]
            } else {
                c64::new(0.0, 0.0)
            }
        });
        let c = Mat::from_fn(3, 2, |row, col| match (row, col) {
            (0, 0) => c64::new(1.0, 0.25),
            (0, 1) => c64::new(-0.5, 2.0),
            (1, 0) => c64::new(0.0, -1.0),
            (1, 1) => c64::new(1.5, 0.5),
            (2, 0) => c64::new(-2.0, 1.0),
            _ => c64::new(0.25, -0.75),
        });

        let q = dense_mul_adjoint_lhs(c.as_ref());
        let expected = diagonal_solution_from_q(&[diag[0].conj(), diag[1].conj()], &q);
        let solve = observability_gramian_dense(a.as_ref(), c.as_ref()).unwrap();

        assert_close(&solve.solution, &expected, 1.0e-11);
        assert!(solve.residual_norm <= 1.0e-11);
    }

    #[test]
    fn residual_recomputation_is_small_for_nondiagonal_real_case() {
        let a = Mat::from_fn(3, 3, |row, col| match (row, col) {
            (0, 0) => -2.0,
            (0, 1) => 0.5,
            (0, 2) => 0.0,
            (1, 0) => -0.25,
            (1, 1) => -1.5,
            (1, 2) => 0.75,
            (2, 0) => 0.0,
            (2, 1) => -0.4,
            _ => -0.8,
        });
        let q = Mat::from_fn(3, 3, |row, col| match (row, col) {
            (0, 0) => 3.0,
            (0, 1) => -0.5,
            (0, 2) => 0.25,
            (1, 0) => -0.5,
            (1, 1) => 2.0,
            (1, 2) => -0.3,
            (2, 0) => 0.25,
            (2, 1) => -0.3,
            _ => 1.0,
        });

        let solve = solve_continuous_lyapunov_dense(a.as_ref(), q.as_ref()).unwrap();
        let residual = continuous_residual(a.as_ref(), solve.solution.as_ref(), q.as_ref());
        let residual_norm = frobenius_norm(residual.as_ref());
        assert!(residual_norm <= 1.0e-11);
    }

    #[test]
    fn rejects_dimension_mismatches() {
        let a = Mat::<f64>::identity(2, 2);
        let q = Mat::<f64>::identity(3, 3);
        let err = solve_continuous_lyapunov_dense(a.as_ref(), q.as_ref()).unwrap_err();
        assert_eq!(
            err,
            LyapunovError::DimensionMismatch {
                which: "q",
                expected_nrows: 2,
                expected_ncols: 2,
                actual_nrows: 3,
                actual_ncols: 3,
            }
        );
    }
}
