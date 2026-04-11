//! Dense discrete-time Stein solves and discrete Gramians.
//!
//! This module provides the discrete-time analogue of the dense Lyapunov
//! helpers in [`super::lyapunov`]. For a stable discrete-time system, the
//! controllability and observability Gramians solve Stein equations of the form
//!
//! `X - A X A^H = Q`
//!
//! rather than the continuous-time Lyapunov form `A X + X A^H + Q = 0`.
//!
//! The first implementation is intentionally direct and numerically explicit:
//! it vectorizes the matrix equation into a dense linear system
//!
//! `(I - conj(A) ⊗ A) vec(X) = vec(Q)`
//!
//! and solves that system with `faer`'s full-pivoting LU factorization.
//! That is not the asymptotically best dense algorithm, but it is a reliable
//! reference path that fits the current control module cleanly and unlocks
//! dense discrete Gramians for balanced truncation work.

use crate::sparse::compensated::{CompensatedField, CompensatedSum};
use crate::sum::twosum::TwoSum;
use core::fmt;
use faer::linalg::solvers::Solve;
use faer::{Mat, MatRef};
use faer_traits::ComplexField;
use faer_traits::ext::ComplexFieldExt;
use num_traits::{Float, One, Zero};

/// Result of a dense discrete-time Stein solve.
///
/// `solution` is the dense matrix `X` satisfying `X - A X A^H = Q`.
/// `residual_norm` is the compensated Frobenius norm of the final residual.
#[derive(Clone, Debug)]
pub struct DenseSteinSolve<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    /// Dense Stein solution matrix.
    pub solution: Mat<T>,
    /// Compensated Frobenius norm of `X - A X A^H - Q`.
    pub residual_norm: T::Real,
}

/// Errors that can occur while building or solving dense Stein systems.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SteinError {
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

impl fmt::Display for SteinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for SteinError {}

/// Solves the dense discrete-time Stein equation `X - A X A^H = Q`.
///
/// This is the dense direct reference path for modest problem sizes where
/// explicitly building the `n^2 × n^2` Kronecker system is acceptable. The
/// result is projected back onto the Hermitian subspace with `(X + X^H) / 2`,
/// since exact discrete Gramians are Hermitian and the direct solve can pick up
/// small asymmetry from finite precision.
///
/// Compared with the continuous-time Lyapunov equation, the sign pattern
/// changes because the discrete dynamics accumulate through repeated powers of
/// the one-step transition matrix. That is why the dense operator here is
/// `I - conj(A) ⊗ A` rather than a Kronecker sum.
pub fn solve_discrete_stein_dense<T>(
    a: MatRef<'_, T>,
    q: MatRef<'_, T>,
) -> Result<DenseSteinSolve<T>, SteinError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    validate_square("a", a.nrows(), a.ncols())?;
    validate_dims("q", q.nrows(), q.ncols(), a.nrows(), a.ncols())?;

    let n = a.nrows();
    if n == 0 {
        return Ok(DenseSteinSolve {
            solution: Mat::zeros(0, 0),
            residual_norm: <T::Real as Zero>::zero(),
        });
    }

    let operator = build_discrete_operator(a);
    let rhs = Mat::from_fn(n * n, 1, |index, _| {
        let row = index % n;
        let col = index / n;
        q[(row, col)]
    });

    let vectorized = operator.full_piv_lu().solve(rhs.as_ref());
    if !vectorized.as_ref().is_all_finite() {
        return Err(SteinError::SolveFailed);
    }

    let mut solution = unvectorize_square(vectorized.as_ref(), n);
    // Exact discrete Gramians are Hermitian. This projection removes the small
    // skew-Hermitian component introduced by the finite-precision dense solve
    // without changing the intended solution materially.
    hermitianize_in_place(&mut solution);

    let residual = discrete_residual(a, solution.as_ref(), q);
    let residual_norm = frobenius_norm(residual.as_ref());
    if !residual_norm.is_finite() {
        return Err(SteinError::SolveFailed);
    }

    Ok(DenseSteinSolve {
        solution,
        residual_norm,
    })
}

/// Computes the dense discrete-time controllability Gramian.
///
/// For a stable system `x[k + 1] = A x[k] + B u[k]`, the controllability
/// Gramian satisfies
///
/// `Wc - A Wc A^H = B B^H`
///
/// Intuitively, `B B^H` measures how the inputs inject energy into the state,
/// and the Stein solve accumulates that effect across repeated applications of
/// the one-step transition `A`.
///
/// This is the discrete-time analogue of the continuous controllability
/// Gramian, with `B B^H` acting as the per-step input energy injection term.
pub fn controllability_gramian_discrete_dense<T>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Result<DenseSteinSolve<T>, SteinError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    validate_square("a", a.nrows(), a.ncols())?;
    validate_dims("b", b.nrows(), b.ncols(), a.nrows(), b.ncols())?;

    // `B B^H` is the natural Gramian forcing term in the controllability
    // equation. It stays Hermitian and positive semidefinite in exact
    // arithmetic even for complex-valued systems.
    let q = dense_mul_adjoint_rhs(b, b);
    solve_discrete_stein_dense(a, q.as_ref())
}

/// Computes the dense discrete-time observability Gramian.
///
/// For a stable system `y[k] = C x[k]`, the observability Gramian satisfies
///
/// `Wo - A^H Wo A = C^H C`
///
/// This is the dual controllability problem. The implementation reuses the same
/// Stein core by solving with `A_obs = A^H` and `Q = C^H C`.
///
/// Writing the observability equation this way keeps one dense Stein solver at
/// the center of the implementation instead of maintaining two nearly
/// identical direct paths.
pub fn observability_gramian_discrete_dense<T>(
    a: MatRef<'_, T>,
    c: MatRef<'_, T>,
) -> Result<DenseSteinSolve<T>, SteinError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    validate_square("a", a.nrows(), a.ncols())?;
    validate_dims("c", c.nrows(), c.ncols(), c.nrows(), a.ncols())?;

    // `C^H C` is the dual forcing term: it measures how strongly each state
    // direction is exposed through the outputs.
    let q = dense_mul_adjoint_lhs(c, c);
    let a_adjoint = dense_adjoint(a);
    solve_discrete_stein_dense(a_adjoint.as_ref(), q.as_ref())
}

fn validate_square(which: &'static str, nrows: usize, ncols: usize) -> Result<(), SteinError> {
    if nrows != ncols {
        if which == "a" {
            return Err(SteinError::NonSquare { nrows, ncols });
        }
        return Err(SteinError::DimensionMismatch {
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
) -> Result<(), SteinError> {
    if actual_nrows != expected_nrows || actual_ncols != expected_ncols {
        return Err(SteinError::DimensionMismatch {
            which,
            expected_nrows,
            expected_ncols,
            actual_nrows,
            actual_ncols,
        });
    }
    Ok(())
}

fn build_discrete_operator<T>(a: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    let n = a.nrows();
    let mut operator = Mat::<T>::zeros(n * n, n * n);

    // Column-major vectorization turns `A X A^H` into
    // `(conj(A) ⊗ A) vec(X)`. The Stein equation is then
    // `(I - conj(A) ⊗ A) vec(X) = vec(Q)`.
    //
    // The implementation writes the dense operator entry-by-entry rather than
    // relying on a more opaque Kronecker helper so the exact indexing contract
    // stays visible in this reference path.
    for col in 0..n {
        for row in 0..n {
            let eq = vec_index(row, col, n);
            operator[(eq, eq)] = T::one_impl();
            for right_col in 0..n {
                for left_row in 0..n {
                    operator[(eq, vec_index(left_row, right_col, n))] -=
                        a[(row, left_row)] * a[(col, right_col)].conj();
                }
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

fn dense_adjoint<T>(matrix: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    Mat::from_fn(matrix.ncols(), matrix.nrows(), |row, col| {
        matrix[(col, row)].conj()
    })
}

fn hermitianize_in_place<T>(matrix: &mut Mat<T>)
where
    T: ComplexField + Copy,
    T::Real: Float + Copy,
{
    let half = <T::Real as One>::one() / (<T::Real as One>::one() + <T::Real as One>::one());
    for col in 0..matrix.ncols() {
        for row in 0..=col.min(matrix.nrows().saturating_sub(1)) {
            let avg = (matrix[(row, col)] + matrix[(col, row)].conj()).mul_real(half);
            matrix[(row, col)] = avg;
            matrix[(col, row)] = avg.conj();
        }
    }
}

fn dense_mul<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    // These dense helpers are not meant to outcompete faer's dense kernels.
    // They exist so the residual and Gramian forcing terms can use the same
    // compensated reduction style as the rest of this crate's numerically
    // conservative control code.
    Mat::from_fn(lhs.nrows(), rhs.ncols(), |row, col| {
        let mut acc = CompensatedSum::<T>::default();
        for k in 0..lhs.ncols() {
            acc.add(lhs[(row, k)] * rhs[(k, col)]);
        }
        acc.finish()
    })
}

fn dense_mul_adjoint_rhs<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
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

fn dense_mul_adjoint_lhs<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
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

fn discrete_residual<T>(a: MatRef<'_, T>, x: MatRef<'_, T>, q: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let ax = dense_mul(a, x);
    let axah = dense_mul_adjoint_rhs(ax.as_ref(), a);
    Mat::from_fn(x.nrows(), x.ncols(), |row, col| {
        let mut acc = CompensatedSum::<T>::default();
        // Recompute the residual in the original matrix form instead of
        // recycling the vectorized solve state. That gives a more meaningful
        // post-solve accuracy check for downstream control code.
        acc.add(x[(row, col)]);
        acc.add(-axah[(row, col)]);
        acc.add(-q[(row, col)]);
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
        None => <T::Real as Zero>::zero(),
    }
}

#[cfg(test)]
mod test {
    use super::{
        SteinError, controllability_gramian_discrete_dense, discrete_residual,
        observability_gramian_discrete_dense, solve_discrete_stein_dense,
    };
    use faer::{Mat, c64};
    use faer_traits::ext::ComplexFieldExt;

    fn diagonal_solution_from_q<T>(diag: &[T], q: &Mat<T>) -> Mat<T>
    where
        T: super::CompensatedField,
        T::Real: num_traits::Float + Copy,
    {
        Mat::from_fn(diag.len(), diag.len(), |row, col| {
            q[(row, col)] / (T::one_impl() - diag[row] * diag[col].conj())
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
    fn dense_discrete_stein_matches_closed_form_real_diagonal_case() {
        let diag = [0.25f64, -0.5];
        let a = Mat::from_fn(2, 2, |row, col| if row == col { diag[row] } else { 0.0 });
        let q = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => 2.0,
            (0, 1) => -1.0,
            (1, 0) => -1.0,
            _ => 4.0,
        });

        let solve = solve_discrete_stein_dense(a.as_ref(), q.as_ref()).unwrap();
        let expected = diagonal_solution_from_q(&diag, &q);
        assert_close(&solve.solution, &expected, 1.0e-12);
        assert!(solve.residual_norm <= 1.0e-12);
    }

    #[test]
    fn controllability_gramian_matches_closed_form_complex_diagonal_case() {
        let diag = [c64::new(0.25, 0.1), c64::new(-0.35, -0.2)];
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

        let q = super::dense_mul_adjoint_rhs(b.as_ref(), b.as_ref());
        let expected = diagonal_solution_from_q(&diag, &q);
        let solve = controllability_gramian_discrete_dense(a.as_ref(), b.as_ref()).unwrap();

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
        let diag = [c64::new(0.4, 0.15), c64::new(-0.2, -0.3)];
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

        let q = super::dense_mul_adjoint_lhs(c.as_ref(), c.as_ref());
        let expected = diagonal_solution_from_q(&[diag[0].conj(), diag[1].conj()], &q);
        let solve = observability_gramian_discrete_dense(a.as_ref(), c.as_ref()).unwrap();

        assert_close(&solve.solution, &expected, 1.0e-11);
        assert!(solve.residual_norm <= 1.0e-11);
    }

    #[test]
    fn residual_recomputation_is_small_for_nondiagonal_real_case() {
        let a = Mat::from_fn(3, 3, |row, col| match (row, col) {
            (0, 0) => 0.4,
            (0, 1) => 0.1,
            (0, 2) => 0.0,
            (1, 0) => -0.2,
            (1, 1) => -0.3,
            (1, 2) => 0.15,
            (2, 0) => 0.0,
            (2, 1) => -0.1,
            _ => 0.2,
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

        let solve = solve_discrete_stein_dense(a.as_ref(), q.as_ref()).unwrap();
        let residual = discrete_residual(a.as_ref(), solve.solution.as_ref(), q.as_ref());
        let residual_norm = super::frobenius_norm(residual.as_ref());
        assert!(residual_norm <= 1.0e-11);
    }

    #[test]
    fn rejects_dimension_mismatches() {
        let a = Mat::<f64>::identity(2, 2);
        let q = Mat::<f64>::identity(3, 3);
        let err = solve_discrete_stein_dense(a.as_ref(), q.as_ref()).unwrap_err();
        assert!(matches!(
            err,
            SteinError::DimensionMismatch {
                which: "q",
                expected_nrows: 2,
                expected_ncols: 2,
                actual_nrows: 3,
                actual_ncols: 3,
            }
        ));
    }
}
