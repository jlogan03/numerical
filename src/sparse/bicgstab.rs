use super::col::{col_from_slice, col_slice, col_slice_mut, copy_col, zero_col};
use super::compensated::{dotc, norm2, norm2_sq};
use super::field::Field;
use super::matvec::SparseMatVec;
use faer::Col;
use num_traits::Float;

/// Stabilized bi-conjugate gradient solver.
#[derive(Debug)]
pub struct BiCGSTAB<T: Field, A: SparseMatVec<T>> {
    iteration_count: usize,
    soft_restart_threshold: T::Real,
    soft_restart_count: usize,
    hard_restart_count: usize,
    err: T::Real,
    a: A,
    b: Col<T>,
    x: Col<T>,
    r: Col<T>,
    rhat: Col<T>,
    p: Col<T>,
    v: Col<T>,
    s: Col<T>,
    t: Col<T>,
    scratch: Col<T>,
    rho: T,
}

impl<T: Field, A: SparseMatVec<T>> BiCGSTAB<T, A> {
    /// Initializes a solver with a fresh residual estimate.
    #[inline]
    #[must_use]
    pub fn new(a: A, x0: &[T], b: &[T]) -> Self {
        assert_eq!(a.nrows(), a.ncols(), "BiCGSTAB requires a square matrix");
        assert_eq!(x0.len(), a.ncols(), "Initial guess has the wrong length");
        assert_eq!(b.len(), a.nrows(), "Right-hand side has the wrong length");

        let n = a.nrows();
        let b_col = col_from_slice(b);
        let x = col_from_slice(x0);
        let mut scratch = zero_col::<T>(n);
        let mut r = zero_col::<T>(n);

        a.apply_compensated(col_slice_mut(&mut scratch), col_slice(&x));
        for ((r, &b), &ax) in col_slice_mut(&mut r)
            .iter_mut()
            .zip(col_slice(&b_col).iter())
            .zip(col_slice(&scratch).iter())
        {
            *r = b - ax;
        }

        let err = norm2::<T>(col_slice(&r));
        let rho = dotc::<T>(col_slice(&r), col_slice(&r));
        let rhat = col_from_slice(col_slice(&r));
        let p = col_from_slice(col_slice(&r));
        let v = zero_col::<T>(n);
        let s = zero_col::<T>(n);
        let t = zero_col::<T>(n);

        Self {
            iteration_count: 0,
            soft_restart_threshold: T::real_from_f64(0.1),
            soft_restart_count: 0,
            hard_restart_count: 0,
            err,
            a,
            b: b_col,
            x,
            r,
            rhat,
            p,
            v,
            s,
            t,
            scratch,
            rho,
        }
    }

    /// Attempts to solve the system to the given absolute residual tolerance.
    pub fn solve(a: A, x0: &[T], b: &[T], tol: T::Real, max_iter: usize) -> Result<Self, Self> {
        let mut solver = Self::new(a, x0, b);
        if solver.err() < tol {
            return Ok(solver);
        }

        for _ in 0..max_iter {
            solver.step();
            if solver.err() < tol {
                solver.hard_restart();
                if solver.err() < tol {
                    return Ok(solver);
                }
            }
        }

        Err(solver)
    }

    /// Resets the shadow residual to avoid a singular update.
    pub fn soft_restart(&mut self) {
        self.soft_restart_count += 1;
        copy_col(&mut self.rhat, &self.r);
        self.rho = dotc::<T>(col_slice(&self.r), col_slice(&self.r));
        copy_col(&mut self.p, &self.r);
    }

    /// Recomputes the residual from scratch with compensated accumulation.
    pub fn hard_restart(&mut self) {
        self.hard_restart_count += 1;
        self.a
            .apply_compensated(col_slice_mut(&mut self.scratch), col_slice(&self.x));

        for ((r, &b), &ax) in col_slice_mut(&mut self.r)
            .iter_mut()
            .zip(col_slice(&self.b).iter())
            .zip(col_slice(&self.scratch).iter())
        {
            *r = b - ax;
        }

        self.err = norm2::<T>(col_slice(&self.r));
        self.soft_restart();
        self.soft_restart_count -= 1;
    }

    /// Advances the solver by one BiCGSTAB iteration.
    pub fn step(&mut self) -> T::Real {
        self.iteration_count += 1;

        self.a.apply(col_slice_mut(&mut self.v), col_slice(&self.p));

        let denom = dotc::<T>(col_slice(&self.rhat), col_slice(&self.v));
        if denom.abs_value() <= T::real_epsilon() {
            self.soft_restart();
            return self.err;
        }

        let alpha = self.rho / denom;

        for ((h, &x), &p) in col_slice_mut(&mut self.scratch)
            .iter_mut()
            .zip(col_slice(&self.x).iter())
            .zip(col_slice(&self.p).iter())
        {
            *h = x + p * alpha;
        }

        for ((s, &r), &v) in col_slice_mut(&mut self.s)
            .iter_mut()
            .zip(col_slice(&self.r).iter())
            .zip(col_slice(&self.v).iter())
        {
            *s = r - v * alpha;
        }

        let s_norm_sq = norm2_sq::<T>(col_slice(&self.s));
        if s_norm_sq <= T::real_epsilon() {
            copy_col(&mut self.x, &self.scratch);
            copy_col(&mut self.r, &self.s);
            self.err = s_norm_sq.sqrt();
            self.rho = dotc::<T>(col_slice(&self.rhat), col_slice(&self.r));
            return self.err;
        }

        self.a.apply(col_slice_mut(&mut self.t), col_slice(&self.s));

        let t_norm_sq = norm2_sq::<T>(col_slice(&self.t));
        if t_norm_sq <= T::real_epsilon() {
            copy_col(&mut self.x, &self.scratch);
            copy_col(&mut self.r, &self.s);
            self.err = norm2::<T>(col_slice(&self.r));
            self.rho = dotc::<T>(col_slice(&self.rhat), col_slice(&self.r));
            return self.err;
        }

        let omega = dotc::<T>(col_slice(&self.t), col_slice(&self.s)).scale_real(t_norm_sq.recip());

        for ((x, &h), &s) in col_slice_mut(&mut self.x)
            .iter_mut()
            .zip(col_slice(&self.scratch).iter())
            .zip(col_slice(&self.s).iter())
        {
            *x = h + s * omega;
        }

        for ((r, &s), &t) in col_slice_mut(&mut self.r)
            .iter_mut()
            .zip(col_slice(&self.s).iter())
            .zip(col_slice(&self.t).iter())
        {
            *r = s - t * omega;
        }

        let err_sq = norm2_sq::<T>(col_slice(&self.r));
        self.err = err_sq.sqrt();

        let rho_prev = self.rho;
        self.rho = dotc::<T>(col_slice(&self.rhat), col_slice(&self.r));

        let should_restart =
            err_sq > T::real_zero() && self.rho.abs_value() / err_sq < self.soft_restart_threshold;
        if should_restart
            || rho_prev.abs_value() <= T::real_epsilon()
            || omega.abs_value() <= T::real_epsilon()
        {
            self.soft_restart();
        } else {
            let beta = (self.rho / rho_prev) * (alpha / omega);

            for (((p_new, &r), &p_old), &v) in col_slice_mut(&mut self.scratch)
                .iter_mut()
                .zip(col_slice(&self.r).iter())
                .zip(col_slice(&self.p).iter())
                .zip(col_slice(&self.v).iter())
            {
                *p_new = r + (p_old - v * omega) * beta;
            }

            copy_col(&mut self.p, &self.scratch);
        }

        self.err
    }

    /// Sets the soft-restart threshold.
    #[must_use]
    pub fn with_restart_threshold(mut self, threshold: T::Real) -> Self {
        self.soft_restart_threshold = threshold;
        self
    }

    /// Current iteration count.
    pub fn iteration_count(&self) -> usize {
        self.iteration_count
    }

    /// Current soft-restart threshold.
    pub fn soft_restart_threshold(&self) -> T::Real {
        self.soft_restart_threshold
    }

    /// Number of soft restarts that have been performed.
    pub fn soft_restart_count(&self) -> usize {
        self.soft_restart_count
    }

    /// Number of hard restarts that have been performed.
    pub fn hard_restart_count(&self) -> usize {
        self.hard_restart_count
    }

    /// Latest residual norm estimate.
    pub fn err(&self) -> T::Real {
        self.err
    }

    /// Current `rho = dotc(rhat, r)`.
    pub fn rho(&self) -> T {
        self.rho
    }

    /// Problem matrix.
    pub fn a(&self) -> A {
        self.a
    }

    /// Latest solution estimate.
    pub fn x(&self) -> &Col<T> {
        &self.x
    }

    /// Right-hand side.
    pub fn b(&self) -> &Col<T> {
        &self.b
    }

    /// Latest residual vector.
    pub fn r(&self) -> &Col<T> {
        &self.r
    }

    /// Shadow residual direction.
    pub fn rhat(&self) -> &Col<T> {
        &self.rhat
    }

    /// Current search direction.
    pub fn p(&self) -> &Col<T> {
        &self.p
    }
}

#[cfg(test)]
#[cfg(not(feature = "std"))]
mod test {
    #[test]
    fn require_std_for_tests() {
        panic!("`std` feature is required for tests")
    }
}

#[cfg(feature = "std")]
#[cfg(test)]
mod test {
    use super::BiCGSTAB;
    use crate::sparse::col::{col_slice, col_slice_mut};
    use crate::sparse::compensated::norm2;
    use crate::sparse::field::Field;
    use crate::sparse::matvec::SparseMatVec;
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::mat::AsMatRef;
    use faer::matrix_free::bicgstab::{BicgParams, bicgstab as faer_bicgstab, bicgstab_scratch};
    use faer::matrix_free::{IdentityPrecond, InitialGuessStatus};
    use faer::sparse::{SparseColMat, SparseRowMat, Triplet};
    use faer::{Col, Mat, Par, c32, c64};
    use num_traits::Float;

    fn apply_to_col<T, A>(a: A, x: &[T]) -> Col<T>
    where
        T: Field,
        A: SparseMatVec<T>,
    {
        let mut out = crate::sparse::col::zero_col::<T>(a.nrows());
        a.apply_compensated(col_slice_mut(&mut out), x);
        out
    }

    fn assert_solution_close<T, A>(a: A, x_true: &[T], x0: &[T], tol: T::Real)
    where
        T: Field,
        A: SparseMatVec<T>,
    {
        let b = apply_to_col(a, x_true);
        let solver = BiCGSTAB::solve(a, x0, col_slice(&b), tol, 50).unwrap();

        let x = col_slice(solver.x());
        let mut diff = crate::sparse::col::zero_col::<T>(x.len());
        for ((dst, &lhs), &rhs) in col_slice_mut(&mut diff)
            .iter_mut()
            .zip(x.iter())
            .zip(x_true.iter())
        {
            *dst = lhs - rhs;
        }

        assert!(solver.err() < tol);
        assert!(norm2::<T>(col_slice(&diff)) < tol.sqrt());
    }

    fn residual_norm<T, A>(a: A, x: &[T], b: &[T]) -> T::Real
    where
        T: Field,
        A: SparseMatVec<T>,
    {
        let ax = apply_to_col(a, x);
        let mut residual = crate::sparse::col::zero_col::<T>(b.len());
        for ((dst, &lhs), &rhs) in col_slice_mut(&mut residual)
            .iter_mut()
            .zip(col_slice(&ax).iter())
            .zip(b.iter())
        {
            *dst = rhs - lhs;
        }

        norm2::<T>(col_slice(&residual))
    }

    #[test]
    fn solves_csr_f64_system() {
        let a = SparseRowMat::<usize, f64>::try_new_from_triplets(
            4,
            4,
            &[
                Triplet::new(0, 0, 4.0),
                Triplet::new(0, 1, -1.0),
                Triplet::new(1, 0, 2.0),
                Triplet::new(1, 1, 5.0),
                Triplet::new(1, 2, 1.0),
                Triplet::new(2, 1, 2.0),
                Triplet::new(2, 2, 4.0),
                Triplet::new(2, 3, -1.0),
                Triplet::new(3, 0, 1.0),
                Triplet::new(3, 3, 3.0),
            ],
        )
        .unwrap();

        let x_true = [1.0, -2.0, 0.5, 3.0];
        let x0 = [0.25, 0.25, 0.25, 0.25];
        assert_solution_close(a.as_ref(), &x_true, &x0, 1.0e-12);
    }

    #[test]
    fn solves_csc_f32_system() {
        let a = SparseColMat::<usize, f32>::try_new_from_triplets(
            4,
            4,
            &[
                Triplet::new(0, 0, 4.0),
                Triplet::new(0, 1, -1.0),
                Triplet::new(1, 0, 2.0),
                Triplet::new(1, 1, 5.0),
                Triplet::new(1, 2, 1.0),
                Triplet::new(2, 1, 2.0),
                Triplet::new(2, 2, 4.0),
                Triplet::new(2, 3, -1.0),
                Triplet::new(3, 0, 1.0),
                Triplet::new(3, 3, 3.0),
            ],
        )
        .unwrap();

        let x_true = [1.0f32, -2.0, 0.5, 3.0];
        let x0 = [0.1f32, 0.1, 0.1, 0.1];
        assert_solution_close(a.as_ref(), &x_true, &x0, 1.0e-4);
    }

    #[test]
    fn solves_complex_csr_system() {
        let a = SparseRowMat::<usize, c64>::try_new_from_triplets(
            3,
            3,
            &[
                Triplet::new(0, 0, c64::new(4.0, 1.0)),
                Triplet::new(0, 1, c64::new(-1.0, 0.5)),
                Triplet::new(1, 0, c64::new(2.0, -0.5)),
                Triplet::new(1, 1, c64::new(5.0, 0.0)),
                Triplet::new(1, 2, c64::new(1.0, 1.0)),
                Triplet::new(2, 1, c64::new(2.0, -1.0)),
                Triplet::new(2, 2, c64::new(3.0, 0.25)),
            ],
        )
        .unwrap();

        let x_true = [
            c64::new(1.0, -0.5),
            c64::new(-2.0, 1.0),
            c64::new(0.5, 0.25),
        ];
        let x0 = [c64::new(0.0, 0.0); 3];
        assert_solution_close(a.as_ref(), &x_true, &x0, 1.0e-11);
    }

    #[test]
    fn solves_complex_csc_system() {
        let a = SparseColMat::<usize, c32>::try_new_from_triplets(
            3,
            3,
            &[
                Triplet::new(0, 0, c32::new(4.0, 1.0)),
                Triplet::new(0, 1, c32::new(-1.0, 0.5)),
                Triplet::new(1, 0, c32::new(2.0, -0.5)),
                Triplet::new(1, 1, c32::new(5.0, 0.0)),
                Triplet::new(1, 2, c32::new(1.0, 1.0)),
                Triplet::new(2, 1, c32::new(2.0, -1.0)),
                Triplet::new(2, 2, c32::new(3.0, 0.25)),
            ],
        )
        .unwrap();

        let x_true = [
            c32::new(1.0, -0.5),
            c32::new(-2.0, 1.0),
            c32::new(0.5, 0.25),
        ];
        let x0 = [c32::new(0.0, 0.0); 3];
        assert_solution_close(a.as_ref(), &x_true, &x0, 1.0e-3);
    }

    #[test]
    fn reports_failure_when_iteration_limit_is_too_small() {
        let a = SparseRowMat::<usize, f64>::try_new_from_triplets(
            4,
            4,
            &[
                Triplet::new(0, 0, 4.0),
                Triplet::new(0, 1, -1.0),
                Triplet::new(1, 0, 2.0),
                Triplet::new(1, 1, 5.0),
                Triplet::new(1, 2, 1.0),
                Triplet::new(2, 1, 2.0),
                Triplet::new(2, 2, 4.0),
                Triplet::new(2, 3, -1.0),
                Triplet::new(3, 0, 1.0),
                Triplet::new(3, 3, 3.0),
            ],
        )
        .unwrap();

        let x_true = [1.0, -2.0, 0.5, 3.0];
        let b = apply_to_col(a.as_ref(), &x_true);
        let result = BiCGSTAB::solve(a.as_ref(), &[0.0; 4], col_slice(&b), 1.0e-12, 1);

        assert!(result.is_err());
    }

    #[test]
    fn compensated_bicgstab_matches_or_beats_faer_on_poorly_conditioned_system() {
        let n = 12usize;
        let mut triplets = Vec::with_capacity(n * n);
        for row in 0..n {
            for col in 0..n {
                let hilbert = 1.0 / (row + col + 1) as f64;
                let skew = if row > col {
                    (row - col) as f64 * 1.0e-14
                } else {
                    -((col - row) as f64) * 1.0e-14
                };
                let diag_shift = if row == col {
                    (row + 1) as f64 * 1.0e-12
                } else {
                    0.0
                };
                triplets.push(Triplet::new(row, col, hilbert + skew + diag_shift));
            }
        }

        let a = SparseRowMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();
        let x_true: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let x0 = vec![0.0; n];
        let b = apply_to_col(a.as_ref(), &x_true);
        let tol = 1.0e-8;

        let ours = match BiCGSTAB::solve(a.as_ref(), &x0, col_slice(&b), tol, 400) {
            Ok(solver) | Err(solver) => solver,
        };
        let ours_residual = residual_norm(a.as_ref(), col_slice(ours.x()), col_slice(&b));
        let ours_error = norm2::<f64>(
            &col_slice(ours.x())
                .iter()
                .zip(x_true.iter())
                .map(|(&lhs, &rhs)| lhs - rhs)
                .collect::<Vec<_>>(),
        );

        let mut faer_out = Mat::<f64>::zeros(n, 1);
        let identity = IdentityPrecond { dim: n };
        let mut params = BicgParams::default();
        params.initial_guess = InitialGuessStatus::Zero;
        params.rel_tolerance = tol;
        params.max_iters = 400;

        let faer_result = faer_bicgstab(
            faer_out.as_mut(),
            identity,
            identity,
            a.as_ref(),
            b.as_mat_ref().as_dyn(),
            params,
            |_| {},
            Par::Seq,
            MemStack::new(&mut MemBuffer::new(bicgstab_scratch(
                identity,
                identity,
                a.as_ref(),
                1,
                Par::Seq,
            ))),
        );

        let faer_x: Vec<f64> = (0..n).map(|i| faer_out[(i, 0)]).collect();
        let faer_residual = residual_norm(a.as_ref(), &faer_x, col_slice(&b));
        let faer_error = norm2::<f64>(
            &faer_x
                .iter()
                .zip(x_true.iter())
                .map(|(&lhs, &rhs)| lhs - rhs)
                .collect::<Vec<_>>(),
        );

        assert!(ours_residual.is_finite());
        assert!(ours_error.is_finite());
        assert!(faer_result.is_err() || (faer_residual.is_finite() && faer_error.is_finite()));
        assert!(faer_residual.is_nan() || ours_residual <= faer_residual);
        assert!(faer_error.is_nan() || ours_error <= faer_error);
    }
}
