use crate::sum::twosum::TwoSum;
use faer::sparse::{SparseColMatRef, SparseRowMatRef};
use faer::{Col, Index, Unbind, c32, c64};
use faer_traits::Conjugate;
use num_traits::{Float, Zero};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, Sub};

/// Scalar types supported by this BiCGSTAB implementation.
pub trait BiCGScalar:
    Conjugate<Canonical = Self>
    + Copy
    + Debug
    + Add<Output = Self>
    + AddAssign
    + Div<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
{
    /// Underlying real scalar type used for norms and tolerances.
    type Real: Float + Debug;

    /// The additive identity.
    fn zero_value() -> Self;

    /// Builds a scalar from real and imaginary parts.
    fn from_parts(real: Self::Real, imag: Self::Real) -> Self;

    /// Returns the real part.
    fn real_part(self) -> Self::Real;

    /// Returns the imaginary part.
    fn imag_part(self) -> Self::Real;

    /// Returns the complex conjugate, or `self` for real scalars.
    fn conj_value(self) -> Self;

    /// Returns the magnitude.
    fn abs_value(self) -> Self::Real;

    /// Returns the squared magnitude.
    fn abs2_value(self) -> Self::Real;

    /// Multiplies by a real scalar.
    fn scale_real(self, rhs: Self::Real) -> Self;

    #[inline]
    fn real_zero() -> Self::Real {
        <Self::Real as Zero>::zero()
    }

    #[inline]
    fn real_epsilon() -> Self::Real {
        <Self::Real as Float>::epsilon()
    }

    /// Builds a real scalar from an `f64` literal used in configuration.
    fn real_from_f64(value: f64) -> Self::Real;
}

impl BiCGScalar for f32 {
    type Real = f32;

    #[inline]
    fn zero_value() -> Self {
        0.0
    }

    #[inline]
    fn from_parts(real: Self::Real, _imag: Self::Real) -> Self {
        real
    }

    #[inline]
    fn real_part(self) -> Self::Real {
        self
    }

    #[inline]
    fn imag_part(self) -> Self::Real {
        0.0
    }

    #[inline]
    fn conj_value(self) -> Self {
        self
    }

    #[inline]
    fn abs_value(self) -> Self::Real {
        self.abs()
    }

    #[inline]
    fn abs2_value(self) -> Self::Real {
        self * self
    }

    #[inline]
    fn scale_real(self, rhs: Self::Real) -> Self {
        self * rhs
    }

    #[inline]
    fn real_from_f64(value: f64) -> Self::Real {
        value as f32
    }
}

impl BiCGScalar for f64 {
    type Real = f64;

    #[inline]
    fn zero_value() -> Self {
        0.0
    }

    #[inline]
    fn from_parts(real: Self::Real, _imag: Self::Real) -> Self {
        real
    }

    #[inline]
    fn real_part(self) -> Self::Real {
        self
    }

    #[inline]
    fn imag_part(self) -> Self::Real {
        0.0
    }

    #[inline]
    fn conj_value(self) -> Self {
        self
    }

    #[inline]
    fn abs_value(self) -> Self::Real {
        self.abs()
    }

    #[inline]
    fn abs2_value(self) -> Self::Real {
        self * self
    }

    #[inline]
    fn scale_real(self, rhs: Self::Real) -> Self {
        self * rhs
    }

    #[inline]
    fn real_from_f64(value: f64) -> Self::Real {
        value
    }
}

impl BiCGScalar for c32 {
    type Real = f32;

    #[inline]
    fn zero_value() -> Self {
        Self::new(0.0, 0.0)
    }

    #[inline]
    fn from_parts(real: Self::Real, imag: Self::Real) -> Self {
        Self::new(real, imag)
    }

    #[inline]
    fn real_part(self) -> Self::Real {
        self.re
    }

    #[inline]
    fn imag_part(self) -> Self::Real {
        self.im
    }

    #[inline]
    fn conj_value(self) -> Self {
        self.conj()
    }

    #[inline]
    fn abs_value(self) -> Self::Real {
        self.re.hypot(self.im)
    }

    #[inline]
    fn abs2_value(self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    fn scale_real(self, rhs: Self::Real) -> Self {
        self * Self::new(rhs, 0.0)
    }

    #[inline]
    fn real_from_f64(value: f64) -> Self::Real {
        value as f32
    }
}

impl BiCGScalar for c64 {
    type Real = f64;

    #[inline]
    fn zero_value() -> Self {
        Self::new(0.0, 0.0)
    }

    #[inline]
    fn from_parts(real: Self::Real, imag: Self::Real) -> Self {
        Self::new(real, imag)
    }

    #[inline]
    fn real_part(self) -> Self::Real {
        self.re
    }

    #[inline]
    fn imag_part(self) -> Self::Real {
        self.im
    }

    #[inline]
    fn conj_value(self) -> Self {
        self.conj()
    }

    #[inline]
    fn abs_value(self) -> Self::Real {
        self.re.hypot(self.im)
    }

    #[inline]
    fn abs2_value(self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    fn scale_real(self, rhs: Self::Real) -> Self {
        self * Self::new(rhs, 0.0)
    }

    #[inline]
    fn real_from_f64(value: f64) -> Self::Real {
        value
    }
}

/// Sparse matrix-vector product interface used by `BiCGSTAB`.
pub trait SparseMatVec<T: BiCGScalar>: Copy + Debug {
    /// Number of rows.
    fn nrows(&self) -> usize;

    /// Number of columns.
    fn ncols(&self) -> usize;

    /// Computes `out = self * rhs`.
    fn apply(&self, out: &mut [T], rhs: &[T]);

    /// Computes `out = self * rhs` with compensated accumulation.
    fn apply_compensated(&self, out: &mut [T], rhs: &[T]);
}

impl<T, I, ViewT> SparseMatVec<T> for SparseRowMatRef<'_, I, ViewT>
where
    T: BiCGScalar,
    I: Index,
    ViewT: Conjugate<Canonical = T>,
{
    #[inline]
    fn nrows(&self) -> usize {
        self.symbolic().nrows().unbound()
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.symbolic().ncols().unbound()
    }

    #[inline]
    fn apply(&self, out: &mut [T], rhs: &[T]) {
        let matrix = self.canonical();
        let nrows = matrix.nrows().unbound();
        let ncols = matrix.ncols().unbound();

        assert_eq!(out.len(), nrows);
        assert_eq!(rhs.len(), ncols);

        let row_ptr = matrix.row_ptr();
        let col_idx = matrix.col_idx();
        let values = matrix.val();

        out.fill(T::zero_value());
        for row in 0..nrows {
            let start = row_ptr[row].zx();
            let end = row_ptr[row + 1].zx();
            let mut sum = T::zero_value();

            for idx in start..end {
                sum += values[idx] * rhs[col_idx[idx].zx()];
            }

            out[row] = sum;
        }
    }

    #[inline]
    fn apply_compensated(&self, out: &mut [T], rhs: &[T]) {
        let matrix = self.canonical();
        let nrows = matrix.nrows().unbound();
        let ncols = matrix.ncols().unbound();

        assert_eq!(out.len(), nrows);
        assert_eq!(rhs.len(), ncols);

        let row_ptr = matrix.row_ptr();
        let col_idx = matrix.col_idx();
        let values = matrix.val();

        out.fill(T::zero_value());
        for row in 0..nrows {
            let start = row_ptr[row].zx();
            let end = row_ptr[row + 1].zx();
            let mut acc = CompensatedSum::<T>::default();

            for idx in start..end {
                acc.add(values[idx] * rhs[col_idx[idx].zx()]);
            }

            out[row] = acc.finish();
        }
    }
}

impl<T, I, ViewT> SparseMatVec<T> for SparseColMatRef<'_, I, ViewT>
where
    T: BiCGScalar,
    I: Index,
    ViewT: Conjugate<Canonical = T>,
{
    #[inline]
    fn nrows(&self) -> usize {
        self.symbolic().nrows().unbound()
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.symbolic().ncols().unbound()
    }

    #[inline]
    fn apply(&self, out: &mut [T], rhs: &[T]) {
        let matrix = self.canonical();
        let nrows = matrix.nrows().unbound();
        let ncols = matrix.ncols().unbound();

        assert_eq!(out.len(), nrows);
        assert_eq!(rhs.len(), ncols);

        let col_ptr = matrix.col_ptr();
        let row_idx = matrix.row_idx();
        let values = matrix.val();

        out.fill(T::zero_value());
        for col in 0..ncols {
            let rhs_value = rhs[col];
            let start = col_ptr[col].zx();
            let end = col_ptr[col + 1].zx();

            for idx in start..end {
                out[row_idx[idx].zx()] += values[idx] * rhs_value;
            }
        }
    }

    #[inline]
    fn apply_compensated(&self, out: &mut [T], rhs: &[T]) {
        let matrix = self.canonical();
        let nrows = matrix.nrows().unbound();
        let ncols = matrix.ncols().unbound();

        assert_eq!(out.len(), nrows);
        assert_eq!(rhs.len(), ncols);

        let col_ptr = matrix.col_ptr();
        let row_idx = matrix.row_idx();
        let values = matrix.val();
        let mut acc = vec![CompensatedSum::<T>::default(); nrows];

        for col in 0..ncols {
            let rhs_value = rhs[col];
            let start = col_ptr[col].zx();
            let end = col_ptr[col + 1].zx();

            for idx in start..end {
                acc[row_idx[idx].zx()].add(values[idx] * rhs_value);
            }
        }

        for (dst, acc) in out.iter_mut().zip(acc.into_iter()) {
            *dst = acc.finish();
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RealCompensatedSum<R: Float> {
    acc: Option<TwoSum<R>>,
}

impl<R: Float> Default for RealCompensatedSum<R> {
    #[inline]
    fn default() -> Self {
        Self { acc: None }
    }
}

impl<R: Float> RealCompensatedSum<R> {
    #[inline]
    fn add(&mut self, value: R) {
        match self.acc.as_mut() {
            Some(acc) => acc.add(value),
            None => self.acc = Some(TwoSum::new(value)),
        }
    }

    #[inline]
    fn finish(self) -> R {
        match self.acc {
            Some(acc) => {
                let (sum, residual) = acc.finish();
                sum + residual
            }
            None => R::zero(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CompensatedSum<T: BiCGScalar> {
    real: RealCompensatedSum<T::Real>,
    imag: RealCompensatedSum<T::Real>,
}

impl<T: BiCGScalar> Default for CompensatedSum<T> {
    #[inline]
    fn default() -> Self {
        Self {
            real: RealCompensatedSum::default(),
            imag: RealCompensatedSum::default(),
        }
    }
}

impl<T: BiCGScalar> CompensatedSum<T> {
    #[inline]
    fn add(&mut self, value: T) {
        self.real.add(value.real_part());
        self.imag.add(value.imag_part());
    }

    #[inline]
    fn finish(self) -> T {
        T::from_parts(self.real.finish(), self.imag.finish())
    }
}

#[inline]
fn col_from_slice<T: BiCGScalar>(values: &[T]) -> Col<T> {
    Col::from_fn(values.len(), |i| values[i.unbound()])
}

#[inline]
fn zero_col<T: BiCGScalar>(len: usize) -> Col<T> {
    Col::from_fn(len, |_| T::zero_value())
}

#[inline]
fn col_slice<T>(col: &Col<T>) -> &[T] {
    col.try_as_col_major().unwrap().as_slice()
}

#[inline]
fn col_slice_mut<T>(col: &mut Col<T>) -> &mut [T] {
    col.try_as_col_major_mut().unwrap().as_slice_mut()
}

#[inline]
fn copy_col<T: BiCGScalar>(dst: &mut Col<T>, src: &Col<T>) {
    col_slice_mut(dst).copy_from_slice(col_slice(src));
}

#[inline]
fn dotc<T: BiCGScalar>(lhs: &[T], rhs: &[T]) -> T {
    assert_eq!(lhs.len(), rhs.len());

    let mut acc = CompensatedSum::<T>::default();
    for (&lhs, &rhs) in lhs.iter().zip(rhs.iter()) {
        acc.add(lhs.conj_value() * rhs);
    }

    acc.finish()
}

#[inline]
fn norm2_sq<T: BiCGScalar>(values: &[T]) -> T::Real {
    let mut acc = RealCompensatedSum::<T::Real>::default();
    for &value in values {
        acc.add(value.abs2_value());
    }

    acc.finish()
}

#[inline]
fn norm2<T: BiCGScalar>(values: &[T]) -> T::Real {
    norm2_sq::<T>(values).sqrt()
}

/// Stabilized bi-conjugate gradient solver.
#[derive(Debug)]
pub struct BiCGSTAB<T: BiCGScalar, A: SparseMatVec<T>> {
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

impl<T: BiCGScalar, A: SparseMatVec<T>> BiCGSTAB<T, A> {
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
    use super::{BiCGSTAB, BiCGScalar, SparseMatVec, dotc, norm2};
    use faer::sparse::{SparseColMat, SparseRowMat, Triplet};
    use faer::{Col, c32, c64};
    use num_traits::Float;

    fn apply_to_col<T, A>(a: A, x: &[T]) -> Col<T>
    where
        T: BiCGScalar,
        A: SparseMatVec<T>,
    {
        let mut out = super::zero_col::<T>(a.nrows());
        a.apply_compensated(super::col_slice_mut(&mut out), x);
        out
    }

    fn assert_solution_close<T, A>(a: A, x_true: &[T], x0: &[T], tol: T::Real)
    where
        T: BiCGScalar,
        A: SparseMatVec<T>,
    {
        let b = apply_to_col(a, x_true);
        let solver = BiCGSTAB::solve(a, x0, super::col_slice(&b), tol, 50).unwrap();

        let x = super::col_slice(solver.x());
        let mut diff = super::zero_col::<T>(x.len());
        for ((dst, &lhs), &rhs) in super::col_slice_mut(&mut diff)
            .iter_mut()
            .zip(x.iter())
            .zip(x_true.iter())
        {
            *dst = lhs - rhs;
        }

        assert!(solver.err() < tol);
        assert!(norm2::<T>(super::col_slice(&diff)) < tol.sqrt());
    }

    #[test]
    fn compensated_dotc_handles_real_cancellation() {
        let lhs = [1.0e16f64, 1.0, -1.0e16];
        let rhs = [1.0f64, 1.0, 1.0];

        assert_eq!(dotc(&lhs, &rhs), 1.0);
    }

    #[test]
    fn compensated_dotc_uses_conjugation_for_complex_inputs() {
        let lhs = [c64::new(1.0, 2.0), c64::new(-3.0, 4.0)];
        let rhs = [c64::new(5.0, -1.0), c64::new(2.0, 3.0)];
        let dot = dotc(&lhs, &rhs);

        let expected = lhs[0].conj() * rhs[0] + lhs[1].conj() * rhs[1];
        assert_eq!(dot, expected);
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
        let result = BiCGSTAB::solve(a.as_ref(), &[0.0; 4], super::col_slice(&b), 1.0e-12, 1);

        assert!(result.is_err());
    }
}
