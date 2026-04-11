//! Balanced truncation for dense and low-rank state-space models.
//!
//! The control module now has the full Gramian solver set needed for balanced
//! truncation:
//!
//! - dense continuous Lyapunov
//! - sparse continuous low-rank Lyapunov
//! - dense discrete Stein
//! - sparse discrete low-rank Stein
//!
//! This module sits above those solvers and performs the balancing step:
//!
//! 1. obtain controllability and observability Gramian factors
//! 2. form the dense balancing core `Ro^H Rc`
//! 3. compute its SVD to get the Hankel singular values
//! 4. build the left/right projection factors
//! 5. assemble the reduced state-space system
//!
//! The returned result always includes the actual projection operators used to
//! build the reduced model, and can optionally retain more of the internal
//! balancing algebra on request.

use super::lyapunov::{
    LowRankFactor, LyapunovError, LyapunovParams, ShiftStrategy, controllability_gramian_dense,
    controllability_gramian_low_rank, observability_gramian_dense, observability_gramian_low_rank,
};
use super::state_space::{ContinuousStateSpace, ContinuousTime, DiscreteStateSpace, DiscreteTime};
use super::stein::{
    SteinError, controllability_gramian_discrete_dense, controllability_gramian_discrete_low_rank,
    observability_gramian_discrete_dense, observability_gramian_discrete_low_rank,
};
use crate::decomp::{DecompError, DenseDecompParams, dense_self_adjoint_eigen, dense_svd};
use crate::sparse::compensated::{CompensatedField, CompensatedSum};
use faer::sparse::SparseColMatRef;
use faer::{Col, Index, Mat, MatRef, Unbind};
use faer_traits::ComplexField;
use faer_traits::Conjugate;
use faer_traits::ext::ComplexFieldExt;
use num_traits::{Float, One, Zero};
use std::fmt;

/// Controls how much internal balancing data is retained in the result.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InternalsLevel {
    /// Return only the reduced model, Hankel singular values, error bound, and
    /// the final left/right projection operators.
    #[default]
    Summary,
    /// Also retain the Gramian factors and balancing-core SVD data.
    Factors,
    /// Also retain dense Gramian and dense square-root data when that is
    /// mathematically available and practical to materialize.
    Full,
}

impl InternalsLevel {
    #[inline]
    fn keep_factors(self) -> bool {
        !matches!(self, Self::Summary)
    }

    #[inline]
    fn keep_full(self) -> bool {
        matches!(self, Self::Full)
    }
}

/// Builder-style parameters for balanced truncation.
///
/// The truncation decision is driven by the Hankel singular values of the
/// balancing core. `order` chooses how many balanced modes to keep, while
/// `sigma_tol` can be used to drop modes whose Hankel singular values are too
/// small to treat as numerically meaningful.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BalancedParams<R> {
    /// Requested reduced order. `None` keeps all numerically retained balanced
    /// modes.
    pub order: Option<usize>,
    /// Optional Hankel singular value tolerance. If omitted, a relative default
    /// based on machine precision is used.
    pub sigma_tol: Option<R>,
    /// Requested internal detail level for the returned result.
    pub internals: InternalsLevel,
}

impl<R> Default for BalancedParams<R> {
    fn default() -> Self {
        Self {
            order: None,
            sigma_tol: None,
            internals: InternalsLevel::Summary,
        }
    }
}

impl<R> BalancedParams<R> {
    /// Creates parameters with documented defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests a specific reduced order.
    #[must_use]
    pub fn with_order(mut self, order: usize) -> Self {
        self.order = Some(order);
        self
    }

    /// Overrides the Hankel singular value retention tolerance.
    #[must_use]
    pub fn with_sigma_tol(mut self, sigma_tol: R) -> Self {
        self.sigma_tol = Some(sigma_tol);
        self
    }

    /// Chooses how much internal balancing data to retain.
    #[must_use]
    pub fn with_internals(mut self, internals: InternalsLevel) -> Self {
        self.internals = internals;
        self
    }
}

/// Optional retained internal balancing data.
///
/// These values expose the algebra that produced the reduced model. In
/// particular, `controllability_factor`, `observability_factor`, and the
/// balancing-core SVD data are the direct ingredients used to build the final
/// projection operators.
#[derive(Clone, Debug)]
pub struct BalancedInternals<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    /// Controllability Gramian factor used in the balancing core.
    pub controllability_factor: Mat<T>,
    /// Observability Gramian factor used in the balancing core.
    pub observability_factor: Mat<T>,
    /// Dense balancing core `Ro^H Rc`.
    pub core: Mat<T>,
    /// Left singular vectors of the balancing core.
    pub core_u: Mat<T>,
    /// Singular values of the balancing core.
    pub core_singular_values: Col<T::Real>,
    /// Adjoint of the right singular vector matrix of the balancing core.
    pub core_vh: Mat<T>,
    /// Dense controllability Gramian, when retained.
    pub dense_controllability_gramian: Option<Mat<T>>,
    /// Dense observability Gramian, when retained.
    pub dense_observability_gramian: Option<Mat<T>>,
    /// Dense controllability square-root factor, when retained.
    pub dense_controllability_sqrt: Option<Mat<T>>,
    /// Dense observability square-root factor, when retained.
    pub dense_observability_sqrt: Option<Mat<T>>,
}

/// Result of balanced truncation.
///
/// The reduced model is accompanied by the actual left/right projection
/// operators used to assemble it. Those transforms are returned even at the
/// summary level so callers can inspect or reuse the balancing map directly.
#[derive(Clone, Debug)]
pub struct BalancedTruncationResult<T: CompensatedField, Domain>
where
    T::Real: Float + Copy,
{
    /// Reduced state-space model.
    pub reduced: super::state_space::StateSpace<T, Domain>,
    /// Hankel singular values in descending order.
    pub hankel_singular_values: Col<T::Real>,
    /// Final retained order.
    pub reduced_order: usize,
    /// Standard balanced-truncation tail bound when available.
    pub error_bound: Option<T::Real>,
    /// Left projection operator used to build the reduced model.
    pub left_projection: Mat<T>,
    /// Right projection operator used to build the reduced model.
    pub right_projection: Mat<T>,
    /// Optional retained internal balancing data.
    pub internals: Option<BalancedInternals<T>>,
}

/// Errors produced by balanced-truncation front-ends.
#[derive(Debug)]
pub enum BalancedError<R> {
    /// Dense or sparse continuous Gramian solve failed.
    Lyapunov(LyapunovError),
    /// Dense or sparse discrete Gramian solve failed.
    Stein(SteinError),
    /// Dense balancing decomposition failed.
    Decomposition(DecompError),
    /// Reduced state-space construction failed.
    StateSpace(super::state_space::StateSpaceError),
    /// The requested reduced order exceeds the available core dimension.
    InvalidOrder { requested: usize, available: usize },
    /// No numerically meaningful balanced modes remained.
    EmptyRetainedSpectrum,
    /// A dense Gramian expected to be PSD had a clearly negative eigenvalue.
    IndefiniteGramian { which: &'static str, eigenvalue: R },
    /// A balancing intermediate became non-finite.
    NonFiniteResult { which: &'static str },
}

impl<R: fmt::Debug> fmt::Display for BalancedError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl<R: fmt::Debug> std::error::Error for BalancedError<R> {}

impl<R> From<LyapunovError> for BalancedError<R> {
    fn from(value: LyapunovError) -> Self {
        Self::Lyapunov(value)
    }
}

impl<R> From<SteinError> for BalancedError<R> {
    fn from(value: SteinError) -> Self {
        Self::Stein(value)
    }
}

impl<R> From<DecompError> for BalancedError<R> {
    fn from(value: DecompError) -> Self {
        Self::Decomposition(value)
    }
}

impl<R> From<super::state_space::StateSpaceError> for BalancedError<R> {
    fn from(value: super::state_space::StateSpaceError) -> Self {
        Self::StateSpace(value)
    }
}

#[derive(Clone, Debug)]
struct DenseFactorData<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    factor: Mat<T>,
    dense_gramian: Mat<T>,
    dense_sqrt: Mat<T>,
}

#[derive(Clone, Debug)]
struct BalanceCore<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    hankel_singular_values: Col<T::Real>,
    reduced_order: usize,
    error_bound: Option<T::Real>,
    left_projection: Mat<T>,
    right_projection: Mat<T>,
    core: Mat<T>,
    core_u: Mat<T>,
    core_singular_values: Col<T::Real>,
    core_vh: Mat<T>,
}

/// Computes dense continuous-time balanced truncation.
///
/// This path solves the dense continuous controllability and observability
/// Gramians, converts each Gramian into a PSD square-root factor, forms the
/// balancing core `Ro^H Rc`, and then builds the reduced model from the
/// resulting left/right projection operators.
pub fn balanced_truncation_continuous_dense<T>(
    system: &ContinuousStateSpace<T>,
    params: &BalancedParams<T::Real>,
) -> Result<BalancedTruncationResult<T, ContinuousTime>, BalancedError<T::Real>>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let wc = controllability_gramian_dense(system.a(), system.b())?;
    let wo = observability_gramian_dense(system.a(), system.c())?;
    let wc_factor = dense_psd_factor("controllability", wc.solution.as_ref())?;
    let wo_factor = dense_psd_factor("observability", wo.solution.as_ref())?;

    let core = build_balance_core(wc_factor.factor.as_ref(), wo_factor.factor.as_ref(), params)?;
    let reduced = build_dense_reduced_system(
        system.a(),
        system.b(),
        system.c(),
        system.d(),
        core.left_projection.as_ref(),
        core.right_projection.as_ref(),
    )?;

    let internals = dense_internals(params.internals, wc_factor, wo_factor, &core);
    Ok(BalancedTruncationResult {
        reduced,
        hankel_singular_values: core.hankel_singular_values.clone(),
        reduced_order: core.reduced_order,
        error_bound: core.error_bound,
        left_projection: core.left_projection.clone(),
        right_projection: core.right_projection.clone(),
        internals,
    })
}

/// Computes dense discrete-time balanced truncation.
///
/// The balancing algebra is identical to the continuous-time path; only the
/// source of the Gramians changes from Lyapunov solves to discrete Stein
/// solves.
pub fn balanced_truncation_discrete_dense<T>(
    system: &DiscreteStateSpace<T>,
    params: &BalancedParams<T::Real>,
) -> Result<BalancedTruncationResult<T, DiscreteTime<T::Real>>, BalancedError<T::Real>>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let wc = controllability_gramian_discrete_dense(system.a(), system.b())?;
    let wo = observability_gramian_discrete_dense(system.a(), system.c())?;
    let wc_factor = dense_psd_factor("controllability", wc.solution.as_ref())?;
    let wo_factor = dense_psd_factor("observability", wo.solution.as_ref())?;

    let core = build_balance_core(wc_factor.factor.as_ref(), wo_factor.factor.as_ref(), params)?;
    let reduced = build_dense_reduced_discrete_system(
        system.a(),
        system.b(),
        system.c(),
        system.d(),
        system.sample_time(),
        core.left_projection.as_ref(),
        core.right_projection.as_ref(),
    )?;

    let internals = dense_internals(params.internals, wc_factor, wo_factor, &core);
    Ok(BalancedTruncationResult {
        reduced,
        hankel_singular_values: core.hankel_singular_values.clone(),
        reduced_order: core.reduced_order,
        error_bound: core.error_bound,
        left_projection: core.left_projection.clone(),
        right_projection: core.right_projection.clone(),
        internals,
    })
}

/// Computes sparse continuous-time low-rank balanced truncation.
///
/// This path never forms dense full Gramians. It uses the low-rank factors
/// returned by the sparse Lyapunov solver directly, so the only dense object
/// introduced by the reduction step is the small balancing core `Zo^H Zc`.
pub fn balanced_truncation_continuous_low_rank<I, T, ViewT>(
    a: SparseColMatRef<'_, I, ViewT>,
    b: MatRef<'_, T>,
    c: MatRef<'_, T>,
    d: MatRef<'_, T>,
    shifts: &ShiftStrategy<T>,
    gramian_params: LyapunovParams<T::Real>,
    params: &BalancedParams<T::Real>,
) -> Result<BalancedTruncationResult<T, ContinuousTime>, BalancedError<T::Real>>
where
    I: Index,
    T: CompensatedField,
    T::Real: Float + Copy,
    ViewT: Conjugate<Canonical = T>,
{
    validate_sparse_system_dims(a.nrows().unbound(), b, c, d)?;

    let wc = controllability_gramian_low_rank(a, b, shifts, gramian_params)?;
    let wo = observability_gramian_low_rank(a, c, shifts, gramian_params)?;
    let core = build_balance_core(wc.factor.z.as_ref(), wo.factor.z.as_ref(), params)?;
    let reduced = build_sparse_reduced_system(
        a.canonical(),
        b,
        c,
        d,
        core.left_projection.as_ref(),
        core.right_projection.as_ref(),
    )?;

    let internals = low_rank_internals(params.internals, wc.factor, wo.factor, &core);
    Ok(BalancedTruncationResult {
        reduced,
        hankel_singular_values: core.hankel_singular_values.clone(),
        reduced_order: core.reduced_order,
        error_bound: core.error_bound,
        left_projection: core.left_projection.clone(),
        right_projection: core.right_projection.clone(),
        internals,
    })
}

/// Computes sparse discrete-time low-rank balanced truncation.
///
/// As in the sparse continuous-time path, the discrete solver contributes only
/// low-rank Gramian factors. The reduction itself stays factor-based and avoids
/// materializing any dense `n x n` discrete Gramian.
pub fn balanced_truncation_discrete_low_rank<I, T, ViewT>(
    a: SparseColMatRef<'_, I, ViewT>,
    b: MatRef<'_, T>,
    c: MatRef<'_, T>,
    d: MatRef<'_, T>,
    sample_time: T::Real,
    shifts: &ShiftStrategy<T>,
    gramian_params: LyapunovParams<T::Real>,
    params: &BalancedParams<T::Real>,
) -> Result<BalancedTruncationResult<T, DiscreteTime<T::Real>>, BalancedError<T::Real>>
where
    I: Index,
    T: CompensatedField,
    T::Real: Float + Copy,
    ViewT: Conjugate<Canonical = T>,
{
    validate_sparse_system_dims(a.nrows().unbound(), b, c, d)?;
    if !sample_time.is_finite() || sample_time <= <T::Real as Zero>::zero() {
        return Err(BalancedError::StateSpace(
            super::state_space::StateSpaceError::InvalidSampleTime,
        ));
    }

    let wc = controllability_gramian_discrete_low_rank(a, b, shifts, gramian_params)?;
    let wo = observability_gramian_discrete_low_rank(a, c, shifts, gramian_params)?;
    let core = build_balance_core(wc.factor.z.as_ref(), wo.factor.z.as_ref(), params)?;
    let reduced = build_sparse_reduced_discrete_system(
        a.canonical(),
        b,
        c,
        d,
        sample_time,
        core.left_projection.as_ref(),
        core.right_projection.as_ref(),
    )?;

    let internals = low_rank_internals(params.internals, wc.factor, wo.factor, &core);
    Ok(BalancedTruncationResult {
        reduced,
        hankel_singular_values: core.hankel_singular_values.clone(),
        reduced_order: core.reduced_order,
        error_bound: core.error_bound,
        left_projection: core.left_projection.clone(),
        right_projection: core.right_projection.clone(),
        internals,
    })
}

impl<T> ContinuousStateSpace<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    /// Computes dense continuous-time balanced truncation for this model.
    pub fn balanced_truncation(
        &self,
        params: &BalancedParams<T::Real>,
    ) -> Result<BalancedTruncationResult<T, ContinuousTime>, BalancedError<T::Real>> {
        balanced_truncation_continuous_dense(self, params)
    }
}

impl<T> DiscreteStateSpace<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    /// Computes dense discrete-time balanced truncation for this model.
    pub fn balanced_truncation(
        &self,
        params: &BalancedParams<T::Real>,
    ) -> Result<BalancedTruncationResult<T, DiscreteTime<T::Real>>, BalancedError<T::Real>> {
        balanced_truncation_discrete_dense(self, params)
    }
}

fn dense_internals<T>(
    level: InternalsLevel,
    wc: DenseFactorData<T>,
    wo: DenseFactorData<T>,
    core: &BalanceCore<T>,
) -> Option<BalancedInternals<T>>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    if !level.keep_factors() {
        return None;
    }

    Some(BalancedInternals {
        controllability_factor: wc.factor,
        observability_factor: wo.factor,
        core: core.core.clone(),
        core_u: core.core_u.clone(),
        core_singular_values: core.core_singular_values.clone(),
        core_vh: core.core_vh.clone(),
        dense_controllability_gramian: level.keep_full().then_some(wc.dense_gramian),
        dense_observability_gramian: level.keep_full().then_some(wo.dense_gramian),
        dense_controllability_sqrt: level.keep_full().then_some(wc.dense_sqrt),
        dense_observability_sqrt: level.keep_full().then_some(wo.dense_sqrt),
    })
}

fn low_rank_internals<T>(
    level: InternalsLevel,
    wc: LowRankFactor<T>,
    wo: LowRankFactor<T>,
    core: &BalanceCore<T>,
) -> Option<BalancedInternals<T>>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    if !level.keep_factors() {
        return None;
    }

    Some(BalancedInternals {
        controllability_factor: wc.z,
        observability_factor: wo.z,
        core: core.core.clone(),
        core_u: core.core_u.clone(),
        core_singular_values: core.core_singular_values.clone(),
        core_vh: core.core_vh.clone(),
        dense_controllability_gramian: None,
        dense_observability_gramian: None,
        dense_controllability_sqrt: None,
        dense_observability_sqrt: None,
    })
}

fn dense_psd_factor<T>(
    which: &'static str,
    gramian: MatRef<'_, T>,
) -> Result<DenseFactorData<T>, BalancedError<T::Real>>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    // Balanced truncation only needs a Gramian square root, not the Gramian
    // itself. Using a self-adjoint eigendecomposition here is more robust than
    // assuming strict positive definiteness and requiring Cholesky.
    let eig = dense_self_adjoint_eigen(gramian, &DenseDecompParams::<T>::new())?;
    let mut max_abs = <T::Real as Zero>::zero();
    for i in 0..eig.values.nrows() {
        let abs = eig.values[i].abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    let eig_tol = <T::Real as Float>::epsilon().sqrt() * max_abs;
    let mut kept = Vec::new();
    for i in 0..eig.values.nrows() {
        let lambda = eig.values[i].real();
        // Tiny negative eigenvalues are treated as roundoff noise; clearly
        // negative ones mean the solved Gramian is not PSD enough to support a
        // meaningful balancing factor.
        if lambda < -eig_tol {
            return Err(BalancedError::IndefiniteGramian {
                which,
                eigenvalue: lambda,
            });
        }
        if lambda > eig_tol {
            kept.push((i, lambda.sqrt()));
        }
    }

    let factor = Mat::from_fn(gramian.nrows(), kept.len(), |row, col| {
        eig.vectors[(row, kept[col].0)].mul_real(kept[col].1)
    });

    if !factor.as_ref().is_all_finite() {
        return Err(BalancedError::NonFiniteResult { which });
    }

    Ok(DenseFactorData {
        factor: factor.clone(),
        dense_gramian: clone_mat(gramian),
        dense_sqrt: factor,
    })
}

fn build_balance_core<T>(
    controllability_factor: MatRef<'_, T>,
    observability_factor: MatRef<'_, T>,
    params: &BalancedParams<T::Real>,
) -> Result<BalanceCore<T>, BalancedError<T::Real>>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    // The dense balancing core is small even when the original system is
    // sparse, because it is built from Gramian factors rather than from the
    // full Gramians themselves.
    let core = dense_mul_adjoint_lhs(observability_factor, controllability_factor);
    let svd = dense_svd(core.as_ref(), &DenseDecompParams::<T>::new())?;
    let hankel_singular_values = Col::from_fn(svd.s.nrows(), |i| svd.s[i].abs());

    let mut max_sigma = <T::Real as Zero>::zero();
    for i in 0..hankel_singular_values.nrows() {
        let sigma = hankel_singular_values[i];
        if sigma > max_sigma {
            max_sigma = sigma;
        }
    }
    let sigma_tol = params
        .sigma_tol
        .unwrap_or_else(|| <T::Real as Float>::epsilon().sqrt() * max_sigma);
    // `available` counts the modes that still look numerically meaningful after
    // applying the retention threshold. Callers can either request an explicit
    // order or let the solver keep this whole retained set.
    let available = (0..hankel_singular_values.nrows())
        .take_while(|&i| hankel_singular_values[i] > sigma_tol)
        .count();

    let reduced_order = match params.order {
        Some(order) if order > hankel_singular_values.nrows() => {
            return Err(BalancedError::InvalidOrder {
                requested: order,
                available: hankel_singular_values.nrows(),
            });
        }
        Some(order) if params.sigma_tol.is_some() => order.min(available),
        Some(order) if order > available => {
            return Err(BalancedError::InvalidOrder {
                requested: order,
                available,
            });
        }
        Some(order) => order,
        None => available,
    };

    if reduced_order == 0 && params.order != Some(0) {
        return Err(BalancedError::EmptyRetainedSpectrum);
    }

    let error_bound = Some(
        (<T::Real as One>::one() + <T::Real as One>::one())
            * tail_sum(hankel_singular_values.as_ref(), reduced_order),
    );

    // The final projection factors are exactly the square-root balanced
    // truncation formulas:
    //
    // T_r = Rc V_r Sigma_r^{-1/2}
    // S_r = Ro U_r Sigma_r^{-1/2}
    let right_projection = build_projection(
        controllability_factor,
        svd.v.as_ref(),
        hankel_singular_values.as_ref(),
        reduced_order,
    );
    let left_projection = build_projection(
        observability_factor,
        svd.u.as_ref(),
        hankel_singular_values.as_ref(),
        reduced_order,
    );
    if !right_projection.as_ref().is_all_finite() {
        return Err(BalancedError::NonFiniteResult {
            which: "right_projection",
        });
    }
    if !left_projection.as_ref().is_all_finite() {
        return Err(BalancedError::NonFiniteResult {
            which: "left_projection",
        });
    }

    Ok(BalanceCore {
        hankel_singular_values: hankel_singular_values.clone(),
        reduced_order,
        error_bound,
        left_projection,
        right_projection,
        core,
        core_u: svd.u,
        core_singular_values: hankel_singular_values,
        core_vh: dense_adjoint(svd.v.as_ref()),
    })
}

fn build_projection<T>(
    factor: MatRef<'_, T>,
    basis: MatRef<'_, T>,
    hankel_singular_values: faer::ColRef<'_, T::Real>,
    order: usize,
) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    // Scale the retained singular-vector basis by Sigma_r^{-1/2} before
    // applying the controllability/observability factor. This is the step that
    // converts the core SVD into a state-space projection.
    let scaled_basis = Mat::from_fn(basis.nrows(), order, |row, col| {
        let sigma_inv_sqrt = hankel_singular_values[col].sqrt().recip();
        basis[(row, col)].mul_real(sigma_inv_sqrt)
    });
    dense_mul(factor, scaled_basis.as_ref())
}

fn build_dense_reduced_system<T>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    c: MatRef<'_, T>,
    d: MatRef<'_, T>,
    left_projection: MatRef<'_, T>,
    right_projection: MatRef<'_, T>,
) -> Result<ContinuousStateSpace<T>, BalancedError<T::Real>>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let ar = dense_mul_adjoint_lhs(left_projection, dense_mul(a, right_projection).as_ref());
    let br = dense_mul_adjoint_lhs(left_projection, b);
    let cr = dense_mul(c, right_projection);
    let dr = clone_mat(d);
    Ok(ContinuousStateSpace::new(ar, br, cr, dr)?)
}

fn build_dense_reduced_discrete_system<T>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    c: MatRef<'_, T>,
    d: MatRef<'_, T>,
    sample_time: T::Real,
    left_projection: MatRef<'_, T>,
    right_projection: MatRef<'_, T>,
) -> Result<DiscreteStateSpace<T>, BalancedError<T::Real>>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let ar = dense_mul_adjoint_lhs(left_projection, dense_mul(a, right_projection).as_ref());
    let br = dense_mul_adjoint_lhs(left_projection, b);
    let cr = dense_mul(c, right_projection);
    let dr = clone_mat(d);
    Ok(DiscreteStateSpace::new(ar, br, cr, dr, sample_time)?)
}

fn build_sparse_reduced_system<I, T>(
    a: SparseColMatRef<'_, I, T>,
    b: MatRef<'_, T>,
    c: MatRef<'_, T>,
    d: MatRef<'_, T>,
    left_projection: MatRef<'_, T>,
    right_projection: MatRef<'_, T>,
) -> Result<ContinuousStateSpace<T>, BalancedError<T::Real>>
where
    I: Index,
    T: CompensatedField,
    T::Real: Float + Copy,
{
    // The reduced sparse paths still return ordinary dense reduced models. The
    // expensive sparse object is only used while applying `A` to the retained
    // right projection.
    let ar = dense_mul_adjoint_lhs(
        left_projection,
        sparse_matmul_dense(a, right_projection).as_ref(),
    );
    let br = dense_mul_adjoint_lhs(left_projection, b);
    let cr = dense_mul(c, right_projection);
    let dr = clone_mat(d);
    Ok(ContinuousStateSpace::new(ar, br, cr, dr)?)
}

fn build_sparse_reduced_discrete_system<I, T>(
    a: SparseColMatRef<'_, I, T>,
    b: MatRef<'_, T>,
    c: MatRef<'_, T>,
    d: MatRef<'_, T>,
    sample_time: T::Real,
    left_projection: MatRef<'_, T>,
    right_projection: MatRef<'_, T>,
) -> Result<DiscreteStateSpace<T>, BalancedError<T::Real>>
where
    I: Index,
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let ar = dense_mul_adjoint_lhs(
        left_projection,
        sparse_matmul_dense(a, right_projection).as_ref(),
    );
    let br = dense_mul_adjoint_lhs(left_projection, b);
    let cr = dense_mul(c, right_projection);
    let dr = clone_mat(d);
    Ok(DiscreteStateSpace::new(ar, br, cr, dr, sample_time)?)
}

fn validate_sparse_system_dims<T>(
    nstates: usize,
    b: MatRef<'_, T>,
    c: MatRef<'_, T>,
    d: MatRef<'_, T>,
) -> Result<(), BalancedError<T::Real>>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    if b.nrows() != nstates {
        return Err(BalancedError::StateSpace(
            super::state_space::StateSpaceError::DimensionMismatch {
                which: "b",
                expected_nrows: nstates,
                expected_ncols: b.ncols(),
                actual_nrows: b.nrows(),
                actual_ncols: b.ncols(),
            },
        ));
    }
    if c.ncols() != nstates {
        return Err(BalancedError::StateSpace(
            super::state_space::StateSpaceError::DimensionMismatch {
                which: "c",
                expected_nrows: c.nrows(),
                expected_ncols: nstates,
                actual_nrows: c.nrows(),
                actual_ncols: c.ncols(),
            },
        ));
    }
    if d.nrows() != c.nrows() || d.ncols() != b.ncols() {
        return Err(BalancedError::StateSpace(
            super::state_space::StateSpaceError::DimensionMismatch {
                which: "d",
                expected_nrows: c.nrows(),
                expected_ncols: b.ncols(),
                actual_nrows: d.nrows(),
                actual_ncols: d.ncols(),
            },
        ));
    }
    Ok(())
}

fn dense_mul<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
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

fn sparse_matmul_dense<I, T>(lhs: SparseColMatRef<'_, I, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    I: Index,
    T: CompensatedField,
    T::Real: Float + Copy,
{
    let lhs = lhs.canonical();
    let nrows = lhs.nrows().unbound();
    let ncols = lhs.ncols().unbound();
    assert_eq!(rhs.nrows(), ncols);

    let mut out = Mat::<T>::zeros(nrows, rhs.ncols());
    let col_ptr = lhs.col_ptr();
    let row_idx = lhs.row_idx();
    let values = lhs.val();

    for out_col in 0..rhs.ncols() {
        let mut acc = vec![CompensatedSum::<T>::default(); nrows];
        for lhs_col in 0..ncols {
            let rhs_value = rhs[(lhs_col, out_col)];
            let start = col_ptr[lhs_col].zx();
            let end = col_ptr[lhs_col + 1].zx();
            for idx in start..end {
                // Accumulate each output row with compensated sums so the
                // projection step is consistent with the full-accuracy policy
                // used throughout the control module.
                acc[row_idx[idx].zx()].add(values[idx] * rhs_value);
            }
        }
        for row in 0..nrows {
            out[(row, out_col)] = acc[row].finish();
        }
    }

    out
}

fn dense_adjoint<T>(matrix: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    Mat::from_fn(matrix.ncols(), matrix.nrows(), |row, col| {
        matrix[(col, row)].conj()
    })
}

fn clone_mat<T: Clone>(matrix: MatRef<'_, T>) -> Mat<T> {
    Mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        matrix[(row, col)].clone()
    })
}

fn tail_sum<R>(values: faer::ColRef<'_, R>, from: usize) -> R
where
    R: Float + Copy,
{
    let mut sum = <R as Zero>::zero();
    for i in from..values.nrows() {
        sum = sum + values[i];
    }
    sum
}

#[cfg(test)]
mod test {
    use super::{
        BalancedParams, InternalsLevel, balanced_truncation_continuous_dense,
        balanced_truncation_continuous_low_rank, balanced_truncation_discrete_dense,
        balanced_truncation_discrete_low_rank,
    };
    use crate::control::{ContinuousStateSpace, LyapunovParams, ShiftStrategy};
    use faer::sparse::{SparseColMat, Triplet};
    use faer::{Mat, c64};

    fn assert_close(lhs: &Mat<f64>, rhs: &Mat<f64>, tol: f64) {
        assert_eq!(lhs.nrows(), rhs.nrows());
        assert_eq!(lhs.ncols(), rhs.ncols());
        for col in 0..lhs.ncols() {
            for row in 0..lhs.nrows() {
                let err = (lhs[(row, col)] - rhs[(row, col)]).abs();
                assert!(
                    err <= tol,
                    "entry ({row}, {col}) mismatch: err={err}, tol={tol}"
                );
            }
        }
    }

    #[test]
    fn dense_continuous_balanced_truncation_returns_requested_internals() {
        let a = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => -1.0,
            (1, 1) => -2.0,
            _ => 0.0,
        });
        let b = Mat::<f64>::identity(2, 2);
        let c = Mat::<f64>::identity(2, 2);
        let sys = ContinuousStateSpace::with_zero_feedthrough(a, b, c).unwrap();

        let result = balanced_truncation_continuous_dense(
            &sys,
            &BalancedParams::new().with_internals(InternalsLevel::Full),
        )
        .unwrap();

        assert_eq!(result.reduced.nstates(), 2);
        assert_eq!(result.left_projection.nrows(), 2);
        assert_eq!(result.right_projection.nrows(), 2);
        let internals = result.internals.unwrap();
        assert!(internals.dense_controllability_gramian.is_some());
        assert!(internals.dense_observability_gramian.is_some());
        assert!(internals.dense_controllability_sqrt.is_some());
        assert!(internals.dense_observability_sqrt.is_some());
    }

    #[test]
    fn dense_continuous_balanced_truncation_supports_order_zero() {
        let a = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => -1.0,
            (1, 1) => -3.0,
            _ => 0.0,
        });
        let b = Mat::<f64>::identity(2, 1);
        let c = Mat::<f64>::identity(1, 2);
        let d = Mat::<f64>::from_fn(1, 1, |_, _| 2.0);
        let sys = ContinuousStateSpace::new(a, b, c, d.clone()).unwrap();

        let result =
            balanced_truncation_continuous_dense(&sys, &BalancedParams::new().with_order(0))
                .unwrap();

        assert_eq!(result.reduced.nstates(), 0);
        assert_eq!(result.reduced.b().nrows(), 0);
        assert_eq!(result.reduced.c().ncols(), 0);
        assert_close(&super::clone_mat(result.reduced.d()), &d, 1.0e-12);
    }

    #[test]
    fn dense_discrete_balanced_truncation_runs() {
        let a = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => 0.25,
            (1, 1) => -0.4,
            _ => 0.0,
        });
        let b = Mat::<f64>::identity(2, 2);
        let c = Mat::<f64>::identity(2, 2);
        let sys = crate::control::DiscreteStateSpace::with_zero_feedthrough(a, b, c, 0.1).unwrap();

        let result = balanced_truncation_discrete_dense(&sys, &BalancedParams::new()).unwrap();
        assert_eq!(result.reduced.sample_time(), 0.1);
        assert_eq!(result.hankel_singular_values.nrows(), 2);
    }

    #[test]
    fn sparse_continuous_low_rank_balanced_truncation_matches_dense_reference() {
        let a_sparse = SparseColMat::<usize, f64>::try_new_from_triplets(
            3,
            3,
            &[
                Triplet::new(0, 0, -2.0),
                Triplet::new(1, 0, 0.2),
                Triplet::new(0, 1, 0.5),
                Triplet::new(1, 1, -1.5),
                Triplet::new(2, 1, -0.4),
                Triplet::new(1, 2, 0.75),
                Triplet::new(2, 2, -0.8),
            ],
        )
        .unwrap();
        let b = Mat::from_fn(3, 1, |row, _| match row {
            0 => 1.0,
            1 => -0.25,
            _ => 0.5,
        });
        let c = Mat::from_fn(1, 3, |_, col| match col {
            0 => 1.0,
            1 => -0.5,
            _ => 0.25,
        });
        let d = Mat::<f64>::zeros(1, 1);
        let shifts = ShiftStrategy::user_provided(vec![-0.5, -1.0, -2.0, -4.0]);
        let gramian_params = LyapunovParams {
            tol: 1.0e-10,
            max_iters: 24,
        };

        let sparse = balanced_truncation_continuous_low_rank(
            a_sparse.as_ref(),
            b.as_ref(),
            c.as_ref(),
            d.as_ref(),
            &shifts,
            gramian_params,
            &BalancedParams::new().with_order(1),
        )
        .unwrap();

        let a_dense = a_sparse.as_ref().to_dense();
        let dense_sys =
            ContinuousStateSpace::new(a_dense, b.clone(), c.clone(), d.clone()).unwrap();
        let dense =
            balanced_truncation_continuous_dense(&dense_sys, &BalancedParams::new().with_order(1))
                .unwrap();

        assert_close(
            &super::clone_mat(sparse.reduced.a()),
            &super::clone_mat(dense.reduced.a()),
            1.0e-6,
        );
        assert!(
            (sparse.hankel_singular_values[0] - dense.hankel_singular_values[0]).abs() <= 1.0e-8
        );
        assert_close(
            &super::dense_mul(sparse.reduced.c(), sparse.reduced.b()),
            &super::dense_mul(dense.reduced.c(), dense.reduced.b()),
            1.0e-6,
        );
        assert!((sparse.error_bound.unwrap() - dense.error_bound.unwrap()).abs() <= 1.0e-8);
    }

    #[test]
    fn sparse_discrete_low_rank_balanced_truncation_matches_dense_reference() {
        let a_sparse = SparseColMat::<usize, f64>::try_new_from_triplets(
            3,
            3,
            &[
                Triplet::new(0, 0, 0.25),
                Triplet::new(1, 0, 0.05),
                Triplet::new(0, 1, -0.1),
                Triplet::new(1, 1, -0.4),
                Triplet::new(2, 1, 0.03),
                Triplet::new(1, 2, 0.08),
                Triplet::new(2, 2, 0.15),
            ],
        )
        .unwrap();
        let b = Mat::from_fn(3, 1, |row, _| match row {
            0 => 1.0,
            1 => -0.25,
            _ => 0.5,
        });
        let c = Mat::from_fn(1, 3, |_, col| match col {
            0 => 1.0,
            1 => -0.5,
            _ => 0.25,
        });
        let d = Mat::<f64>::zeros(1, 1);
        let shifts = ShiftStrategy::user_provided(vec![-0.25, -0.5, -1.0, -2.0]);
        let gramian_params = LyapunovParams {
            tol: 1.0e-10,
            max_iters: 24,
        };

        let sparse = balanced_truncation_discrete_low_rank(
            a_sparse.as_ref(),
            b.as_ref(),
            c.as_ref(),
            d.as_ref(),
            0.1,
            &shifts,
            gramian_params,
            &BalancedParams::new().with_order(1),
        )
        .unwrap();

        let a_dense = a_sparse.as_ref().to_dense();
        let dense_sys =
            crate::control::DiscreteStateSpace::new(a_dense, b.clone(), c.clone(), d.clone(), 0.1)
                .unwrap();
        let dense =
            balanced_truncation_discrete_dense(&dense_sys, &BalancedParams::new().with_order(1))
                .unwrap();

        assert_close(
            &super::clone_mat(sparse.reduced.a()),
            &super::clone_mat(dense.reduced.a()),
            1.0e-6,
        );
        assert!(
            (sparse.hankel_singular_values[0] - dense.hankel_singular_values[0]).abs() <= 1.0e-8
        );
        assert_close(
            &super::dense_mul(sparse.reduced.c(), sparse.reduced.b()),
            &super::dense_mul(dense.reduced.c(), dense.reduced.b()),
            1.0e-6,
        );
        assert!((sparse.error_bound.unwrap() - dense.error_bound.unwrap()).abs() <= 1.0e-8);
    }

    #[test]
    fn dense_balanced_truncation_handles_complex_system() {
        let a = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => c64::new(-1.0, 0.2),
            (1, 1) => c64::new(-2.0, -0.1),
            _ => c64::new(0.0, 0.0),
        });
        let b = Mat::from_fn(2, 1, |row, _| match row {
            0 => c64::new(1.0, -0.5),
            _ => c64::new(-0.25, 0.75),
        });
        let c = Mat::from_fn(1, 2, |_, col| match col {
            0 => c64::new(1.0, 0.25),
            _ => c64::new(-0.5, 0.5),
        });
        let sys = ContinuousStateSpace::with_zero_feedthrough(a, b, c).unwrap();

        let result =
            balanced_truncation_continuous_dense(&sys, &BalancedParams::new().with_order(1))
                .unwrap();
        assert_eq!(result.reduced.nstates(), 1);
    }
}
