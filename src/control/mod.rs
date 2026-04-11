//! Control-systems algorithms built on the crate's dense and sparse numerics.
//!
//! The module is organized in layers:
//!
//! - `state_space` and `lti` provide model representations and analysis
//! - `lyapunov`, `stein`, and `hsvd` provide reusable matrix-equation and
//!   balancing-core numerics
//! - `balanced` builds reduced-order models on top of those lower-level pieces
//!
//! The current public surface is intentionally dense-first for higher-level
//! model manipulation, while sparse support is concentrated in the reusable
//! solver layers underneath.

pub mod balanced;
pub mod hsvd;
pub mod lti;
pub mod lyapunov;
pub mod state_space;
pub mod stein;

pub use balanced::{
    BalancedError, BalancedInternals, BalancedParams, BalancedTruncationResult, InternalsLevel,
    balanced_truncation_continuous_dense, balanced_truncation_continuous_low_rank,
    balanced_truncation_discrete_dense, balanced_truncation_discrete_low_rank,
};
pub use hsvd::{
    HsvdError, HsvdInternals, HsvdInternalsLevel, HsvdParams, HsvdResult, hsvd_from_dense_gramians,
    hsvd_from_factors,
};
pub use lyapunov::{
    DenseLyapunovSolve, LowRankFactor, LowRankLyapunovSolve, LyapunovError, LyapunovParams,
    ShiftStrategy, controllability_gramian_dense, controllability_gramian_low_rank,
    observability_gramian_dense, observability_gramian_low_rank, solve_continuous_lyapunov_dense,
};
pub use state_space::{
    ContinuousStateSpace, ContinuousTime, ContinuousizationMethod, DiscreteStateSpace,
    DiscreteTime, DiscretizationMethod, StateSpace, StateSpaceError,
};
pub use stein::{
    DenseSteinSolve, SteinError, controllability_gramian_discrete_dense,
    controllability_gramian_discrete_low_rank, observability_gramian_discrete_dense,
    observability_gramian_discrete_low_rank, solve_discrete_stein_dense,
};
