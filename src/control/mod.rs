//! Control-systems algorithms built on the crate's dense and sparse numerics.
//!
//! The module is organized in layers:
//!
//! - `state_space` and `lti` provide model representations and analysis
//! - `lyapunov`, `stein`, and `hsvd` provide reusable matrix-equation and
//!   balancing-core numerics
//! - `balanced` builds reduced-order models on top of those lower-level pieces
//!
//! The current public surface is still dense-first for higher-level model
//! manipulation, but it now includes selected sparse state-space workflows on
//! top of the lower-level sparse solver layers. Dense conversion, dense
//! full-spectrum analysis, and dense direct model manipulation remain the most
//! complete paths.

pub mod balanced;
pub mod estimator;
pub mod hsvd;
pub mod identification;
pub mod lqr;
pub mod lti;
pub mod lyapunov;
pub mod realization;
pub mod riccati;
pub mod state_space;
pub mod stein;

pub use balanced::{
    BalancedError, BalancedInternals, BalancedParams, BalancedTruncationResult, InternalsLevel,
    balanced_truncation_continuous_dense, balanced_truncation_continuous_low_rank,
    balanced_truncation_discrete_dense, balanced_truncation_discrete_low_rank,
};
pub use estimator::{
    DiscreteKalmanFilter, EstimatorError, KalmanPrediction, KalmanUpdate, LqeSolve, dlqe_dense,
    lqe_dense,
};
pub use hsvd::{
    HsvdError, HsvdInternals, HsvdInternalsLevel, HsvdParams, HsvdResult, hsvd_from_dense_gramians,
    hsvd_from_factors,
};
pub use identification::{
    EraError, EraInternals, EraInternalsLevel, EraParams, EraResult, OkidError, OkidParams,
    OkidResult, era_from_markov, era_from_shifted_hankel, okid,
};
pub use lqr::{LqrError, LqrSolve, dlqr_dense, lqr_dense};
pub use lyapunov::{
    DenseLyapunovSolve, LowRankFactor, LowRankLyapunovSolve, LyapunovError, LyapunovParams,
    ShiftStrategy, controllability_gramian_dense, controllability_gramian_low_rank,
    observability_gramian_dense, observability_gramian_low_rank, solve_continuous_lyapunov_dense,
};
pub use realization::{
    BlockHankel, MarkovSequence, RealizationError, ShiftedBlockHankelPair, hankel_matrix_shape,
    max_square_era_block_dim, recommended_square_era_block_dim, required_markov_len,
};
pub use riccati::{
    RiccatiError, RiccatiSolve, care_gain_from_solution, dare_gain_from_solution, solve_care_dense,
    solve_dare_dense,
};
pub use state_space::{
    ContinuousStateSpace, ContinuousTime, ContinuousizationMethod, DiscreteStateSpace,
    DiscreteTime, DiscretizationMethod, ObserverControllerComposition, SparseContinuousStateSpace,
    SparseDiscreteStateSpace, SparseStateSpace, StateSpace, StateSpaceError,
};
pub use stein::{
    DenseSteinSolve, SteinError, controllability_gramian_discrete_dense,
    controllability_gramian_discrete_low_rank, observability_gramian_discrete_dense,
    observability_gramian_discrete_low_rank, solve_discrete_stein_dense,
};
