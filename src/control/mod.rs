//! Control-systems algorithms built on the crate's dense and sparse numerics.
//!
//! The module is organized in layers:
//!
//! - `state_space` and `lti` provide model representations and analysis
//! - `lyapunov`, `stein`, and `hsvd` provide reusable matrix-equation and
//!   balancing-core numerics
//! - `balanced` builds reduced-order models on top of those lower-level pieces
//! - `riccati`, `lqr`, `estimator`, `lqg`, and `pid` provide controller and
//!   estimator design plus practical runtime controller wrappers
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
pub mod lqg;
pub mod lqr;
pub mod lti;
pub mod lyapunov;
pub mod nonlinear_estimator;
pub mod pid;
pub mod pid_design;
pub mod realization;
pub mod riccati;
pub mod stein;

pub use balanced::{
    BalancedError, BalancedInternals, BalancedParams, BalancedTruncationResult, InternalsLevel,
    balanced_truncation_continuous_dense, balanced_truncation_continuous_low_rank,
    balanced_truncation_discrete_dense, balanced_truncation_discrete_low_rank,
};
pub use estimator::{
    ContinuousObserver, ContinuousObserverDerivative, CovarianceUpdate, DiscreteKalmanFilter,
    EstimatorError, KalmanPrediction, KalmanUpdate, LqeSolve, SteadyStateKalmanFilter,
    SteadyStateKalmanPrediction, SteadyStateKalmanUpdate, dlqe_dense, lqe_dense,
};
pub use hsvd::{
    HsvdError, HsvdInternals, HsvdInternalsLevel, HsvdParams, HsvdResult, hsvd_from_dense_gramians,
    hsvd_from_factors,
};
pub use identification::{
    EraError, EraInternals, EraInternalsLevel, EraParams, EraResult, OkidError, OkidParams,
    OkidResult, era_from_markov, era_from_shifted_hankel, okid,
};
pub use lqg::{LqgError, LqgSolve, dlqg_dense, lqg_dense};
pub use lqr::{LqrError, LqrSolve, dlqr_dense, lqr_dense};
pub use lti::state_space::{
    ContinuousStateSpace, ContinuousTime, ContinuousizationMethod, DiscreteStateSpace,
    DiscreteTime, DiscretizationMethod, ObserverControllerComposition, SparseContinuousStateSpace,
    SparseDiscreteStateSpace, SparseStateSpace, StateSpace, StateSpaceError,
};
pub use lyapunov::{
    DenseLyapunovSolve, LowRankFactor, LowRankLyapunovSolve, LyapunovError, LyapunovParams,
    ShiftStrategy, controllability_gramian_dense, controllability_gramian_low_rank,
    observability_gramian_dense, observability_gramian_low_rank, solve_continuous_lyapunov_dense,
};
pub use nonlinear_estimator::{
    DiscreteExtendedKalmanModel, DiscreteNonlinearModel, ExtendedKalmanFilter,
    NonlinearEstimatorError, NonlinearKalmanPrediction, NonlinearKalmanUpdate, SigmaPointProvider,
    SigmaPointSet, SigmaPointStrategy, UkfStage, UnscentedKalmanFilter, UnscentedParams,
};
pub use pid::{AntiWindup, Pid, PidError, PidOutput, PidState};
pub use pid_design::{
    FopdtModel, FrequencyPidDesign, FrequencyPidParams, OkidEraPidDesign, PidControllerKind,
    PidDesignError, ProcessFitResult, ProcessPidDesign, SampledIoData, SimcPidParams, SopdtModel,
    StepFitPidDesign, StepOptimizationPidDesign, StepOptimizationPidParams, StepResponseData,
    design_pid_from_continuous_state_space_frequency, design_pid_from_continuous_tf_frequency,
    design_pid_from_discrete_state_space_frequency,
    design_pid_from_discrete_state_space_step_optimization, design_pid_from_discrete_tf_frequency,
    design_pid_from_discrete_tf_step_optimization, design_pid_from_fopdt, design_pid_from_okid_era,
    design_pid_from_sopdt, design_pid_from_step_response_fopdt,
    design_pid_from_step_response_sopdt, fit_fopdt_from_step_response,
    fit_sopdt_from_step_response,
};
pub use realization::{
    BlockHankel, MarkovSequence, RealizationError, ShiftedBlockHankelPair, hankel_matrix_shape,
    max_square_era_block_dim, recommended_square_era_block_dim, required_markov_len,
};
pub use riccati::{
    RiccatiError, RiccatiSolve, care_gain_from_solution, dare_gain_from_solution, solve_care_dense,
    solve_dare_dense,
};
pub use stein::{
    DenseSteinSolve, SteinError, controllability_gramian_discrete_dense,
    controllability_gramian_discrete_low_rank, observability_gramian_discrete_dense,
    observability_gramian_discrete_low_rank, solve_discrete_stein_dense,
};
