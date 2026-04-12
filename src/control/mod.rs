//! Control-systems algorithms built on the crate's dense and sparse numerics.
//!
//! The module is organized in layers:
//!
//! - `lti` provides model representations, analysis, and filtering
//! - `matrix_equations` provides reusable Lyapunov, Stein, and Riccati solvers
//! - `reduction` provides HSVD and balanced truncation
//! - `estimation` provides linear and nonlinear state estimation
//! - `synthesis` provides controller design and practical runtime controllers
//!
//! The current public surface is still dense-first for higher-level model
//! manipulation, but it now includes selected sparse state-space workflows on
//! top of the lower-level sparse solver layers. Dense conversion, dense
//! full-spectrum analysis, and dense direct model manipulation remain the most
//! complete paths.

pub mod estimation;
pub mod identification;
pub mod lti;
pub mod matrix_equations;
pub mod realization;
pub mod reduction;
pub mod synthesis;

pub use estimation::{
    ContinuousObserver, ContinuousObserverDerivative, CovarianceUpdate,
    DiscreteExtendedKalmanModel, DiscreteKalmanFilter, DiscreteNonlinearModel, EstimatorError,
    ExtendedKalmanFilter, KalmanPrediction, KalmanUpdate, LqeSolve, NonlinearEstimatorError,
    NonlinearKalmanPrediction, NonlinearKalmanUpdate, SigmaPointProvider, SigmaPointSet,
    SigmaPointStrategy, SteadyStateKalmanFilter, SteadyStateKalmanPrediction,
    SteadyStateKalmanUpdate, UkfStage, UnscentedKalmanFilter, UnscentedParams, dlqe_dense,
    lqe_dense,
};
pub use identification::{
    EraError, EraInternals, EraInternalsLevel, EraParams, EraResult, OkidError, OkidParams,
    OkidResult, era_from_markov, era_from_shifted_hankel, okid,
};
pub use lti::state_space::{
    ContinuousStateSpace, ContinuousTime, ContinuousizationMethod, DiscreteStateSpace,
    DiscreteTime, DiscretizationMethod, ObserverControllerComposition, SparseContinuousStateSpace,
    SparseDiscreteStateSpace, SparseStateSpace, StateSpace, StateSpaceError,
};
pub use matrix_equations::{
    DenseLyapunovSolve, DenseSteinSolve, LowRankFactor, LowRankLyapunovSolve, LyapunovError,
    LyapunovParams, RiccatiError, RiccatiSolve, ShiftStrategy, SteinError, care_gain_from_solution,
    controllability_gramian_dense, controllability_gramian_discrete_dense,
    controllability_gramian_discrete_low_rank, controllability_gramian_low_rank,
    dare_gain_from_solution, observability_gramian_dense, observability_gramian_discrete_dense,
    observability_gramian_discrete_low_rank, observability_gramian_low_rank, solve_care_dense,
    solve_continuous_lyapunov_dense, solve_dare_dense, solve_discrete_stein_dense,
};
pub use realization::{
    BlockHankel, MarkovSequence, RealizationError, ShiftedBlockHankelPair, hankel_matrix_shape,
    max_square_era_block_dim, recommended_square_era_block_dim, required_markov_len,
};
pub use reduction::{
    BalancedError, BalancedInternals, BalancedParams, BalancedTruncationResult, HsvdError,
    HsvdInternals, HsvdInternalsLevel, HsvdParams, HsvdResult, InternalsLevel,
    balanced_truncation_continuous_dense, balanced_truncation_continuous_low_rank,
    balanced_truncation_discrete_dense, balanced_truncation_discrete_low_rank,
    hsvd_from_dense_gramians, hsvd_from_factors,
};
pub use synthesis::{
    AntiWindup, FopdtModel, FrequencyPidDesign, FrequencyPidParams, LqgError, LqgSolve, LqrError,
    LqrSolve, OkidEraPidDesign, Pid, PidControllerKind, PidDesignError, PidError, PidOutput,
    PidState, ProcessFitResult, ProcessPidDesign, SampledIoData, SimcPidParams, SopdtModel,
    StepFitPidDesign, StepOptimizationPidDesign, StepOptimizationPidParams, StepResponseData,
    design_pid_from_continuous_state_space_frequency, design_pid_from_continuous_tf_frequency,
    design_pid_from_discrete_state_space_frequency,
    design_pid_from_discrete_state_space_step_optimization, design_pid_from_discrete_tf_frequency,
    design_pid_from_discrete_tf_step_optimization, design_pid_from_fopdt, design_pid_from_okid_era,
    design_pid_from_sopdt, design_pid_from_step_response_fopdt,
    design_pid_from_step_response_sopdt, dlqg_dense, dlqr_dense, fit_fopdt_from_step_response,
    fit_sopdt_from_step_response, lqg_dense, lqr_dense,
};
