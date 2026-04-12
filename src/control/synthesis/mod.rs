//! Controller synthesis and practical controller runtime layers.
//!
//! This module groups:
//!
//! - [`lqr`] and [`lqg`] for optimal controller design
//! - [`pole_placement`] for classical dense pole assignment
//! - [`pid`] and [`pid_design`] for PID runtime control and tuning workflows

pub mod lqg;
pub mod lqr;
pub mod pid;
pub mod pid_design;
pub mod pole_placement;

pub use lqg::{LqgError, LqgSolve, dlqg_dense, lqg_dense};
pub use lqr::{LqrError, LqrSolve, dlqr_dense, lqr_dense};
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
pub use pole_placement::{
    PolePlacementError, PolePlacementSolve, dplace_observer_poles_dense, dplace_poles_dense,
    place_observer_poles_dense, place_poles_dense,
};
