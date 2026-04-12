//! Algebraic matrix-equation solvers used by higher-level control workflows.
//!
//! This module groups the low-level dense and sparse solver kernels that sit
//! below synthesis, estimation, and model-reduction routines:
//!
//! - [`lyapunov`] for continuous-time Lyapunov equations and Gramians
//! - [`stein`] for discrete-time Stein equations and Gramians
//! - [`riccati`] for continuous/discrete algebraic Riccati equations

pub mod lyapunov;
pub mod riccati;
pub mod stein;

pub use lyapunov::{
    DenseLyapunovSolve, LowRankFactor, LowRankLyapunovSolve, LyapunovError, LyapunovParams,
    ShiftStrategy, controllability_gramian_dense, controllability_gramian_low_rank,
    observability_gramian_dense, observability_gramian_low_rank, solve_continuous_lyapunov_dense,
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
