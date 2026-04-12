//! Linear time-invariant system analysis and alternate SISO representations.
//!
//! The broader control module is built around numerical primitives such as
//! Lyapunov/Stein solvers and balanced truncation. This `lti` layer sits above
//! that foundation and groups the user-facing model-analysis and
//! transfer-function-style APIs that are specifically about linear
//! time-invariant systems.
//!
//! The current implementation is still dense-first overall, but selected
//! sparse state-space workflows are now available:
//!
//! - dense state-space analysis lives in `analysis`
//! - dense sampled responses live in `response`
//! - fixed-timestep digital filtering helpers live in `sim`
//! - real-coefficient SISO alternate representations live in
//!   `transfer_function`, `zpk`, and `sos`
//! - sparse CSC-backed state-space models support transfer evaluation,
//!   frequency response, and discrete-time simulation through the same
//!   conceptual API
//!
//! Broader sparse/operator-backed analysis, especially continuous-time
//! matrix-function actions and large-scale stability diagnostics, still belongs
//! to later phases once the required Krylov and matrix-function machinery is in
//! place.

mod analysis;
mod error;
mod filter_design;
mod fir;
mod plot_data;
mod response;
mod sim;
mod sos;
mod transfer_function;
mod util;
mod zpk;

pub use error::LtiError;
pub use filter_design::{
    AnalogFilterFamily, AnalogFilterSpec, DigitalFilterFamily, DigitalFilterSpec,
    FilterDesignError, FilterShape, design_analog_filter_sos, design_analog_filter_tf,
    design_analog_filter_zpk, design_digital_filter_sos, design_digital_filter_tf,
    design_digital_filter_zpk,
};
pub use fir::{Fir, FirFilterState, SavGolSpec, design_savgol};
pub use plot_data::{BodeData, PoleZeroData};
pub use response::{
    ContinuousImpulseResponse, ContinuousSimulation, DiscreteSimulation, SampledResponse,
};
pub use sim::{
    FiltFiltPadLen, FiltFiltPadMode, FiltFiltParams, FilteredSignal, SosFilterState,
    StatefulFilteredSignal,
};
pub use sos::{ContinuousSos, DiscreteSos, SecondOrderSection, Sos};
pub use transfer_function::{
    ContinuousTransferFunction, DiscreteTransferFunction, TransferFunction,
};
pub use zpk::{ContinuousZpk, DiscreteZpk, Zpk};

pub use super::state_space::{
    ContinuousStateSpace, ContinuousTime, DiscreteStateSpace, DiscreteTime,
    SparseContinuousStateSpace, SparseDiscreteStateSpace, SparseStateSpace, StateSpace,
    StateSpaceError,
};
