//! Linear time-invariant system analysis and alternate SISO representations.
//!
//! The broader control module is built around numerical primitives such as
//! Lyapunov/Stein solvers and balanced truncation. This `lti` layer sits above
//! that foundation and groups the user-facing model-analysis and
//! transfer-function-style APIs that are specifically about linear
//! time-invariant systems.

mod analysis;
mod error;
mod sos;
mod transfer_function;
mod util;
mod zpk;

pub use error::LtiError;
pub use sos::{ContinuousSos, DiscreteSos, SecondOrderSection, Sos};
pub use transfer_function::{
    ContinuousTransferFunction, DiscreteTransferFunction, TransferFunction,
};
pub use zpk::{ContinuousZpk, DiscreteZpk, Zpk};

pub use super::state_space::{
    ContinuousStateSpace, ContinuousTime, DiscreteStateSpace, DiscreteTime, StateSpace,
    StateSpaceError,
};
