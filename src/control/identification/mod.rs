//! System-identification algorithms built on the realization utilities.
//!
//! The identification layer sits above [`crate::control::realization`] and
//! turns either:
//!
//! - structured response data, such as Markov parameters, or
//! - sampled input/output data
//!
//! into discrete-time state-space models.

mod era;
mod okid;

pub use era::{
    EraError, EraInternals, EraInternalsLevel, EraParams, EraResult, era_from_markov,
    era_from_shifted_hankel,
};
pub use okid::{OkidError, OkidParams, OkidResult, okid};
