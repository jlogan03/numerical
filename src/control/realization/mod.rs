//! Reusable realization-building utilities shared by identification methods.
//!
//! This layer sits between:
//!
//! - known or estimated discrete-time Markov parameters
//! - higher-level realization algorithms such as ERA and OKID
//!
//! The first implementation keeps the core deliberately small:
//!
//! - [`MarkovSequence`] stores the block sequence `H_0, H_1, ...`
//! - [`BlockHankel`] assembles dense block-Hankel matrices from that sequence
//! - [`ShiftedBlockHankelPair`] builds the standard ERA pair `(H0, H1)`
//!
//! The module is discrete-time focused. Continuous-time impulse responses are
//! functions of time rather than finite recurrence blocks, so a continuous
//! Markov abstraction would need an explicit sampling policy that belongs in a
//! later layer.

mod error;
mod hankel;
mod markov;

pub use error::RealizationError;
pub use hankel::{
    BlockHankel, ShiftedBlockHankelPair, hankel_matrix_shape, max_square_era_block_dim,
    recommended_square_era_block_dim, required_markov_len,
};
pub use markov::MarkovSequence;
