//! Model-reduction building blocks and balanced-truncation front ends.
//!
//! This module groups the reduction-side layers:
//!
//! - [`hsvd`] for the reusable balancing core
//! - [`balanced`] for dense and low-rank balanced truncation

pub mod balanced;
pub mod hsvd;

pub use balanced::{
    BalancedError, BalancedInternals, BalancedParams, BalancedTruncationResult, InternalsLevel,
    balanced_truncation_continuous_dense, balanced_truncation_continuous_low_rank,
    balanced_truncation_discrete_dense, balanced_truncation_discrete_low_rank,
};
pub use hsvd::{
    HsvdError, HsvdInternals, HsvdInternalsLevel, HsvdParams, HsvdResult, hsvd_from_dense_gramians,
    hsvd_from_factors,
};
