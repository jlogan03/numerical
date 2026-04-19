#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
pub mod control;
#[cfg(feature = "std")]
pub mod decomp;
/// Embedded-friendly runtime control and estimation kernels.
pub mod embedded;
#[cfg(feature = "std")]
/// Internal scalar arithmetic helpers shared across the crate.
pub(crate) mod scalar;
#[cfg(feature = "std")]
/// Sparse direct solvers, iterative solvers, and preconditioners.
pub mod sparse;
/// Accurate floating-point summation utilities.
pub mod sum;
