#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
pub mod control;
#[cfg(feature = "std")]
pub mod decomp;
#[cfg(feature = "std")]
pub mod sparse;
pub mod sum;
