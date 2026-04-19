//! Dynamic-size `no_std + alloc` embedded runtimes.

pub mod estimation;
mod matrix;

pub use matrix::{Matrix, Vector};
