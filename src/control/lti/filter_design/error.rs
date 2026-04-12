use super::super::error::LtiError;
use core::fmt;

/// Errors produced by analog and digital IIR filter design.
#[derive(Debug)]
pub enum FilterDesignError {
    /// Filter order must be at least one.
    InvalidOrder,
    /// The supplied ripple parameter is invalid.
    InvalidRipple,
    /// One cutoff frequency is invalid.
    InvalidCutoff { which: &'static str },
    /// Band edges are invalid.
    InvalidBandEdges,
    /// Sample rate must be positive and finite.
    InvalidSampleRate,
    /// The requested digital Bessel design is intentionally unsupported.
    UnsupportedDigitalBessel,
    /// A lower-level LTI conversion or root-extraction helper failed.
    Lti(LtiError),
}

impl fmt::Display for FilterDesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for FilterDesignError {}

impl From<LtiError> for FilterDesignError {
    fn from(value: LtiError) -> Self {
        Self::Lti(value)
    }
}
