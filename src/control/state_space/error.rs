use core::fmt;

/// Errors produced by state-space construction and conversion routines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StateSpaceError {
    /// The state matrix `A` must be square.
    NonSquareA { nrows: usize, ncols: usize },
    /// One of the `A/B/C/D` blocks had incompatible dimensions.
    DimensionMismatch {
        which: &'static str,
        expected_nrows: usize,
        expected_ncols: usize,
        actual_nrows: usize,
        actual_ncols: usize,
    },
    /// A discrete-time sample interval must be positive and finite.
    InvalidSampleTime,
    /// A bilinear prewarp frequency must be finite and nonnegative.
    InvalidPrewarpFrequency,
    /// A conversion required solving against a singular or numerically
    /// unsuitable matrix.
    SingularConversion { which: &'static str },
    /// A dense matrix function or solve produced non-finite output.
    NonFiniteResult { which: &'static str },
    /// The requested conversion is recognized but not implemented in this
    /// first state-space pass.
    UnsupportedConversion(&'static str),
}

impl fmt::Display for StateSpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for StateSpaceError {}
