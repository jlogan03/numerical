//! Small scalar helpers shared across the embedded runtime modules.

use super::EmbeddedError;
use num_traits::Float;

/// Returns `value` if it is finite, otherwise reports a named runtime error.
pub(crate) fn ensure_finite<T>(value: T, which: &'static str) -> Result<T, EmbeddedError>
where
    T: Float + Copy,
{
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EmbeddedError::NonFiniteValue { which })
    }
}

/// Clamps one scalar into the inclusive range `[low, high]`.
pub(crate) fn clamp_value<T>(value: T, low: T, high: T) -> T
where
    T: Float + Copy,
{
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}
