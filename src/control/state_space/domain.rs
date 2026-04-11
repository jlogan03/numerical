use core::fmt;

/// Marker type for continuous-time state-space systems.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContinuousTime;

/// Metadata carried by discrete-time state-space systems.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiscreteTime<R> {
    sample_time: R,
}

impl<R> DiscreteTime<R> {
    /// Creates discrete-time metadata with the given sample interval.
    #[must_use]
    pub fn new(sample_time: R) -> Self {
        Self { sample_time }
    }

    /// Sample interval used by the discrete-time model.
    #[must_use]
    pub fn sample_time(&self) -> R
    where
        R: Copy,
    {
        self.sample_time
    }
}

impl<R: fmt::Display> fmt::Display for DiscreteTime<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dt={}", self.sample_time)
    }
}
