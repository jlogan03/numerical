use core::fmt;

/// Errors produced by Markov-sequence and block-Hankel utilities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RealizationError {
    /// A Markov sequence built from owned blocks needs at least one block so
    /// its input/output shape can be inferred.
    EmptySequence,
    /// One Markov block had a different `output x input` shape from the rest
    /// of the sequence.
    InconsistentBlockShape {
        index: usize,
        expected_nrows: usize,
        expected_ncols: usize,
        actual_nrows: usize,
        actual_ncols: usize,
    },
    /// A block-Hankel builder was given zero row or column block count.
    ZeroBlockCount { which: &'static str },
    /// The Markov sequence is too short for the requested Hankel layout.
    SequenceTooShort {
        available: usize,
        required: usize,
        start_index: usize,
        row_blocks: usize,
        col_blocks: usize,
    },
}

impl fmt::Display for RealizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for RealizationError {}
