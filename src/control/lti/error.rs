use core::fmt;

use crate::control::state_space::StateSpaceError;
use crate::sparse::lu::SparseLuError;
use faer::linalg::solvers::{EvdError, SvdError};
use faer::sparse::{CreationError, FaerError};

/// Errors produced by LTI analysis and representation-conversion routines.
#[derive(Debug)]
pub enum LtiError {
    /// Dense eigendecomposition failed while extracting poles or roots.
    Eigen(EvdError),
    /// Dense SVD failed while making a numerical rank decision.
    Svd(SvdError),
    /// A discrete-time representation was given a nonpositive or nonfinite
    /// sample interval.
    InvalidSampleTime,
    /// A response grid contained an invalid point.
    InvalidSamplePoint { which: &'static str },
    /// A sampling grid had inconsistent structure, such as mismatched lengths
    /// or non-monotone time points.
    InvalidSampleGrid { which: &'static str },
    /// An analysis or simulation input had incompatible dimensions.
    DimensionMismatch {
        which: &'static str,
        expected_nrows: usize,
        expected_ncols: usize,
        actual_nrows: usize,
        actual_ncols: usize,
    },
    /// A polynomial representation was missing required coefficients.
    EmptyPolynomial { which: &'static str },
    /// The leading coefficient of a polynomial must be nonzero.
    ZeroLeadingCoefficient { which: &'static str },
    /// A conversion expected a single-input single-output state-space system.
    NonSisoStateSpace { ninputs: usize, noutputs: usize },
    /// A state-space realization exists only for proper transfer functions.
    ImproperTransferFunction {
        numerator_degree: usize,
        denominator_degree: usize,
    },
    /// A conversion from complex roots back to real coefficients requires the
    /// root set to be closed under complex conjugation.
    NotConjugateClosed { which: &'static str },
    /// A response or conversion formula produced non-finite values.
    NonFiniteResult { which: &'static str },
    /// A second-order-section cascade must contain at least one section.
    EmptySos,
    /// A dense state-space helper used underneath an LTI analysis routine
    /// failed.
    StateSpace(StateSpaceError),
    /// Sparse CSC construction failed while building an analysis operator.
    SparseBuild(CreationError),
    /// Sparse format conversion failed.
    SparseFormat(FaerError),
    /// Sparse LU analysis, factorization, or solve failed.
    SparseLu(SparseLuError),
}

impl fmt::Display for LtiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for LtiError {}

impl From<EvdError> for LtiError {
    fn from(value: EvdError) -> Self {
        Self::Eigen(value)
    }
}

impl From<SvdError> for LtiError {
    fn from(value: SvdError) -> Self {
        Self::Svd(value)
    }
}

impl From<StateSpaceError> for LtiError {
    fn from(value: StateSpaceError) -> Self {
        Self::StateSpace(value)
    }
}

impl From<CreationError> for LtiError {
    fn from(value: CreationError) -> Self {
        Self::SparseBuild(value)
    }
}

impl From<FaerError> for LtiError {
    fn from(value: FaerError) -> Self {
        Self::SparseFormat(value)
    }
}

impl From<SparseLuError> for LtiError {
    fn from(value: SparseLuError) -> Self {
        Self::SparseLu(value)
    }
}
