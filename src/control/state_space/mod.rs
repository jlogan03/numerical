//! Dense state-space model types and dense continuous/discrete conversions.
//!
//! The first state-space layer in this crate is intentionally dense-first.
//! Exact `c2d` conversions use dense matrix functions, and even bilinear
//! conversion generally produces dense explicit discrete-time state matrices.
//!
//! The design still keeps continuous and discrete systems distinct in the type
//! system so downstream algorithms can state their assumptions clearly.
//!
//! This module is the model layer that sits above the lower-level Gramian and
//! matrix-equation routines in [`super::lyapunov`]. It gives those routines a
//! structured `A/B/C/D` home and makes the time domain part of the type
//! instead of leaving it as an implicit convention at the call site.

mod convert;
mod domain;
mod error;

pub use convert::{ContinuousizationMethod, DiscretizationMethod};
pub use domain::{ContinuousTime, DiscreteTime};
pub use error::StateSpaceError;

use super::lyapunov::{
    DenseLyapunovSolve, LyapunovError, controllability_gramian_dense, observability_gramian_dense,
};
use super::stein::{
    DenseSteinSolve, SteinError, controllability_gramian_discrete_dense,
    observability_gramian_discrete_dense,
};
use crate::sparse::compensated::CompensatedField;
use faer::{Mat, MatRef};
use faer_traits::ComplexField;
use num_traits::{Float, Zero};

/// Dense linear time-invariant state-space system.
///
/// The same storage layout is used for both continuous and discrete systems.
/// The `Domain` type parameter carries the semantic difference:
///
/// - continuous systems interpret `A` as the infinitesimal generator in
///   `x' = A x + B u`
/// - discrete systems interpret `A` as the one-step state transition in
///   `x[k+1] = A x[k] + B u[k]`
#[derive(Clone, Debug, PartialEq)]
pub struct StateSpace<T, Domain> {
    pub(crate) a: Mat<T>,
    pub(crate) b: Mat<T>,
    pub(crate) c: Mat<T>,
    pub(crate) d: Mat<T>,
    pub(crate) domain: Domain,
}

/// Dense continuous-time state-space system.
pub type ContinuousStateSpace<T> = StateSpace<T, ContinuousTime>;

/// Dense discrete-time state-space system.
pub type DiscreteStateSpace<T> = StateSpace<T, DiscreteTime<<T as ComplexField>::Real>>;

impl<T, Domain> StateSpace<T, Domain> {
    /// Number of states.
    ///
    /// This is the size of the internal state vector `x`.
    #[must_use]
    pub fn nstates(&self) -> usize {
        self.a.nrows()
    }

    /// Number of inputs.
    ///
    /// This is the width of the input vector `u`.
    #[must_use]
    pub fn ninputs(&self) -> usize {
        self.b.ncols()
    }

    /// Number of outputs.
    ///
    /// This is the height of the output vector `y`.
    #[must_use]
    pub fn noutputs(&self) -> usize {
        self.c.nrows()
    }

    /// Returns whether the system is single-input single-output.
    #[must_use]
    pub fn is_siso(&self) -> bool {
        self.ninputs() == 1 && self.noutputs() == 1
    }

    /// State matrix `A`.
    #[must_use]
    pub fn a(&self) -> MatRef<'_, T> {
        self.a.as_ref()
    }

    /// Input matrix `B`.
    #[must_use]
    pub fn b(&self) -> MatRef<'_, T> {
        self.b.as_ref()
    }

    /// Output matrix `C`.
    #[must_use]
    pub fn c(&self) -> MatRef<'_, T> {
        self.c.as_ref()
    }

    /// Feedthrough matrix `D`.
    #[must_use]
    pub fn d(&self) -> MatRef<'_, T> {
        self.d.as_ref()
    }

    /// Domain metadata carried by the system.
    #[must_use]
    pub fn domain(&self) -> &Domain {
        &self.domain
    }

    /// Splits the state-space system back into its owned parts.
    ///
    /// This is mainly useful when a caller wants to reuse the validated model
    /// storage in a different representation without cloning the matrices.
    #[must_use]
    pub fn into_parts(self) -> (Mat<T>, Mat<T>, Mat<T>, Mat<T>, Domain) {
        (self.a, self.b, self.c, self.d, self.domain)
    }
}

impl<T> ContinuousStateSpace<T>
where
    T: ComplexField,
{
    /// Creates a continuous-time state-space system after validating the
    /// `A/B/C/D` block dimensions.
    ///
    /// The validated model represents
    ///
    /// `x' = A x + B u`
    ///
    /// `y  = C x + D u`
    pub fn new(a: Mat<T>, b: Mat<T>, c: Mat<T>, d: Mat<T>) -> Result<Self, StateSpaceError> {
        validate_blocks(
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols(),
            c.nrows(),
            c.ncols(),
            d.nrows(),
            d.ncols(),
        )?;
        Ok(Self {
            a,
            b,
            c,
            d,
            domain: ContinuousTime,
        })
    }

    /// Creates a continuous-time model with a zero feedthrough matrix.
    ///
    /// This is the common case in state-space analysis workflows where the
    /// direct input-to-output path is absent or intentionally omitted.
    pub fn with_zero_feedthrough(a: Mat<T>, b: Mat<T>, c: Mat<T>) -> Result<Self, StateSpaceError> {
        let d = Mat::zeros(c.nrows(), b.ncols());
        Self::new(a, b, c, d)
    }
}

impl<T> ContinuousStateSpace<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    /// Computes the dense continuous-time controllability Gramian of the model.
    ///
    /// Intuitively, this measures how strongly the input channels in `B` can
    /// drive the internal state through the stable continuous-time dynamics.
    pub fn controllability_gramian(&self) -> Result<DenseLyapunovSolve<T>, LyapunovError> {
        controllability_gramian_dense(self.a.as_ref(), self.b.as_ref())
    }

    /// Computes the dense continuous-time observability Gramian of the model.
    ///
    /// Intuitively, this measures how strongly the internal state is reflected
    /// in the outputs through `C`.
    pub fn observability_gramian(&self) -> Result<DenseLyapunovSolve<T>, LyapunovError> {
        observability_gramian_dense(self.a.as_ref(), self.c.as_ref())
    }

    /// Converts the continuous-time model into a discrete-time one using the
    /// requested method and sample interval.
    ///
    /// The method is explicit because different `c2d` conversions encode
    /// different intersample assumptions. Zero-order hold models piecewise
    /// constant inputs; bilinear/Tustin models the trapezoidal-rule map that is
    /// common in digital filter and controller design.
    pub fn discretize(
        &self,
        sample_time: T::Real,
        method: DiscretizationMethod<T::Real>,
    ) -> Result<DiscreteStateSpace<T>, StateSpaceError> {
        convert::discretize(self, sample_time, method)
    }
}

impl<T> DiscreteStateSpace<T>
where
    T: ComplexField,
    T::Real: Float + Copy,
{
    /// Creates a discrete-time state-space system after validating dimensions
    /// and sample interval.
    ///
    /// The validated model represents
    ///
    /// `x[k + 1] = A x[k] + B u[k]`
    ///
    /// `y[k]     = C x[k] + D u[k]`
    pub fn new(
        a: Mat<T>,
        b: Mat<T>,
        c: Mat<T>,
        d: Mat<T>,
        sample_time: T::Real,
    ) -> Result<Self, StateSpaceError> {
        validate_blocks(
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols(),
            c.nrows(),
            c.ncols(),
            d.nrows(),
            d.ncols(),
        )?;
        if !sample_time.is_finite() || sample_time <= T::Real::zero() {
            return Err(StateSpaceError::InvalidSampleTime);
        }
        Ok(Self {
            a,
            b,
            c,
            d,
            domain: DiscreteTime::new(sample_time),
        })
    }

    /// Creates a discrete-time model with zero feedthrough.
    pub fn with_zero_feedthrough(
        a: Mat<T>,
        b: Mat<T>,
        c: Mat<T>,
        sample_time: T::Real,
    ) -> Result<Self, StateSpaceError> {
        let d = Mat::zeros(c.nrows(), b.ncols());
        Self::new(a, b, c, d, sample_time)
    }

    /// Sample interval used by the discrete-time model.
    ///
    /// This is the spacing between state updates in the discrete-time
    /// interpretation of the system.
    #[must_use]
    pub fn sample_time(&self) -> T::Real {
        self.domain.sample_time()
    }
}

impl<T> DiscreteStateSpace<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    /// Computes the dense discrete-time controllability Gramian of the model.
    ///
    /// This measures how strongly the sampled input channels can drive the
    /// state through repeated applications of the one-step transition matrix.
    pub fn controllability_gramian(&self) -> Result<DenseSteinSolve<T>, SteinError> {
        controllability_gramian_discrete_dense(self.a.as_ref(), self.b.as_ref())
    }

    /// Computes the dense discrete-time observability Gramian of the model.
    ///
    /// This measures how strongly the internal state is visible at the outputs
    /// after repeated propagation through the sampled dynamics.
    pub fn observability_gramian(&self) -> Result<DenseSteinSolve<T>, SteinError> {
        observability_gramian_discrete_dense(self.a.as_ref(), self.c.as_ref())
    }

    /// Converts the discrete-time model back into a continuous-time one using
    /// the requested reconstruction assumption.
    ///
    /// The method is explicit because `d2c` is not unique: a sampled model
    /// only corresponds to a continuous-time model after choosing how the input
    /// behaves between samples.
    pub fn continuousize(
        &self,
        method: ContinuousizationMethod<T::Real>,
    ) -> Result<ContinuousStateSpace<T>, StateSpaceError> {
        convert::continuousize(self, method)
    }
}

fn validate_blocks(
    a_nrows: usize,
    a_ncols: usize,
    b_nrows: usize,
    b_ncols: usize,
    c_nrows: usize,
    c_ncols: usize,
    d_nrows: usize,
    d_ncols: usize,
) -> Result<(), StateSpaceError> {
    // Constructor-time validation keeps every downstream control routine from
    // having to re-check the same `A/B/C/D` compatibility rules.
    if a_nrows != a_ncols {
        return Err(StateSpaceError::NonSquareA {
            nrows: a_nrows,
            ncols: a_ncols,
        });
    }
    if b_nrows != a_nrows {
        return Err(StateSpaceError::DimensionMismatch {
            which: "b",
            expected_nrows: a_nrows,
            expected_ncols: b_ncols,
            actual_nrows: b_nrows,
            actual_ncols: b_ncols,
        });
    }
    if c_ncols != a_ncols {
        return Err(StateSpaceError::DimensionMismatch {
            which: "c",
            expected_nrows: c_nrows,
            expected_ncols: a_ncols,
            actual_nrows: c_nrows,
            actual_ncols: c_ncols,
        });
    }
    if d_nrows != c_nrows || d_ncols != b_ncols {
        return Err(StateSpaceError::DimensionMismatch {
            which: "d",
            expected_nrows: c_nrows,
            expected_ncols: b_ncols,
            actual_nrows: d_nrows,
            actual_ncols: d_ncols,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ContinuousStateSpace, ContinuousizationMethod, DiscreteStateSpace, DiscretizationMethod,
        StateSpaceError,
    };
    use faer::{Mat, c64};
    use faer_traits::ext::ComplexFieldExt;

    fn assert_close(lhs: &Mat<f64>, rhs: &Mat<f64>, tol: f64) {
        assert_eq!(lhs.nrows(), rhs.nrows());
        assert_eq!(lhs.ncols(), rhs.ncols());
        for col in 0..lhs.ncols() {
            for row in 0..lhs.nrows() {
                let err = (lhs[(row, col)] - rhs[(row, col)]).abs();
                assert!(
                    err <= tol,
                    "entry ({row}, {col}) differs: lhs={}, rhs={}, err={err}, tol={tol}",
                    lhs[(row, col)],
                    rhs[(row, col)],
                );
            }
        }
    }

    fn assert_close_c64(lhs: &Mat<c64>, rhs: &Mat<c64>, tol: f64) {
        assert_eq!(lhs.nrows(), rhs.nrows());
        assert_eq!(lhs.ncols(), rhs.ncols());
        for col in 0..lhs.ncols() {
            for row in 0..lhs.nrows() {
                let err = (lhs[(row, col)] - rhs[(row, col)]).abs1();
                assert!(
                    err <= tol,
                    "entry ({row}, {col}) differs: lhs={:?}, rhs={:?}, err={err}, tol={tol}",
                    lhs[(row, col)],
                    rhs[(row, col)],
                );
            }
        }
    }

    #[test]
    fn continuous_constructor_rejects_bad_dimensions() {
        let a = Mat::<f64>::identity(2, 2);
        let b = Mat::<f64>::zeros(3, 1);
        let c = Mat::<f64>::zeros(1, 2);
        let d = Mat::<f64>::zeros(1, 1);
        let err = ContinuousStateSpace::new(a, b, c, d).unwrap_err();
        assert!(matches!(
            err,
            StateSpaceError::DimensionMismatch { which: "b", .. }
        ));
    }

    #[test]
    fn discrete_constructor_rejects_invalid_sample_time() {
        let a = Mat::<f64>::identity(1, 1);
        let b = Mat::<f64>::zeros(1, 1);
        let c = Mat::<f64>::zeros(1, 1);
        let d = Mat::<f64>::zeros(1, 1);
        let err = DiscreteStateSpace::new(a, b, c, d, 0.0).unwrap_err();
        assert_eq!(err, StateSpaceError::InvalidSampleTime);
    }

    #[test]
    fn zero_feedthrough_constructor_sizes_d_correctly() {
        let a = Mat::<f64>::identity(2, 2);
        let b = Mat::<f64>::zeros(2, 3);
        let c = Mat::<f64>::zeros(4, 2);
        let sys = ContinuousStateSpace::with_zero_feedthrough(a, b, c).unwrap();
        assert_eq!(sys.d().nrows(), 4);
        assert_eq!(sys.d().ncols(), 3);
    }

    #[test]
    fn zoh_discretize_matches_diagonal_closed_form_real_case() {
        let a = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => -1.0,
            (1, 1) => -2.0,
            _ => 0.0,
        });
        let b = Mat::from_fn(2, 2, |row, col| if row == col { 1.0 } else { 0.0 });
        let c = Mat::<f64>::identity(2, 2);
        let d = Mat::<f64>::zeros(2, 2);
        let sys = ContinuousStateSpace::new(a, b, c, d).unwrap();

        let dt = 0.1;
        let disc = sys
            .discretize(dt, DiscretizationMethod::ZeroOrderHold)
            .unwrap();

        let expected_a = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => (-dt).exp(),
            (1, 1) => (-2.0 * dt).exp(),
            _ => 0.0,
        });
        let expected_b = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => 1.0 - (-dt).exp(),
            (1, 1) => (1.0 - (-2.0 * dt).exp()) / 2.0,
            _ => 0.0,
        });

        assert_close(&disc.a, &expected_a, 1e-12);
        assert_close(&disc.b, &expected_b, 1e-12);
    }

    #[test]
    fn bilinear_round_trip_recovers_original_real_system() {
        let a = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => -2.0,
            (0, 1) => 1.0,
            (1, 0) => 0.0,
            (1, 1) => -3.0,
            _ => 0.0,
        });
        let b = Mat::from_fn(2, 1, |row, _| if row == 0 { 1.0 } else { -0.25 });
        let c = Mat::from_fn(1, 2, |_, col| if col == 0 { 0.5 } else { -1.0 });
        let d = Mat::from_fn(1, 1, |_, _| 0.2);
        let sys = ContinuousStateSpace::new(a, b, c, d).unwrap();

        let dt = 0.05;
        let disc = sys
            .discretize(
                dt,
                DiscretizationMethod::Bilinear {
                    prewarp_frequency: None,
                },
            )
            .unwrap();
        let recovered = disc
            .continuousize(ContinuousizationMethod::Bilinear {
                prewarp_frequency: None,
            })
            .unwrap();

        assert_close(&recovered.a, &sys.a, 1e-11);
        assert_close(&recovered.b, &sys.b, 1e-11);
        assert_close(&recovered.c, &sys.c, 1e-11);
        assert_close(&recovered.d, &sys.d, 1e-11);
    }

    #[test]
    fn bilinear_round_trip_recovers_original_complex_system() {
        let a = Mat::from_fn(1, 1, |_, _| c64::new(-1.0, 0.25));
        let b = Mat::from_fn(1, 1, |_, _| c64::new(0.5, -0.1));
        let c = Mat::from_fn(1, 1, |_, _| c64::new(1.25, 0.4));
        let d = Mat::from_fn(1, 1, |_, _| c64::new(-0.2, 0.1));
        let sys = ContinuousStateSpace::new(a, b, c, d).unwrap();

        let disc = sys
            .discretize(
                0.1,
                DiscretizationMethod::Bilinear {
                    prewarp_frequency: None,
                },
            )
            .unwrap();
        let recovered = disc
            .continuousize(ContinuousizationMethod::Bilinear {
                prewarp_frequency: None,
            })
            .unwrap();

        assert_close_c64(&recovered.a, &sys.a, 1e-11);
        assert_close_c64(&recovered.b, &sys.b, 1e-11);
        assert_close_c64(&recovered.c, &sys.c, 1e-11);
        assert_close_c64(&recovered.d, &sys.d, 1e-11);
    }

    #[test]
    fn continuous_gramian_adapters_call_existing_dense_lyapunov_paths() {
        let a = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => -1.0,
            (1, 1) => -2.0,
            _ => 0.0,
        });
        let b = Mat::<f64>::identity(2, 2);
        let c = Mat::<f64>::identity(2, 2);
        let sys = ContinuousStateSpace::with_zero_feedthrough(a, b, c).unwrap();

        let wc = sys.controllability_gramian().unwrap();
        let wo = sys.observability_gramian().unwrap();
        assert!(wc.residual_norm <= 1e-12);
        assert!(wo.residual_norm <= 1e-12);
    }

    #[test]
    fn discrete_gramian_adapters_call_dense_stein_paths() {
        let a = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => 0.25,
            (1, 1) => -0.5,
            _ => 0.0,
        });
        let b = Mat::<f64>::identity(2, 2);
        let c = Mat::<f64>::identity(2, 2);
        let sys = DiscreteStateSpace::with_zero_feedthrough(a, b, c, 0.1).unwrap();

        let wc = sys.controllability_gramian().unwrap();
        let wo = sys.observability_gramian().unwrap();
        assert!(wc.residual_norm <= 1e-12);
        assert!(wo.residual_norm <= 1e-12);
    }

    #[test]
    fn exact_zoh_d2c_is_explicitly_unsupported_for_now() {
        let a = Mat::<f64>::identity(1, 1);
        let b = Mat::<f64>::zeros(1, 1);
        let c = Mat::<f64>::zeros(1, 1);
        let d = Mat::<f64>::zeros(1, 1);
        let sys = DiscreteStateSpace::new(a, b, c, d, 0.1).unwrap();
        let err = sys
            .continuousize(ContinuousizationMethod::ZeroOrderHold)
            .unwrap_err();
        assert!(matches!(err, StateSpaceError::UnsupportedConversion(_)));
    }
}
