//! Dense linear estimator design and discrete-time Kalman filtering.
//!
//! This module is split into two related but distinct layers:
//!
//! - steady-state observer design through continuous/discrete Riccati duality
//! - runtime discrete-time linear Kalman filtering
//!
//! The first layer computes the observer gain `L` and steady-state error
//! covariance `P`. The second layer uses explicit predict/update recursions for
//! a discrete linear Gaussian state estimator.

use super::riccati::{RiccatiError, solve_care_dense, solve_dare_dense};
use super::state_space::{ContinuousStateSpace, DiscreteStateSpace};
use crate::sparse::compensated::{CompensatedField, CompensatedSum};
use core::fmt;
use faer::prelude::Solve;
use faer::{Mat, MatRef};
use faer_traits::ext::ComplexFieldExt;
use faer_traits::{ComplexField, RealField};
use num_traits::{Float, One, Zero};

/// Result of a dense continuous- or discrete-time steady-state estimator solve.
///
/// The gain `L` is the observer gain used in `A - L C`. `covariance` is the
/// steady-state error covariance returned by the dual Riccati equation.
#[derive(Clone, Debug)]
pub struct LqeSolve<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    /// Observer gain.
    pub gain: Mat<T>,
    /// Steady-state estimation-error covariance.
    pub covariance: Mat<T>,
    /// Closed-loop estimator state matrix `A - L C`.
    pub estimator_a: Mat<T>,
    /// Compensated Riccati residual norm.
    pub residual_norm: T::Real,
    /// Whether the dual Riccati solve passed the stabilizing check.
    pub stabilizing: bool,
}

/// Prediction stage of the discrete linear Kalman filter.
#[derive(Clone, Debug)]
pub struct KalmanPrediction<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    /// Predicted state estimate before incorporating the new measurement.
    pub state: Mat<T>,
    /// Predicted covariance before incorporating the new measurement.
    pub covariance: Mat<T>,
}

/// Update result of one discrete linear Kalman filter measurement step.
#[derive(Clone, Debug)]
pub struct KalmanUpdate<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    /// Measurement innovation `y - (C x^- + D u)`.
    pub innovation: Mat<T>,
    /// Innovation covariance `S = C P^- C^H + V`.
    pub innovation_covariance: Mat<T>,
    /// Kalman gain for this update.
    pub gain: Mat<T>,
    /// Updated state estimate.
    pub state: Mat<T>,
    /// Updated covariance.
    pub covariance: Mat<T>,
}

/// Errors produced by dense LQE/DLQE design and discrete Kalman filtering.
#[derive(Debug)]
pub enum EstimatorError {
    /// The dual Riccati solve failed.
    Riccati(RiccatiError),
    /// A supplied matrix had incompatible dimensions.
    DimensionMismatch {
        which: &'static str,
        expected_nrows: usize,
        expected_ncols: usize,
        actual_nrows: usize,
        actual_ncols: usize,
    },
    /// The innovation covariance was singular or numerically unusable.
    SingularInnovationCovariance,
    /// A solve or update produced non-finite output.
    NonFiniteResult { which: &'static str },
}

impl fmt::Display for EstimatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for EstimatorError {}

impl From<RiccatiError> for EstimatorError {
    fn from(value: RiccatiError) -> Self {
        Self::Riccati(value)
    }
}

/// Dense discrete-time linear Kalman filter with explicit predict/update
/// recursions.
///
/// The first implementation assumes the process-noise covariance is already
/// expressed in state coordinates, so the prediction covariance update is
///
/// `P^- = A P A^H + W`
///
/// without a separate process-noise input matrix.
#[derive(Clone, Debug)]
pub struct DiscreteKalmanFilter<T: CompensatedField>
where
    T::Real: Float + Copy,
{
    /// State transition matrix.
    pub a: Mat<T>,
    /// Input matrix.
    pub b: Mat<T>,
    /// Output matrix.
    pub c: Mat<T>,
    /// Feedthrough matrix.
    pub d: Mat<T>,
    /// Process-noise covariance.
    pub w: Mat<T>,
    /// Measurement-noise covariance.
    pub v: Mat<T>,
    /// Current posterior state estimate.
    pub x_hat: Mat<T>,
    /// Current posterior covariance.
    pub p: Mat<T>,
}

/// Solves the dense continuous-time steady-state LQE problem.
///
/// This is the estimator-side dual of continuous-time LQR. The implementation
/// solves the CARE on `(A^H, C^H, W, V)` and converts the resulting dual gain
/// back into the observer gain `L`.
pub fn lqe_dense<T>(
    a: MatRef<'_, T>,
    c: MatRef<'_, T>,
    w: MatRef<'_, T>,
    v: MatRef<'_, T>,
) -> Result<LqeSolve<T>, EstimatorError>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    validate_lqe_dims(a, c, w, v)?;
    // Continuous-time LQE is the dual CARE problem with
    // `(A_dual, B_dual, Q_dual, R_dual) = (A^H, C^H, W, V)`.
    let dual = solve_care_dense(dense_adjoint(a).as_ref(), dense_adjoint(c).as_ref(), w, v)?;
    // The regulator-side gain lives in the dual coordinates, so transpose it
    // back to obtain the observer gain `L`.
    let gain = dense_adjoint(dual.gain.as_ref());
    Ok(LqeSolve {
        estimator_a: estimator_matrix(a, gain.as_ref(), c),
        gain,
        covariance: dual.solution,
        residual_norm: dual.residual_norm,
        stabilizing: dual.stabilizing,
    })
}

/// Solves the dense discrete-time steady-state DLQE problem.
///
/// This is the estimator-side dual of DLQR. The implementation solves the
/// DARE on `(A^H, C^H, W, V)` and converts the resulting dual gain into the
/// predictor-form observer gain `L`.
pub fn dlqe_dense<T>(
    a: MatRef<'_, T>,
    c: MatRef<'_, T>,
    w: MatRef<'_, T>,
    v: MatRef<'_, T>,
) -> Result<LqeSolve<T>, EstimatorError>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    validate_lqe_dims(a, c, w, v)?;
    // Discrete-time DLQE is the DARE dual of DLQR on `(A^H, C^H, W, V)`.
    let dual = solve_dare_dense(dense_adjoint(a).as_ref(), dense_adjoint(c).as_ref(), w, v)?;
    let gain = dense_adjoint(dual.gain.as_ref());
    Ok(LqeSolve {
        estimator_a: estimator_matrix(a, gain.as_ref(), c),
        gain,
        covariance: dual.solution,
        residual_norm: dual.residual_norm,
        stabilizing: dual.stabilizing,
    })
}

impl<T> ContinuousStateSpace<T>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    /// Designs the dense steady-state continuous-time LQE observer.
    pub fn lqe(&self, w: MatRef<'_, T>, v: MatRef<'_, T>) -> Result<LqeSolve<T>, EstimatorError> {
        lqe_dense(self.a(), self.c(), w, v)
    }
}

impl<T> DiscreteStateSpace<T>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    /// Designs the dense steady-state discrete-time DLQE observer.
    pub fn dlqe(&self, w: MatRef<'_, T>, v: MatRef<'_, T>) -> Result<LqeSolve<T>, EstimatorError> {
        dlqe_dense(self.a(), self.c(), w, v)
    }
}

impl<T> DiscreteKalmanFilter<T>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    /// Builds a discrete Kalman filter from explicit model and covariance
    /// matrices.
    pub fn new(
        a: Mat<T>,
        b: Mat<T>,
        c: Mat<T>,
        d: Mat<T>,
        w: Mat<T>,
        v: Mat<T>,
        x_hat: Mat<T>,
        p: Mat<T>,
    ) -> Result<Self, EstimatorError> {
        validate_filter_model(
            a.as_ref(),
            b.as_ref(),
            c.as_ref(),
            d.as_ref(),
            w.as_ref(),
            v.as_ref(),
            x_hat.as_ref(),
            p.as_ref(),
        )?;
        Ok(Self {
            a,
            b,
            c,
            d,
            w,
            v,
            x_hat,
            p,
        })
    }

    /// Builds a discrete Kalman filter from a validated discrete state-space
    /// model plus initial estimate data.
    pub fn from_state_space(
        system: &DiscreteStateSpace<T>,
        w: Mat<T>,
        v: Mat<T>,
        x_hat: Mat<T>,
        p: Mat<T>,
    ) -> Result<Self, EstimatorError> {
        Self::new(
            clone_mat(system.a()),
            clone_mat(system.b()),
            clone_mat(system.c()),
            clone_mat(system.d()),
            w,
            v,
            x_hat,
            p,
        )
    }

    /// Returns the current posterior state estimate.
    #[must_use]
    pub fn state_estimate(&self) -> MatRef<'_, T> {
        self.x_hat.as_ref()
    }

    /// Returns the current posterior covariance.
    #[must_use]
    pub fn covariance(&self) -> MatRef<'_, T> {
        self.p.as_ref()
    }

    /// Computes the prediction step for the supplied input without mutating the
    /// filter state.
    pub fn predict(&self, input: MatRef<'_, T>) -> Result<KalmanPrediction<T>, EstimatorError> {
        validate_column_vector("input", input, self.b.ncols())?;

        // Prediction uses the standard one-step propagation:
        //
        // `x^- = A x + B u`
        // `P^- = A P A^H + W`
        //
        // where `W` is interpreted directly in state coordinates.
        let state = dense_add(
            dense_mul(self.a.as_ref(), self.x_hat.as_ref()).as_ref(),
            dense_mul(self.b.as_ref(), input).as_ref(),
        );
        let covariance = dense_add(
            dense_mul_adjoint_rhs(
                dense_mul(self.a.as_ref(), self.p.as_ref()).as_ref(),
                self.a.as_ref(),
            )
            .as_ref(),
            self.w.as_ref(),
        );

        if !state.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "prediction.state",
            });
        }
        if !covariance.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "prediction.covariance",
            });
        }

        Ok(KalmanPrediction { state, covariance })
    }

    /// Applies one measurement update to an externally supplied prediction.
    pub fn update(
        &self,
        prediction: &KalmanPrediction<T>,
        input: MatRef<'_, T>,
        measurement: MatRef<'_, T>,
    ) -> Result<KalmanUpdate<T>, EstimatorError> {
        validate_column_vector("input", input, self.b.ncols())?;
        validate_column_vector("measurement", measurement, self.c.nrows())?;
        validate_column_vector(
            "prediction.state",
            prediction.state.as_ref(),
            self.a.nrows(),
        )?;
        validate_square(
            "prediction.covariance",
            prediction.covariance.as_ref(),
            self.a.nrows(),
        )?;

        // The update stage forms the innovation
        //
        // `r = y - (C x^- + D u)`
        //
        // and innovation covariance
        //
        // `S = C P^- C^H + V`
        //
        // before solving for the Kalman gain.
        let y_pred = dense_add(
            dense_mul(self.c.as_ref(), prediction.state.as_ref()).as_ref(),
            dense_mul(self.d.as_ref(), input).as_ref(),
        );
        let innovation = dense_sub(measurement, y_pred.as_ref());
        let innovation_covariance = dense_add(
            dense_mul_adjoint_rhs(
                dense_mul(self.c.as_ref(), prediction.covariance.as_ref()).as_ref(),
                self.c.as_ref(),
            )
            .as_ref(),
            self.v.as_ref(),
        );
        let cross = dense_mul_adjoint_rhs(prediction.covariance.as_ref(), self.c.as_ref());
        // `cross * S^-1` is the Kalman gain `K = P^- C^H S^-1`. The helper is
        // written as a right solve so the algebra stays close to that formula.
        let gain = solve_right_checked(
            cross.as_ref(),
            innovation_covariance.as_ref(),
            default_tolerance::<T>(),
            EstimatorError::SingularInnovationCovariance,
        )?;
        let state = dense_add(
            prediction.state.as_ref(),
            dense_mul(gain.as_ref(), innovation.as_ref()).as_ref(),
        );
        // The first pass uses the simple covariance update
        //
        // `P^+ = P^- - K S K^H`
        //
        // rather than the Joseph form. That keeps the implementation aligned
        // with the textbook linear-Gaussian recursion while leaving room for a
        // more conservative update later if needed.
        let covariance = dense_sub(
            prediction.covariance.as_ref(),
            dense_mul_adjoint_rhs(
                dense_mul(gain.as_ref(), innovation_covariance.as_ref()).as_ref(),
                gain.as_ref(),
            )
            .as_ref(),
        );

        if !innovation.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "update.innovation",
            });
        }
        if !gain.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "update.gain",
            });
        }
        if !state.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "update.state",
            });
        }
        if !covariance.as_ref().is_all_finite() {
            return Err(EstimatorError::NonFiniteResult {
                which: "update.covariance",
            });
        }

        Ok(KalmanUpdate {
            innovation,
            innovation_covariance,
            gain,
            state,
            covariance,
        })
    }

    /// Runs one full predict/update cycle and stores the posterior estimate.
    pub fn step(
        &mut self,
        input: MatRef<'_, T>,
        measurement: MatRef<'_, T>,
    ) -> Result<KalmanUpdate<T>, EstimatorError> {
        // `step` is just the stateful convenience wrapper around the pure
        // predict/update stages above.
        let prediction = self.predict(input)?;
        let update = self.update(&prediction, input, measurement)?;
        self.x_hat = clone_mat(update.state.as_ref());
        self.p = clone_mat(update.covariance.as_ref());
        Ok(update)
    }
}

fn validate_lqe_dims<T>(
    a: MatRef<'_, T>,
    c: MatRef<'_, T>,
    w: MatRef<'_, T>,
    v: MatRef<'_, T>,
) -> Result<(), EstimatorError> {
    // The first estimator API assumes process noise already acts in state
    // coordinates, so `W` is `n x n` and `V` is `p x p`.
    validate_square("a", a, a.nrows())?;
    validate_square("w", w, a.nrows())?;
    validate_square("v", v, c.nrows())?;
    if c.ncols() != a.ncols() {
        return Err(EstimatorError::DimensionMismatch {
            which: "c",
            expected_nrows: c.nrows(),
            expected_ncols: a.ncols(),
            actual_nrows: c.nrows(),
            actual_ncols: c.ncols(),
        });
    }
    Ok(())
}

fn validate_filter_model<T>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    c: MatRef<'_, T>,
    d: MatRef<'_, T>,
    w: MatRef<'_, T>,
    v: MatRef<'_, T>,
    x_hat: MatRef<'_, T>,
    p: MatRef<'_, T>,
) -> Result<(), EstimatorError> {
    let n = a.nrows();
    validate_square("a", a, n)?;
    validate_square("w", w, n)?;
    validate_square("p", p, n)?;
    if b.nrows() != n {
        return Err(EstimatorError::DimensionMismatch {
            which: "b",
            expected_nrows: n,
            expected_ncols: b.ncols(),
            actual_nrows: b.nrows(),
            actual_ncols: b.ncols(),
        });
    }
    if c.ncols() != n {
        return Err(EstimatorError::DimensionMismatch {
            which: "c",
            expected_nrows: c.nrows(),
            expected_ncols: n,
            actual_nrows: c.nrows(),
            actual_ncols: c.ncols(),
        });
    }
    if d.nrows() != c.nrows() || d.ncols() != b.ncols() {
        return Err(EstimatorError::DimensionMismatch {
            which: "d",
            expected_nrows: c.nrows(),
            expected_ncols: b.ncols(),
            actual_nrows: d.nrows(),
            actual_ncols: d.ncols(),
        });
    }
    validate_square("v", v, c.nrows())?;
    validate_column_vector("x_hat", x_hat, n)?;
    Ok(())
}

fn validate_square<T>(
    which: &'static str,
    matrix: MatRef<'_, T>,
    expected_dim: usize,
) -> Result<(), EstimatorError> {
    if matrix.nrows() != expected_dim || matrix.ncols() != expected_dim {
        return Err(EstimatorError::DimensionMismatch {
            which,
            expected_nrows: expected_dim,
            expected_ncols: expected_dim,
            actual_nrows: matrix.nrows(),
            actual_ncols: matrix.ncols(),
        });
    }
    Ok(())
}

fn validate_column_vector<T>(
    which: &'static str,
    matrix: MatRef<'_, T>,
    expected_nrows: usize,
) -> Result<(), EstimatorError> {
    if matrix.nrows() != expected_nrows || matrix.ncols() != 1 {
        return Err(EstimatorError::DimensionMismatch {
            which,
            expected_nrows,
            expected_ncols: 1,
            actual_nrows: matrix.nrows(),
            actual_ncols: matrix.ncols(),
        });
    }
    Ok(())
}

fn default_tolerance<T>() -> T::Real
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    T::Real::epsilon().sqrt()
}

fn estimator_matrix<T>(a: MatRef<'_, T>, l: MatRef<'_, T>, c: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    // Both continuous and discrete steady-state observer design report the
    // estimator dynamics in the same algebraic form `A - L C`.
    dense_sub(a, dense_mul(l, c).as_ref())
}

/// Solves `lhs * X = rhs` and rejects numerically unusable results.
///
/// This is used for innovation-covariance solves in the runtime filter and for
/// the small dense dual-gain recovery steps.
fn solve_left_checked<T>(
    lhs: MatRef<'_, T>,
    rhs: MatRef<'_, T>,
    tol: T::Real,
    err: EstimatorError,
) -> Result<Mat<T>, EstimatorError>
where
    T: ComplexField + Copy,
    T::Real: Float + Copy,
{
    let solution = lhs.full_piv_lu().solve(rhs);
    if !solution.as_ref().is_all_finite() {
        return Err(err);
    }

    let residual = dense_sub_plain(dense_mul_plain(lhs, solution.as_ref()).as_ref(), rhs);
    let residual_norm = frobenius_norm_plain(residual.as_ref());
    let scale = frobenius_norm_plain(lhs) * frobenius_norm_plain(solution.as_ref())
        + frobenius_norm_plain(rhs);
    let one = <T::Real as One>::one();
    let threshold = scale.max(one) * tol * (one + one);
    if !residual_norm.is_finite() || residual_norm > threshold {
        return Err(err);
    }

    Ok(solution)
}

/// Solves `X * lhs = rhs` by transposing into [`solve_left_checked`].
///
/// The Kalman gain formula naturally appears as a right solve
/// `K S = P^- C^H`, so this wrapper keeps the calling code in that form.
fn solve_right_checked<T>(
    rhs_left: MatRef<'_, T>,
    lhs_right: MatRef<'_, T>,
    tol: T::Real,
    err: EstimatorError,
) -> Result<Mat<T>, EstimatorError>
where
    T: ComplexField + Copy,
    T::Real: Float + Copy,
{
    let lhs_t = dense_transpose(lhs_right);
    let rhs_t = dense_transpose(rhs_left);
    let solved_t = solve_left_checked(lhs_t.as_ref(), rhs_t.as_ref(), tol, err)?;
    Ok(dense_transpose(solved_t.as_ref()))
}

fn clone_mat<T: Copy>(matrix: MatRef<'_, T>) -> Mat<T> {
    Mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        matrix[(row, col)]
    })
}

fn dense_adjoint<T>(matrix: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    Mat::from_fn(matrix.ncols(), matrix.nrows(), |row, col| {
        matrix[(col, row)].conj()
    })
}

fn dense_transpose<T: Copy>(matrix: MatRef<'_, T>) -> Mat<T> {
    Mat::from_fn(matrix.ncols(), matrix.nrows(), |row, col| {
        matrix[(col, row)]
    })
}

fn dense_mul<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    Mat::from_fn(lhs.nrows(), rhs.ncols(), |row, col| {
        let mut acc = CompensatedSum::<T>::default();
        for k in 0..lhs.ncols() {
            acc.add(lhs[(row, k)] * rhs[(k, col)]);
        }
        acc.finish()
    })
}

fn dense_mul_adjoint_rhs<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    Mat::from_fn(lhs.nrows(), rhs.nrows(), |row, col| {
        let mut acc = CompensatedSum::<T>::default();
        for k in 0..lhs.ncols() {
            acc.add(lhs[(row, k)] * rhs[(col, k)].conj());
        }
        acc.finish()
    })
}

fn dense_add<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| {
        let mut acc = CompensatedSum::<T>::default();
        acc.add(lhs[(row, col)]);
        acc.add(rhs[(row, col)]);
        acc.finish()
    })
}

fn dense_sub<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| {
        let mut acc = CompensatedSum::<T>::default();
        acc.add(lhs[(row, col)]);
        acc.add(-rhs[(row, col)]);
        acc.finish()
    })
}

fn dense_mul_plain<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    Mat::from_fn(lhs.nrows(), rhs.ncols(), |row, col| {
        let mut acc = T::zero();
        for k in 0..lhs.ncols() {
            acc = acc + lhs[(row, k)] * rhs[(k, col)];
        }
        acc
    })
}

fn dense_sub_plain<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| {
        lhs[(row, col)] - rhs[(row, col)]
    })
}

fn frobenius_norm_plain<T>(matrix: MatRef<'_, T>) -> T::Real
where
    T: ComplexField + Copy,
    T::Real: Float + Copy,
{
    let mut acc = <T::Real as Zero>::zero();
    for col in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            acc = acc + matrix[(row, col)].abs2();
        }
    }
    acc.sqrt()
}

#[cfg(test)]
mod test {
    use super::{DiscreteKalmanFilter, EstimatorError, dlqe_dense, lqe_dense};
    use crate::control::state_space::{ContinuousStateSpace, DiscreteStateSpace};
    use faer::Mat;
    use faer_traits::ext::ComplexFieldExt;

    fn assert_close<T>(lhs: &Mat<T>, rhs: &Mat<T>, tol: T::Real)
    where
        T: crate::sparse::compensated::CompensatedField,
        T::Real: num_traits::Float + Copy,
    {
        assert_eq!(lhs.nrows(), rhs.nrows());
        assert_eq!(lhs.ncols(), rhs.ncols());
        for col in 0..lhs.ncols() {
            for row in 0..lhs.nrows() {
                let err = (lhs[(row, col)] - rhs[(row, col)]).abs();
                assert!(
                    err <= tol,
                    "entry ({row}, {col}) mismatch: err={err:?}, tol={tol:?}",
                );
            }
        }
    }

    #[test]
    fn lqe_matches_scalar_dual_closed_form() {
        let a = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let c = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let w = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let v = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let solve = lqe_dense(a.as_ref(), c.as_ref(), w.as_ref(), v.as_ref()).unwrap();

        let expected = 1.0 + 2.0f64.sqrt();
        assert!((solve.covariance[(0, 0)] - expected).abs() < 1.0e-10);
        assert!((solve.gain[(0, 0)] - expected).abs() < 1.0e-10);
        assert!((solve.estimator_a[(0, 0)] + 2.0f64.sqrt()).abs() < 1.0e-10);
        assert!(solve.stabilizing);
    }

    #[test]
    fn dlqe_matches_scalar_dual_closed_form() {
        let a = Mat::from_fn(1, 1, |_, _| 1.2f64);
        let c = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let w = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let v = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let solve = dlqe_dense(a.as_ref(), c.as_ref(), w.as_ref(), v.as_ref()).unwrap();

        let p = (1.44 + (1.44f64 * 1.44 + 4.0).sqrt()) / 2.0;
        let expected_l = 1.2 * p / (1.0 + p);
        assert!((solve.covariance[(0, 0)] - p).abs() < 1.0e-10);
        assert!((solve.gain[(0, 0)] - expected_l).abs() < 1.0e-10);
        assert!((solve.estimator_a[(0, 0)] - (1.2 - expected_l)).abs() < 1.0e-10);
        assert!(solve.stabilizing);
    }

    #[test]
    fn lqe_and_dlqe_state_space_methods_match_free_functions() {
        let a = Mat::from_fn(
            2,
            2,
            |row, col| if row == col { 1.0 + row as f64 } else { 0.0 },
        );
        let b = Mat::zeros(2, 1);
        let c = Mat::from_fn(2, 2, |row, col| if row == col { 1.0 } else { 0.0 });
        let d = Mat::zeros(2, 1);
        let w = Mat::from_fn(
            2,
            2,
            |row, col| if row == col { 1.0 + row as f64 } else { 0.0 },
        );
        let v = Mat::from_fn(2, 2, |row, col| if row == col { 1.0 } else { 0.0 });

        let continuous =
            ContinuousStateSpace::new(a.clone(), b.clone(), c.clone(), d.clone()).unwrap();
        let discrete = DiscreteStateSpace::new(a.clone(), b, c.clone(), d, 0.1).unwrap();

        let free_lqe = lqe_dense(a.as_ref(), c.as_ref(), w.as_ref(), v.as_ref()).unwrap();
        let method_lqe = continuous.lqe(w.as_ref(), v.as_ref()).unwrap();
        assert_close(&free_lqe.gain, &method_lqe.gain, 1.0e-12);

        let free_dlqe = dlqe_dense(a.as_ref(), c.as_ref(), w.as_ref(), v.as_ref()).unwrap();
        let method_dlqe = discrete.dlqe(w.as_ref(), v.as_ref()).unwrap();
        assert_close(&free_dlqe.gain, &method_dlqe.gain, 1.0e-12);
    }

    #[test]
    fn discrete_kalman_predict_update_matches_scalar_reference() {
        let a = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let b = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let c = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let d = Mat::from_fn(1, 1, |_, _| 0.0f64);
        let w = Mat::from_fn(1, 1, |_, _| 0.25f64);
        let v = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let x0 = Mat::from_fn(1, 1, |_, _| 0.0f64);
        let p0 = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let filter = DiscreteKalmanFilter::new(a, b, c, d, w, v, x0, p0).unwrap();

        let u = Mat::from_fn(1, 1, |_, _| 2.0f64);
        let pred = filter.predict(u.as_ref()).unwrap();
        assert!((pred.state[(0, 0)] - 2.0).abs() < 1.0e-12);
        assert!((pred.covariance[(0, 0)] - 1.25).abs() < 1.0e-12);

        let y = Mat::from_fn(1, 1, |_, _| 1.5f64);
        let update = filter.update(&pred, u.as_ref(), y.as_ref()).unwrap();
        let expected_k = 1.25 / 2.25;
        let expected_x = 2.0 + expected_k * (1.5 - 2.0);
        let expected_p = 1.25 - expected_k * 2.25 * expected_k;
        assert!((update.gain[(0, 0)] - expected_k).abs() < 1.0e-12);
        assert!((update.state[(0, 0)] - expected_x).abs() < 1.0e-12);
        assert!((update.covariance[(0, 0)] - expected_p).abs() < 1.0e-12);
    }

    #[test]
    fn discrete_kalman_step_updates_internal_state() {
        let system = DiscreteStateSpace::new(
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            1.0,
        )
        .unwrap();
        let mut filter = DiscreteKalmanFilter::from_state_space(
            &system,
            Mat::from_fn(1, 1, |_, _| 0.1f64),
            Mat::from_fn(1, 1, |_, _| 0.2f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
        )
        .unwrap();

        let u = Mat::from_fn(1, 1, |_, _| 0.0f64);
        let y = Mat::from_fn(1, 1, |_, _| 1.0f64);
        let update = filter.step(u.as_ref(), y.as_ref()).unwrap();
        assert_close(
            &super::clone_mat(filter.state_estimate()),
            &update.state,
            1.0e-12,
        );
        assert_close(
            &super::clone_mat(filter.covariance()),
            &update.covariance,
            1.0e-12,
        );
    }

    #[test]
    fn discrete_kalman_rejects_singular_innovation_covariance() {
        let filter = DiscreteKalmanFilter::new(
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 1.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
            Mat::from_fn(1, 1, |_, _| 0.0f64),
        )
        .unwrap();
        let u = Mat::from_fn(1, 1, |_, _| 0.0f64);
        let pred = filter.predict(u.as_ref()).unwrap();
        let y = Mat::from_fn(1, 1, |_, _| 0.0f64);
        let err = filter.update(&pred, u.as_ref(), y.as_ref()).unwrap_err();
        assert!(matches!(err, EstimatorError::SingularInnovationCovariance));
    }
}
