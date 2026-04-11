use super::error::LtiError;
use crate::control::state_space::{ContinuousStateSpace, DiscreteStateSpace, StateSpace};
use faer::complex::Complex;
use faer::prelude::Solve;
use faer::{Mat, MatRef};
use faer_traits::{ComplexField, RealField};
use faer_traits::ext::ComplexFieldExt;
use faer_traits::math_utils::eps;
use num_traits::{Float, One, Zero};

impl<T, Domain> StateSpace<T, Domain>
where
    T: ComplexField + Copy,
    T::Real: Float + Copy + RealField,
{
    /// Returns the poles of the dense state-space model.
    ///
    /// Poles are sorted by descending magnitude, then by descending real part,
    /// then by descending imaginary part. That gives deterministic output for
    /// testing and for later comparison across alternate representations.
    pub fn poles(&self) -> Result<Vec<Complex<T::Real>>, LtiError> {
        let mut poles = self.a.eigenvalues()?;
        poles.sort_by(|lhs, rhs| compare_poles(*lhs, *rhs));
        Ok(poles)
    }

    /// Returns the controllability matrix `[B, AB, ..., A^(n-1) B]`.
    ///
    /// For the first LTI-analysis pass this is a dense reference
    /// implementation. It is appropriate for diagnostics and small/moderate
    /// models, even though it is not the most efficient route for large-scale
    /// systems.
    #[must_use]
    pub fn controllability_matrix(&self) -> Mat<T> {
        let n = self.nstates();
        let m = self.ninputs();
        let mut out = Mat::zeros(n, n * m);
        let mut block = clone_mat(self.b());
        for k in 0..n {
            copy_block(out.as_mut(), 0, k * m, block.as_ref());
            if k + 1 != n {
                block = dense_mul(self.a(), block.as_ref());
            }
        }
        out
    }

    /// Returns the observability matrix `[C; CA; ...; C A^(n-1)]`.
    #[must_use]
    pub fn observability_matrix(&self) -> Mat<T> {
        let n = self.nstates();
        let p = self.noutputs();
        let mut out = Mat::zeros(n * p, n);
        let mut block = clone_mat(self.c());
        for k in 0..n {
            copy_block(out.as_mut(), k * p, 0, block.as_ref());
            if k + 1 != n {
                block = dense_mul(block.as_ref(), self.a());
            }
        }
        out
    }

    /// Returns the numerical rank of the controllability matrix.
    pub fn controllability_rank(&self) -> Result<usize, LtiError> {
        let ctrb = self.controllability_matrix();
        numerical_rank(ctrb.as_ref())
    }

    /// Returns the numerical rank of the controllability matrix using an
    /// explicit singular-value threshold.
    pub fn controllability_rank_with_tol(&self, tol: T::Real) -> Result<usize, LtiError> {
        let ctrb = self.controllability_matrix();
        numerical_rank_with_tol(ctrb.as_ref(), tol)
    }

    /// Returns the numerical rank of the observability matrix.
    pub fn observability_rank(&self) -> Result<usize, LtiError> {
        let obsv = self.observability_matrix();
        numerical_rank(obsv.as_ref())
    }

    /// Returns the numerical rank of the observability matrix using an
    /// explicit singular-value threshold.
    pub fn observability_rank_with_tol(&self, tol: T::Real) -> Result<usize, LtiError> {
        let obsv = self.observability_matrix();
        numerical_rank_with_tol(obsv.as_ref(), tol)
    }

    /// Returns whether the dense model is numerically controllable.
    pub fn is_controllable(&self) -> Result<bool, LtiError> {
        Ok(self.controllability_rank()? == self.nstates())
    }

    /// Returns whether the dense model is numerically controllable using an
    /// explicit singular-value threshold.
    pub fn is_controllable_with_tol(&self, tol: T::Real) -> Result<bool, LtiError> {
        Ok(self.controllability_rank_with_tol(tol)? == self.nstates())
    }

    /// Returns whether the dense model is numerically observable.
    pub fn is_observable(&self) -> Result<bool, LtiError> {
        Ok(self.observability_rank()? == self.nstates())
    }

    /// Returns whether the dense model is numerically observable using an
    /// explicit singular-value threshold.
    pub fn is_observable_with_tol(&self, tol: T::Real) -> Result<bool, LtiError> {
        Ok(self.observability_rank_with_tol(tol)? == self.nstates())
    }

    /// Returns whether the dense model is numerically minimal.
    ///
    /// In the first pass, minimality is defined by the usual dense rank tests:
    /// controllable and observable.
    pub fn is_minimal(&self) -> Result<bool, LtiError> {
        Ok(self.is_controllable()? && self.is_observable()?)
    }

    /// Evaluates the transfer matrix at the supplied complex point.
    ///
    /// The caller supplies the point in the natural transform variable for the
    /// domain:
    ///
    /// - continuous-time: `s`
    /// - discrete-time: `z`
    ///
    /// This is the dense reference path:
    ///
    /// `G(point) = C (point I - A)^(-1) B + D`
    pub fn transfer_at(&self, point: Complex<T::Real>) -> Result<Mat<Complex<T::Real>>, LtiError> {
        let a = to_complex_mat(self.a());
        let b = to_complex_mat(self.b());
        let c = to_complex_mat(self.c());
        let d = to_complex_mat(self.d());

        let n = a.nrows();
        let lhs = Mat::from_fn(n, n, |row, col| {
            if row == col {
                point - a[(row, col)]
            } else {
                -a[(row, col)]
            }
        });
        let sol = lhs.full_piv_lu().solve(b.as_ref());
        let gain = dense_mul(c.as_ref(), sol.as_ref());
        let out = Mat::from_fn(gain.nrows(), gain.ncols(), |row, col| {
            gain[(row, col)] + d[(row, col)]
        });
        if all_finite_complex(out.as_ref()) {
            Ok(out)
        } else {
            Err(LtiError::NonFiniteResult {
                which: "transfer_at",
            })
        }
    }
}

impl<T> ContinuousStateSpace<T>
where
    T: ComplexField + Copy,
    T::Real: Float + Copy + RealField,
{
    /// Returns whether all poles lie strictly in the open left half-plane.
    ///
    /// The default tolerance is `sqrt(eps)`, which is conservative enough to
    /// avoid classifying numerically marginal poles as safely stable.
    pub fn is_asymptotically_stable(&self) -> Result<bool, LtiError> {
        self.is_asymptotically_stable_with_tol(eps::<T::Real>().sqrt())
    }

    /// Returns whether all poles lie to the left of `-tol`.
    pub fn is_asymptotically_stable_with_tol(&self, tol: T::Real) -> Result<bool, LtiError> {
        Ok(self.poles()?.into_iter().all(|pole| pole.re < -tol))
    }

    /// Returns the DC gain `G(0)`.
    pub fn dc_gain(&self) -> Result<Mat<Complex<T::Real>>, LtiError> {
        self.transfer_at(Complex::new(
            <T::Real as Zero>::zero(),
            <T::Real as Zero>::zero(),
        ))
    }
}

impl<T> DiscreteStateSpace<T>
where
    T: ComplexField + Copy,
    T::Real: Float + Copy + RealField,
{
    /// Returns whether all poles lie strictly inside the unit disk.
    pub fn is_asymptotically_stable(&self) -> Result<bool, LtiError> {
        self.is_asymptotically_stable_with_tol(eps::<T::Real>().sqrt())
    }

    /// Returns whether all poles satisfy `|pole| < 1 - tol`.
    pub fn is_asymptotically_stable_with_tol(&self, tol: T::Real) -> Result<bool, LtiError> {
        Ok(self
            .poles()?
            .into_iter()
            .all(|pole: Complex<T::Real>| pole.norm() < <T::Real as One>::one() - tol))
    }

    /// Returns the DC gain `G(1)`.
    pub fn dc_gain(&self) -> Result<Mat<Complex<T::Real>>, LtiError> {
        self.transfer_at(Complex::new(
            <T::Real as One>::one(),
            <T::Real as Zero>::zero(),
        ))
    }
}

fn compare_poles<R: Float + Copy>(lhs: Complex<R>, rhs: Complex<R>) -> core::cmp::Ordering {
    rhs.norm()
        .partial_cmp(&lhs.norm())
        .unwrap_or(core::cmp::Ordering::Equal)
        .then_with(|| rhs.re.partial_cmp(&lhs.re).unwrap_or(core::cmp::Ordering::Equal))
        .then_with(|| rhs.im.partial_cmp(&lhs.im).unwrap_or(core::cmp::Ordering::Equal))
}

fn clone_mat<T: Copy>(matrix: MatRef<'_, T>) -> Mat<T> {
    Mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| matrix[(row, col)])
}

fn copy_block<T: Copy>(
    mut dst: faer::MatMut<'_, T>,
    row_offset: usize,
    col_offset: usize,
    src: MatRef<'_, T>,
) {
    for col in 0..src.ncols() {
        for row in 0..src.nrows() {
            dst[(row_offset + row, col_offset + col)] = src[(row, col)];
        }
    }
}

fn dense_mul<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
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

fn numerical_rank<T>(matrix: MatRef<'_, T>) -> Result<usize, LtiError>
where
    T: ComplexField,
    T::Real: Float + Copy + RealField,
{
    let sv = matrix.singular_values()?;
    Ok(rank_from_singular_values(&sv, matrix.nrows(), matrix.ncols(), None))
}

fn numerical_rank_with_tol<T>(matrix: MatRef<'_, T>, tol: T::Real) -> Result<usize, LtiError>
where
    T: ComplexField,
    T::Real: Float + Copy + RealField,
{
    let sv = matrix.singular_values()?;
    Ok(rank_from_singular_values(
        &sv,
        matrix.nrows(),
        matrix.ncols(),
        Some(tol),
    ))
}

fn rank_from_singular_values<R: Float + Copy + RealField>(
    singular_values: &[R],
    nrows: usize,
    ncols: usize,
    tol: Option<R>,
) -> usize {
    let Some(&sigma_max) = singular_values.first() else {
        return 0;
    };
    let threshold = tol.unwrap_or_else(|| {
        let dim = R::from((nrows.max(ncols)) as f64).unwrap_or_else(R::one);
        sigma_max * dim * eps::<R>()
    });
    singular_values
        .iter()
        .take_while(|&&sigma| sigma > threshold)
        .count()
}

fn to_complex_mat<T>(matrix: MatRef<'_, T>) -> Mat<Complex<T::Real>>
where
    T: ComplexField + Copy,
    T::Real: Float + Copy + RealField,
{
    Mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        let value = matrix[(row, col)];
        Complex::new(value.real(), value.imag())
    })
}

fn all_finite_complex<R: Float + Copy + RealField>(matrix: MatRef<'_, Complex<R>>) -> bool {
    for col in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            let value = matrix[(row, col)];
            if !value.re.is_finite() || !value.im.is_finite() {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use crate::control::state_space::{ContinuousStateSpace, DiscreteStateSpace};
    use faer::complex::Complex;
    use faer::Mat;

    fn assert_close_complex(lhs: MatRef<'_, Complex<f64>>, rhs: MatRef<'_, Complex<f64>>, tol: f64) {
        assert_eq!(lhs.nrows(), rhs.nrows());
        assert_eq!(lhs.ncols(), rhs.ncols());
        for col in 0..lhs.ncols() {
            for row in 0..lhs.nrows() {
                let err = (lhs[(row, col)] - rhs[(row, col)]).norm();
                assert!(
                    err <= tol,
                    "entry ({row}, {col}) differs: lhs={:?}, rhs={:?}, err={err}, tol={tol}",
                    lhs[(row, col)],
                    rhs[(row, col)],
                );
            }
        }
    }

    use faer::MatRef;

    #[test]
    fn continuous_poles_and_stability_work() {
        let a = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => -1.0,
            (1, 1) => -3.0,
            _ => 0.0,
        });
        let b = Mat::from_fn(2, 1, |row, _| if row == 0 { 1.0 } else { 0.0 });
        let c = Mat::from_fn(1, 2, |_, col| if col == 0 { 1.0 } else { 0.0 });
        let sys = ContinuousStateSpace::with_zero_feedthrough(a, b, c).unwrap();

        let poles = sys.poles().unwrap();
        assert_eq!(poles.len(), 2);
        assert!(poles[0].norm() >= poles[1].norm());
        assert!(sys.is_asymptotically_stable().unwrap());
    }

    #[test]
    fn discrete_stability_detects_unit_disk_violation() {
        let a = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => 0.5,
            (1, 1) => 1.1,
            _ => 0.0,
        });
        let b = Mat::from_fn(2, 1, |row, _| if row == 0 { 1.0 } else { 0.0 });
        let c = Mat::from_fn(1, 2, |_, col| if col == 0 { 1.0 } else { 0.0 });
        let sys = DiscreteStateSpace::with_zero_feedthrough(a, b, c, 0.1).unwrap();

        assert!(!sys.is_asymptotically_stable().unwrap());
    }

    #[test]
    fn controllability_and_observability_detect_nonminimal_system() {
        let a = Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => 0.0,
            (1, 1) => -1.0,
            _ => 0.0,
        });
        let b = Mat::from_fn(2, 1, |row, _| if row == 0 { 1.0 } else { 0.0 });
        let c = Mat::from_fn(1, 2, |_, col| if col == 0 { 1.0 } else { 0.0 });
        let sys = ContinuousStateSpace::with_zero_feedthrough(a, b, c).unwrap();

        assert_eq!(sys.controllability_rank().unwrap(), 1);
        assert_eq!(sys.observability_rank().unwrap(), 1);
        assert!(!sys.is_minimal().unwrap());
    }

    #[test]
    fn transfer_at_matches_first_order_closed_form() {
        let a = Mat::from_fn(1, 1, |_, _| -2.0);
        let b = Mat::from_fn(1, 1, |_, _| 3.0);
        let c = Mat::from_fn(1, 1, |_, _| 4.0);
        let d = Mat::from_fn(1, 1, |_, _| 5.0);
        let sys = ContinuousStateSpace::new(a, b, c, d).unwrap();

        let point = Complex::new(1.0, 2.0);
        let got = sys.transfer_at(point).unwrap();
        let expected = Mat::from_fn(1, 1, |_, _| Complex::new(5.0, 0.0) + Complex::new(12.0, 0.0) / (point + Complex::new(2.0, 0.0)));
        assert_close_complex(got.as_ref(), expected.as_ref(), 1.0e-12);
    }

    #[test]
    fn dc_gain_matches_transfer_at_zero_or_one() {
        let a = Mat::from_fn(1, 1, |_, _| -2.0);
        let b = Mat::from_fn(1, 1, |_, _| 3.0);
        let c = Mat::from_fn(1, 1, |_, _| 4.0);
        let d = Mat::from_fn(1, 1, |_, _| 5.0);
        let cont = ContinuousStateSpace::new(a.clone(), b.clone(), c.clone(), d.clone()).unwrap();
        let disc = DiscreteStateSpace::new(Mat::from_fn(1, 1, |_, _| 0.25), b, c, d, 0.1).unwrap();

        assert_close_complex(
            cont.dc_gain().unwrap().as_ref(),
            cont.transfer_at(Complex::new(0.0, 0.0)).unwrap().as_ref(),
            1.0e-12,
        );
        assert_close_complex(
            disc.dc_gain().unwrap().as_ref(),
            disc.transfer_at(Complex::new(1.0, 0.0)).unwrap().as_ref(),
            1.0e-12,
        );
    }
}
