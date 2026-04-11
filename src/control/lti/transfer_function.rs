use super::error::LtiError;
use super::sos::Sos;
use super::util::{
    normalize_ratio, poly_eval, poly_roots, real_poly_from_roots, trim_leading_zeros,
    validate_sample_time,
};
use super::zpk::Zpk;
use crate::control::state_space::{
    ContinuousStateSpace, ContinuousTime, DiscreteStateSpace, DiscreteTime,
};
use faer::complex::Complex;
use faer::prelude::Solve;
use faer::{Mat, MatRef};
use faer_traits::RealField;
use faer_traits::math_utils::{eps, from_f64};
use num_traits::Float;

/// Real-coefficient single-input single-output transfer function.
///
/// Coefficients are stored in descending-power order.
#[derive(Clone, Debug, PartialEq)]
pub struct TransferFunction<R, Domain> {
    numerator: Vec<R>,
    denominator: Vec<R>,
    domain: Domain,
}

/// Continuous-time SISO transfer function.
pub type ContinuousTransferFunction<R> = TransferFunction<R, ContinuousTime>;

/// Discrete-time SISO transfer function.
pub type DiscreteTransferFunction<R> = TransferFunction<R, DiscreteTime<R>>;

impl<R, Domain> TransferFunction<R, Domain>
where
    R: Float + Copy + RealField,
    Domain: Clone,
{
    /// Creates a normalized transfer function from numerator/denominator
    /// coefficients in descending-power order.
    pub fn new(
        numerator: impl Into<Vec<R>>,
        denominator: impl Into<Vec<R>>,
        domain: Domain,
    ) -> Result<Self, LtiError> {
        let (numerator, denominator) = normalize_ratio(&numerator.into(), &denominator.into())?;
        Ok(Self {
            numerator,
            denominator,
            domain,
        })
    }

    /// Numerator coefficients in descending-power order.
    #[must_use]
    pub fn numerator(&self) -> &[R] {
        &self.numerator
    }

    /// Denominator coefficients in descending-power order.
    #[must_use]
    pub fn denominator(&self) -> &[R] {
        &self.denominator
    }

    /// Domain metadata carried by the transfer function.
    #[must_use]
    pub fn domain(&self) -> &Domain {
        &self.domain
    }

    /// Evaluates the rational transfer function at the supplied complex point.
    #[must_use]
    pub fn evaluate(&self, point: Complex<R>) -> Complex<R> {
        poly_eval(&self.numerator, point) / poly_eval(&self.denominator, point)
    }

    /// Converts coefficient form into zeros/poles/gain form.
    pub fn to_zpk(&self) -> Result<Zpk<R, Domain>, LtiError> {
        let zeros = poly_roots(&self.numerator)?;
        let poles = poly_roots(&self.denominator)?;
        let gain = self.numerator[0] / self.denominator[0];
        Zpk::new(zeros, poles, gain, self.domain.clone())
    }

    /// Converts the transfer function into a second-order-section cascade.
    pub fn to_sos(&self) -> Result<Sos<R, Domain>, LtiError> {
        self.to_zpk()?.to_sos()
    }
}

impl<R> ContinuousTransferFunction<R>
where
    R: Float + Copy + RealField,
{
    /// Creates a continuous-time transfer function.
    pub fn continuous(
        numerator: impl Into<Vec<R>>,
        denominator: impl Into<Vec<R>>,
    ) -> Result<Self, LtiError> {
        Self::new(numerator, denominator, ContinuousTime)
    }

    /// Realizes the proper transfer function as a dense continuous-time
    /// state-space model in controllable companion form.
    pub fn to_state_space(&self) -> Result<ContinuousStateSpace<R>, LtiError> {
        let (a, b, c, d) = companion_realization(self.numerator(), self.denominator())?;
        Ok(ContinuousStateSpace::new(a, b, c, d)?)
    }
}

impl<R> DiscreteTransferFunction<R>
where
    R: Float + Copy + RealField,
{
    /// Creates a discrete-time transfer function with explicit sample time.
    pub fn discrete(
        numerator: impl Into<Vec<R>>,
        denominator: impl Into<Vec<R>>,
        sample_time: R,
    ) -> Result<Self, LtiError> {
        validate_sample_time(sample_time)?;
        Self::new(numerator, denominator, DiscreteTime::new(sample_time))
    }

    /// Sample interval carried by the discrete-time representation.
    #[must_use]
    pub fn sample_time(&self) -> R {
        self.domain.sample_time()
    }

    /// Realizes the proper transfer function as a dense discrete-time
    /// state-space model in controllable companion form.
    pub fn to_state_space(&self) -> Result<DiscreteStateSpace<R>, LtiError> {
        let (a, b, c, d) = companion_realization(self.numerator(), self.denominator())?;
        Ok(DiscreteStateSpace::new(a, b, c, d, self.sample_time())?)
    }
}

impl<R> ContinuousStateSpace<R>
where
    R: Float + Copy + RealField,
{
    /// Converts the dense real SISO continuous-time state-space model into
    /// coefficient form.
    ///
    /// The returned transfer function represents the same input/output map as
    /// the current realization. It does not attempt pole-zero cancellation for
    /// nonminimal systems.
    pub fn to_transfer_function(&self) -> Result<ContinuousTransferFunction<R>, LtiError> {
        ensure_siso(self.ninputs(), self.noutputs())?;
        state_space_to_transfer_function(self.a(), self.b(), self.c(), self.d(), ContinuousTime)
    }

    /// Converts the continuous-time state-space model into zero/pole/gain
    /// form through `TransferFunction`.
    pub fn to_zpk(&self) -> Result<Zpk<R, ContinuousTime>, LtiError> {
        self.to_transfer_function()?.to_zpk()
    }

    /// Converts the continuous-time state-space model into a second-order-
    /// section cascade through `TransferFunction`.
    pub fn to_sos(&self) -> Result<Sos<R, ContinuousTime>, LtiError> {
        self.to_transfer_function()?.to_sos()
    }
}

impl<R> DiscreteStateSpace<R>
where
    R: Float + Copy + RealField,
{
    /// Converts the dense real SISO discrete-time state-space model into
    /// coefficient form.
    pub fn to_transfer_function(&self) -> Result<DiscreteTransferFunction<R>, LtiError> {
        ensure_siso(self.ninputs(), self.noutputs())?;
        state_space_to_transfer_function(
            self.a(),
            self.b(),
            self.c(),
            self.d(),
            DiscreteTime::new(self.sample_time()),
        )
    }

    /// Converts the discrete-time state-space model into zero/pole/gain form
    /// through `TransferFunction`.
    pub fn to_zpk(&self) -> Result<Zpk<R, DiscreteTime<R>>, LtiError> {
        self.to_transfer_function()?.to_zpk()
    }

    /// Converts the discrete-time state-space model into a second-order-
    /// section cascade through `TransferFunction`.
    pub fn to_sos(&self) -> Result<Sos<R, DiscreteTime<R>>, LtiError> {
        self.to_transfer_function()?.to_sos()
    }
}

fn ensure_siso(ninputs: usize, noutputs: usize) -> Result<(), LtiError> {
    if ninputs == 1 && noutputs == 1 {
        Ok(())
    } else {
        Err(LtiError::NonSisoStateSpace { ninputs, noutputs })
    }
}

fn companion_realization<R>(
    numerator: &[R],
    denominator: &[R],
) -> Result<(Mat<R>, Mat<R>, Mat<R>, Mat<R>), LtiError>
where
    R: Float + Copy + RealField,
{
    let numerator = trim_leading_zeros(numerator);
    let denominator = trim_leading_zeros(denominator);
    if denominator.is_empty() {
        return Err(LtiError::EmptyPolynomial {
            which: "denominator",
        });
    }
    if denominator[0] == R::zero() {
        return Err(LtiError::ZeroLeadingCoefficient {
            which: "denominator",
        });
    }

    let n = denominator.len() - 1;
    let m = numerator.len() - 1;
    if m > n {
        return Err(LtiError::ImproperTransferFunction {
            numerator_degree: m,
            denominator_degree: n,
        });
    }

    if n == 0 {
        let d = numerator[0] / denominator[0];
        return Ok((
            Mat::zeros(0, 0),
            Mat::zeros(0, 1),
            Mat::zeros(1, 0),
            Mat::from_fn(1, 1, |_, _| d),
        ));
    }

    let (numerator, denominator) = normalize_ratio(&numerator, &denominator)?;
    let mut padded_numerator = vec![R::zero(); n + 1];
    let offset = n + 1 - numerator.len();
    for (idx, &coef) in numerator.iter().enumerate() {
        padded_numerator[offset + idx] = coef;
    }
    let direct = padded_numerator[0];

    let mut a = Mat::<R>::zeros(n, n);
    for row in 0..(n - 1) {
        a[(row, row + 1)] = R::one();
    }
    for col in 0..n {
        a[(n - 1, col)] = -denominator[n - col];
    }

    let b = Mat::from_fn(
        n,
        1,
        |row, _| if row + 1 == n { R::one() } else { R::zero() },
    );
    let c = Mat::from_fn(1, n, |_, col| {
        let idx = n - col;
        padded_numerator[idx] - denominator[idx] * direct
    });
    let d = Mat::from_fn(1, 1, |_, _| direct);
    Ok((a, b, c, d))
}

fn state_space_to_transfer_function<R, Domain>(
    a: MatRef<'_, R>,
    b: MatRef<'_, R>,
    c: MatRef<'_, R>,
    d: MatRef<'_, R>,
    domain: Domain,
) -> Result<TransferFunction<R, Domain>, LtiError>
where
    R: Float + Copy + RealField,
    Domain: Clone,
{
    if a.nrows() == 0 {
        return TransferFunction::new(vec![d[(0, 0)]], vec![R::one()], domain);
    }

    let poles = a.eigenvalues()?;
    let denominator = real_poly_from_roots(&poles, "state_space_poles")?;
    let numerator = interpolate_numerator(a, b, c, d, &denominator)?;
    TransferFunction::new(numerator, denominator, domain)
}

fn interpolate_numerator<R>(
    a: MatRef<'_, R>,
    b: MatRef<'_, R>,
    c: MatRef<'_, R>,
    d: MatRef<'_, R>,
    denominator: &[R],
) -> Result<Vec<R>, LtiError>
where
    R: Float + Copy + RealField,
{
    let degree = denominator.len() - 1;
    let points = interpolation_points(denominator, degree + 1);
    let vandermonde = Mat::from_fn(points.len(), degree + 1, |row, col| {
        let power = degree - col;
        points[row].powi(power as i32)
    });
    let rhs_values = points
        .iter()
        .map(|&sample| {
            let point = Complex::new(sample, R::zero());
            let gain = dense_transfer_siso(a, b, c, d, point)?;
            let imag_tol =
                (gain.re.abs() + gain.im.abs() + R::one()) * from_f64::<R>(128.0) * eps::<R>();
            if gain.im.abs() > imag_tol {
                return Err(LtiError::NonFiniteResult {
                    which: "state_space_to_transfer_function.imaginary_interpolation_value",
                });
            }
            Ok(gain.re * poly_eval(denominator, point).re)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rhs = Mat::from_fn(points.len(), 1, |row, _| rhs_values[row]);
    let solution = vandermonde.full_piv_lu().solve(rhs.as_ref());
    if !all_finite_real(solution.as_ref()) {
        return Err(LtiError::NonFiniteResult {
            which: "state_space_to_transfer_function.solve",
        });
    }
    let coeffs = (0..solution.nrows())
        .map(|row| solution[(row, 0)])
        .collect::<Vec<_>>();
    Ok(trim_small_leading_coeffs(&coeffs))
}

fn interpolation_points<R>(denominator: &[R], count: usize) -> Vec<R>
where
    R: Float + Copy + RealField,
{
    let mut points = Vec::with_capacity(count);
    let mut k = 0usize;
    while points.len() < count {
        let candidate = match k {
            0 => R::zero(),
            _ if k % 2 == 1 => R::from((k + 1) / 2).unwrap_or_else(R::one),
            _ => -R::from(k / 2).unwrap_or_else(R::one),
        };
        let value = poly_eval(denominator, Complex::new(candidate, R::zero())).norm();
        let threshold = from_f64::<R>(256.0) * eps::<R>();
        if value > threshold {
            points.push(candidate);
        }
        k += 1;
    }
    points
}

fn dense_transfer_siso<R>(
    a: MatRef<'_, R>,
    b: MatRef<'_, R>,
    c: MatRef<'_, R>,
    d: MatRef<'_, R>,
    point: Complex<R>,
) -> Result<Complex<R>, LtiError>
where
    R: Float + Copy + RealField,
{
    let a = Mat::from_fn(a.nrows(), a.ncols(), |row, col| {
        Complex::new(a[(row, col)], R::zero())
    });
    let b = Mat::from_fn(b.nrows(), b.ncols(), |row, col| {
        Complex::new(b[(row, col)], R::zero())
    });
    let c = Mat::from_fn(c.nrows(), c.ncols(), |row, col| {
        Complex::new(c[(row, col)], R::zero())
    });
    let d = Complex::new(d[(0, 0)], R::zero());

    let lhs = Mat::from_fn(a.nrows(), a.ncols(), |row, col| {
        if row == col {
            point - a[(row, col)]
        } else {
            -a[(row, col)]
        }
    });
    let sol = lhs.full_piv_lu().solve(b.as_ref());
    let mut value = d;
    for k in 0..a.nrows() {
        value += c[(0, k)] * sol[(k, 0)];
    }
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(LtiError::NonFiniteResult {
            which: "state_space_to_transfer_function.transfer_at",
        })
    }
}

fn all_finite_real<R: Float + Copy + RealField>(matrix: MatRef<'_, R>) -> bool {
    for col in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            if !matrix[(row, col)].is_finite() {
                return false;
            }
        }
    }
    true
}

fn trim_small_leading_coeffs<R>(coeffs: &[R]) -> Vec<R>
where
    R: Float + Copy + RealField,
{
    let scale = coeffs
        .iter()
        .fold(R::one(), |acc, &value| acc.max(value.abs()));
    let tol = scale * from_f64::<R>(128.0) * eps::<R>();
    let first_nz = coeffs.iter().position(|&value| value.abs() > tol);
    match first_nz {
        Some(idx) => coeffs[idx..].to_vec(),
        None => vec![R::zero()],
    }
}

#[cfg(test)]
mod tests {
    use super::{ContinuousTransferFunction, DiscreteTransferFunction};
    use crate::control::lti::LtiError;
    use crate::control::lti::Sos;
    use faer::complex::Complex;

    fn assert_coeffs_close(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (&lhs, &rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let err = (lhs - rhs).abs();
            assert!(
                err <= tol,
                "coefficient {idx} differs: lhs={lhs}, rhs={rhs}, err={err}, tol={tol}",
            );
        }
    }

    #[test]
    fn constructor_normalizes_denominator() {
        let tf = ContinuousTransferFunction::continuous(vec![2.0, 4.0], vec![2.0, 6.0]).unwrap();
        assert_eq!(tf.numerator(), &[1.0, 2.0]);
        assert_eq!(tf.denominator(), &[1.0, 3.0]);
    }

    #[test]
    fn zpk_round_trip_preserves_coefficients() {
        let tf = ContinuousTransferFunction::continuous(vec![1.0, 3.0, 2.0], vec![1.0, 5.0, 6.0])
            .unwrap();
        let back = tf.to_zpk().unwrap().to_transfer_function().unwrap();
        assert_coeffs_close(back.numerator(), tf.numerator(), 1.0e-12);
        assert_coeffs_close(back.denominator(), tf.denominator(), 1.0e-12);
    }

    #[test]
    fn sos_round_trip_preserves_coefficients() {
        let tf = DiscreteTransferFunction::discrete(
            vec![1.0, 0.0, 5.0, 0.0, 4.0],
            vec![1.0, 0.0, 6.0, 0.0, 9.0],
            0.1,
        )
        .unwrap();
        let sos = tf.to_sos().unwrap();
        let back = sos.to_transfer_function().unwrap();
        assert_coeffs_close(back.numerator(), tf.numerator(), 1.0e-12);
        assert_coeffs_close(back.denominator(), tf.denominator(), 1.0e-12);
        assert_eq!(back.sample_time(), 0.1);
    }

    #[test]
    fn evaluate_matches_closed_form() {
        let tf = ContinuousTransferFunction::continuous(vec![2.0], vec![1.0, 3.0]).unwrap();
        let point = Complex::new(1.0, 2.0);
        let expected = Complex::new(2.0, 0.0) / (point + Complex::new(3.0, 0.0));
        assert!((tf.evaluate(point) - expected).norm() <= 1.0e-12);
    }

    #[test]
    fn discrete_constructor_validates_sample_time() {
        let err = DiscreteTransferFunction::discrete(vec![1.0], vec![1.0, -0.5], 0.0).unwrap_err();
        assert!(matches!(
            err,
            crate::control::lti::LtiError::InvalidSampleTime
        ));
    }

    #[test]
    fn sos_identity_section_round_trip() {
        let sos = Sos::continuous(
            vec![
                crate::control::lti::SecondOrderSection::new([1.0, 0.0, 0.0], [1.0, -0.5, 0.0])
                    .unwrap(),
            ],
            2.0,
        )
        .unwrap();
        let tf = sos.to_transfer_function().unwrap();
        assert_eq!(tf.numerator(), &[2.0, 0.0, 0.0]);
        assert_eq!(tf.denominator(), &[1.0, -0.5, 0.0]);
    }

    #[test]
    fn continuous_transfer_function_realizes_and_round_trips_through_state_space() {
        let tf =
            ContinuousTransferFunction::continuous(vec![2.0, 5.0], vec![1.0, 3.0, 2.0]).unwrap();
        let ss = tf.to_state_space().unwrap();
        let back = ss.to_transfer_function().unwrap();
        assert_coeffs_close(back.numerator(), tf.numerator(), 1.0e-10);
        assert_coeffs_close(back.denominator(), tf.denominator(), 1.0e-10);
    }

    #[test]
    fn discrete_transfer_function_realizes_and_preserves_sample_time() {
        let tf = DiscreteTransferFunction::discrete(vec![1.0, -0.25], vec![1.0, -0.5, 0.125], 0.2)
            .unwrap();
        let ss = tf.to_state_space().unwrap();
        let back = ss.to_transfer_function().unwrap();
        assert_coeffs_close(back.numerator(), tf.numerator(), 1.0e-10);
        assert_coeffs_close(back.denominator(), tf.denominator(), 1.0e-10);
        assert_eq!(back.sample_time(), 0.2);
    }

    #[test]
    fn improper_transfer_function_rejects_state_space_realization() {
        let tf = ContinuousTransferFunction::continuous(vec![1.0, 2.0], vec![1.0]).unwrap();
        let err = tf.to_state_space().unwrap_err();
        assert!(matches!(
            err,
            LtiError::ImproperTransferFunction {
                numerator_degree: 1,
                denominator_degree: 0
            }
        ));
    }

    #[test]
    fn state_space_to_zpk_and_sos_chain_through_transfer_function() {
        let tf =
            ContinuousTransferFunction::continuous(vec![1.0, 3.0], vec![1.0, 5.0, 6.0]).unwrap();
        let ss = tf.to_state_space().unwrap();
        let zpk = ss.to_zpk().unwrap();
        let sos = ss.to_sos().unwrap();
        let tf_from_zpk = zpk.to_transfer_function().unwrap();
        let tf_from_sos = sos.to_transfer_function().unwrap();
        assert_coeffs_close(tf_from_zpk.numerator(), tf.numerator(), 1.0e-10);
        assert_coeffs_close(tf_from_zpk.denominator(), tf.denominator(), 1.0e-10);
        assert_coeffs_close(tf_from_sos.numerator(), tf.numerator(), 1.0e-10);
        assert_coeffs_close(tf_from_sos.denominator(), tf.denominator(), 1.0e-10);
    }
}
