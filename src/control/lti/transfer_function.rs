use super::error::LtiError;
use super::sos::Sos;
use super::util::{normalize_ratio, poly_eval, poly_roots, validate_sample_time};
use super::zpk::Zpk;
use crate::control::state_space::{ContinuousTime, DiscreteTime};
use faer::complex::Complex;
use faer_traits::RealField;
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
}

#[cfg(test)]
mod tests {
    use super::{ContinuousTransferFunction, DiscreteTransferFunction};
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
        let tf =
            ContinuousTransferFunction::continuous(vec![1.0, 3.0, 2.0], vec![1.0, 5.0, 6.0])
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
        assert!(matches!(err, crate::control::lti::LtiError::InvalidSampleTime));
    }

    #[test]
    fn sos_identity_section_round_trip() {
        let sos = Sos::continuous(
            vec![crate::control::lti::SecondOrderSection::new(
                [1.0, 0.0, 0.0],
                [1.0, -0.5, 0.0],
            )
            .unwrap()],
            2.0,
        )
        .unwrap();
        let tf = sos.to_transfer_function().unwrap();
        assert_eq!(tf.numerator(), &[2.0, 0.0, 0.0]);
        assert_eq!(tf.denominator(), &[1.0, -0.5, 0.0]);
    }
}
