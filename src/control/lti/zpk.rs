use super::error::LtiError;
use super::sos::Sos;
use super::transfer_function::TransferFunction;
use super::util::{real_poly_from_roots, validate_sample_time};
use crate::control::state_space::{
    ContinuousStateSpace, ContinuousTime, DiscreteStateSpace, DiscreteTime,
};
use faer::complex::Complex;
use faer_traits::RealField;
use num_traits::Float;

/// Real-gain SISO zero/pole/gain representation.
#[derive(Clone, Debug, PartialEq)]
pub struct Zpk<R, Domain> {
    zeros: Vec<Complex<R>>,
    poles: Vec<Complex<R>>,
    gain: R,
    domain: Domain,
}

/// Continuous-time SISO zero/pole/gain representation.
pub type ContinuousZpk<R> = Zpk<R, ContinuousTime>;

/// Discrete-time SISO zero/pole/gain representation.
pub type DiscreteZpk<R> = Zpk<R, DiscreteTime<R>>;

impl<R, Domain> Zpk<R, Domain>
where
    R: Float + Copy + RealField,
    Domain: Clone,
{
    /// Creates a zero/pole/gain representation.
    pub fn new(
        zeros: impl Into<Vec<Complex<R>>>,
        poles: impl Into<Vec<Complex<R>>>,
        gain: R,
        domain: Domain,
    ) -> Result<Self, LtiError> {
        Ok(Self {
            zeros: zeros.into(),
            poles: poles.into(),
            gain,
            domain,
        })
    }

    /// Zeros of the transfer function.
    #[must_use]
    pub fn zeros(&self) -> &[Complex<R>] {
        &self.zeros
    }

    /// Poles of the transfer function.
    #[must_use]
    pub fn poles(&self) -> &[Complex<R>] {
        &self.poles
    }

    /// Overall scalar gain.
    #[must_use]
    pub fn gain(&self) -> R {
        self.gain
    }

    /// Domain metadata carried by the representation.
    #[must_use]
    pub fn domain(&self) -> &Domain {
        &self.domain
    }

    /// Evaluates the transfer function at the supplied complex point.
    #[must_use]
    pub fn evaluate(&self, point: Complex<R>) -> Complex<R> {
        let num = self
            .zeros
            .iter()
            .fold(Complex::new(self.gain, R::zero()), |acc, &zero| {
                acc * (point - zero)
            });
        let den = self
            .poles
            .iter()
            .fold(Complex::new(R::one(), R::zero()), |acc, &pole| {
                acc * (point - pole)
            });
        num / den
    }

    /// Converts zero/pole/gain form back into coefficient form.
    pub fn to_transfer_function(&self) -> Result<TransferFunction<R, Domain>, LtiError> {
        let mut numerator = real_poly_from_roots(&self.zeros, "zeros")?;
        if let Some(first) = numerator.first_mut() {
            *first = *first * self.gain;
        }
        let denominator = real_poly_from_roots(&self.poles, "poles")?;
        TransferFunction::new(numerator, denominator, self.domain.clone())
    }

    /// Converts zero/pole/gain form into a second-order-section cascade.
    pub fn to_sos(&self) -> Result<Sos<R, Domain>, LtiError> {
        Sos::from_zpk(self)
    }
}

impl<R> ContinuousZpk<R>
where
    R: Float + Copy + RealField,
{
    /// Creates a continuous-time zero/pole/gain representation.
    pub fn continuous(
        zeros: impl Into<Vec<Complex<R>>>,
        poles: impl Into<Vec<Complex<R>>>,
        gain: R,
    ) -> Result<Self, LtiError> {
        Self::new(zeros, poles, gain, ContinuousTime)
    }

    /// Converts zero/pole/gain form to continuous-time state space through
    /// `TransferFunction`.
    pub fn to_state_space(&self) -> Result<ContinuousStateSpace<R>, LtiError> {
        self.to_transfer_function()?.to_state_space()
    }
}

impl<R> DiscreteZpk<R>
where
    R: Float + Copy + RealField,
{
    /// Creates a discrete-time zero/pole/gain representation.
    pub fn discrete(
        zeros: impl Into<Vec<Complex<R>>>,
        poles: impl Into<Vec<Complex<R>>>,
        gain: R,
        sample_time: R,
    ) -> Result<Self, LtiError> {
        validate_sample_time(sample_time)?;
        Self::new(zeros, poles, gain, DiscreteTime::new(sample_time))
    }

    /// Sample interval carried by the discrete-time representation.
    #[must_use]
    pub fn sample_time(&self) -> R {
        self.domain.sample_time()
    }

    /// Converts zero/pole/gain form to discrete-time state space through
    /// `TransferFunction`.
    pub fn to_state_space(&self) -> Result<DiscreteStateSpace<R>, LtiError> {
        self.to_transfer_function()?.to_state_space()
    }
}
