use super::error::FilterDesignError;
use super::spec::AnalogFilterFamily;
use crate::control::lti::util::poly_roots;
use crate::control::lti::zpk::ContinuousZpk;
use crate::control::lti::{ContinuousTime, LtiError};
use faer::complex::Complex;
use faer_traits::RealField;
use num_traits::Float;

pub(super) fn analog_lowpass_prototype_zpk<R>(
    order: usize,
    family: AnalogFilterFamily<R>,
) -> Result<ContinuousZpk<R>, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    match family {
        AnalogFilterFamily::Butterworth => butterworth_prototype(order),
        AnalogFilterFamily::Chebyshev1 { ripple_db } => chebyshev1_prototype(order, ripple_db),
        AnalogFilterFamily::Bessel => bessel_prototype(order),
    }
}

fn butterworth_prototype<R>(order: usize) -> Result<ContinuousZpk<R>, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    let n = R::from(order).unwrap();
    let two = R::from(2.0).unwrap();
    let pi = R::from(core::f64::consts::PI).unwrap();
    let mut poles = Vec::with_capacity(order);
    for k in 0..order {
        let k = R::from(k).unwrap();
        let theta = pi * (two * k + n + R::one()) / (two * n);
        poles.push(Complex::new(theta.cos(), theta.sin()));
    }
    let gain = gain_for_dc_target::<R>(&[], &poles, R::one())?;
    ContinuousZpk::new(Vec::new(), poles, gain, ContinuousTime).map_err(Into::into)
}

fn chebyshev1_prototype<R>(
    order: usize,
    ripple_db: R,
) -> Result<ContinuousZpk<R>, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    let n = R::from(order).unwrap();
    let epsilon = (R::from(10.0)
        .unwrap()
        .powf(ripple_db / R::from(10.0).unwrap())
        - R::one())
    .sqrt();
    let mu = ((R::one() / epsilon) + ((R::one() / epsilon).powi(2) + R::one()).sqrt()).ln() / n;
    let pi = R::from(core::f64::consts::PI).unwrap();
    let two = R::from(2.0).unwrap();
    let sinh_mu = mu.sinh();
    let cosh_mu = mu.cosh();
    let mut poles = Vec::with_capacity(order);
    for k in 0..order {
        let theta = pi * (two * R::from(k).unwrap() + R::one()) / (two * n);
        let re = -sinh_mu * theta.sin();
        let im = cosh_mu * theta.cos();
        poles.push(Complex::new(re, im));
    }
    let dc_target = if order % 2 == 0 {
        R::one() / (R::one() + epsilon * epsilon).sqrt()
    } else {
        R::one()
    };
    let gain = gain_for_dc_target::<R>(&[], &poles, dc_target)?;
    ContinuousZpk::new(Vec::new(), poles, gain, ContinuousTime).map_err(Into::into)
}

fn bessel_prototype<R>(order: usize) -> Result<ContinuousZpk<R>, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    let denominator = reverse_bessel_denominator::<R>(order);
    let poles = poly_roots(&denominator)?;
    let gain = gain_for_dc_target::<R>(&[], &poles, R::one())?;
    ContinuousZpk::new(Vec::new(), poles, gain, ContinuousTime).map_err(Into::into)
}

fn reverse_bessel_denominator<R>(order: usize) -> Vec<R>
where
    R: Float + Copy + RealField,
{
    let mut ascending = Vec::with_capacity(order + 1);
    for k in 0..=order {
        let coeff = factorial_f64(2 * order - k)
            / (2.0f64.powi((order - k) as i32) * factorial_f64(k) * factorial_f64(order - k));
        ascending.push(R::from(coeff).unwrap());
    }
    ascending.into_iter().rev().collect()
}

fn factorial_f64(n: usize) -> f64 {
    (1..=n).fold(1.0, |acc, value| acc * value as f64)
}

fn gain_for_dc_target<R>(
    zeros: &[Complex<R>],
    poles: &[Complex<R>],
    target: R,
) -> Result<R, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    let num = zeros
        .iter()
        .fold(Complex::new(R::one(), R::zero()), |acc, &zero| acc * -zero);
    let den = poles
        .iter()
        .fold(Complex::new(R::one(), R::zero()), |acc, &pole| acc * -pole);
    real_scalar(
        Complex::new(target, R::zero()) * den / num,
        "prototype_gain",
    )
}

fn real_scalar<R>(value: Complex<R>, which: &'static str) -> Result<R, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    let scale = R::one().max(value.re.abs()).max(value.im.abs());
    let tol = R::from(128.0).unwrap() * R::epsilon() * scale;
    if value.im.abs() > tol || !value.re.is_finite() {
        Err(FilterDesignError::Lti(LtiError::NonFiniteResult { which }))
    } else {
        Ok(value.re)
    }
}
