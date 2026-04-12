use super::error::FilterDesignError;
use super::spec::FilterShape;
use crate::control::lti::{ContinuousTime, ContinuousZpk, DiscreteTime, DiscreteZpk};
use faer::complex::Complex;
use faer_traits::RealField;
use num_traits::Float;

pub(super) fn analog_shape_transform<R>(
    prototype: &ContinuousZpk<R>,
    shape: FilterShape<R>,
) -> Result<ContinuousZpk<R>, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    match shape {
        FilterShape::Lowpass { cutoff } => lowpass_zpk(prototype, cutoff),
        FilterShape::Highpass { cutoff } => highpass_zpk(prototype, cutoff),
        FilterShape::Bandpass {
            low_cutoff,
            high_cutoff,
        } => bandpass_zpk(prototype, low_cutoff, high_cutoff),
        FilterShape::Bandstop {
            low_cutoff,
            high_cutoff,
        } => bandstop_zpk(prototype, low_cutoff, high_cutoff),
    }
}

pub(super) fn bilinear_transform_zpk<R>(
    analog: &ContinuousZpk<R>,
    sample_rate: R,
) -> Result<DiscreteZpk<R>, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    let fs2 = sample_rate + sample_rate;
    let fs2_c = Complex::new(fs2, R::zero());
    let z_degree = analog.poles().len().saturating_sub(analog.zeros().len());
    let zeros = analog
        .zeros()
        .iter()
        .map(|&zero| (fs2_c + zero) / (fs2_c - zero))
        .chain((0..z_degree).map(|_| Complex::new(-R::one(), R::zero())))
        .collect::<Vec<_>>();
    let poles = analog
        .poles()
        .iter()
        .map(|&pole| (fs2_c + pole) / (fs2_c - pole))
        .collect::<Vec<_>>();

    let numerator = analog
        .zeros()
        .iter()
        .fold(Complex::new(R::one(), R::zero()), |acc, &zero| {
            acc * (fs2_c - zero)
        });
    let denominator = analog
        .poles()
        .iter()
        .fold(Complex::new(R::one(), R::zero()), |acc, &pole| {
            acc * (fs2_c - pole)
        });
    let gain = real_scalar(Complex::new(analog.gain(), R::zero()) * numerator / denominator)?;

    DiscreteZpk::new(zeros, poles, gain, DiscreteTime::new(sample_rate)).map_err(Into::into)
}

fn lowpass_zpk<R>(
    prototype: &ContinuousZpk<R>,
    cutoff: R,
) -> Result<ContinuousZpk<R>, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    let cutoff_c = Complex::new(cutoff, R::zero());
    let zeros = prototype
        .zeros()
        .iter()
        .map(|&zero| zero * cutoff_c)
        .collect::<Vec<_>>();
    let poles = prototype
        .poles()
        .iter()
        .map(|&pole| pole * cutoff_c)
        .collect::<Vec<_>>();
    let degree = prototype
        .poles()
        .len()
        .saturating_sub(prototype.zeros().len());
    let gain = prototype.gain() * cutoff.powi(degree as i32);
    ContinuousZpk::new(zeros, poles, gain, ContinuousTime).map_err(Into::into)
}

fn highpass_zpk<R>(
    prototype: &ContinuousZpk<R>,
    cutoff: R,
) -> Result<ContinuousZpk<R>, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    let wc = Complex::new(cutoff, R::zero());
    let degree = prototype
        .poles()
        .len()
        .saturating_sub(prototype.zeros().len());
    let zeros = prototype
        .zeros()
        .iter()
        .map(|&zero| wc / zero)
        .chain((0..degree).map(|_| Complex::new(R::zero(), R::zero())))
        .collect::<Vec<_>>();
    let poles = prototype
        .poles()
        .iter()
        .map(|&pole| wc / pole)
        .collect::<Vec<_>>();
    let numerator = prototype
        .zeros()
        .iter()
        .fold(Complex::new(R::one(), R::zero()), |acc, &zero| acc * -zero);
    let denominator = prototype
        .poles()
        .iter()
        .fold(Complex::new(R::one(), R::zero()), |acc, &pole| acc * -pole);
    let gain = real_scalar(Complex::new(prototype.gain(), R::zero()) * numerator / denominator)?;
    ContinuousZpk::new(zeros, poles, gain, ContinuousTime).map_err(Into::into)
}

fn bandpass_zpk<R>(
    prototype: &ContinuousZpk<R>,
    low_cutoff: R,
    high_cutoff: R,
) -> Result<ContinuousZpk<R>, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    let bw = high_cutoff - low_cutoff;
    let w0 = (low_cutoff * high_cutoff).sqrt();
    let half_bw = bw / (R::one() + R::one());
    let half_bw_c = Complex::new(half_bw, R::zero());
    let degree = prototype
        .poles()
        .len()
        .saturating_sub(prototype.zeros().len());
    let mut zeros = Vec::with_capacity(prototype.zeros().len() * 2 + degree);
    for &zero in prototype.zeros() {
        let root = zero * half_bw_c;
        let radical = (root * root - Complex::new(w0 * w0, R::zero())).sqrt();
        zeros.push(root + radical);
        zeros.push(root - radical);
    }
    zeros.extend((0..degree).map(|_| Complex::new(R::zero(), R::zero())));

    let mut poles = Vec::with_capacity(prototype.poles().len() * 2);
    for &pole in prototype.poles() {
        let root = pole * half_bw_c;
        let radical = (root * root - Complex::new(w0 * w0, R::zero())).sqrt();
        poles.push(root + radical);
        poles.push(root - radical);
    }

    let gain = prototype.gain() * bw.powi(degree as i32);
    ContinuousZpk::new(zeros, poles, gain, ContinuousTime).map_err(Into::into)
}

fn bandstop_zpk<R>(
    prototype: &ContinuousZpk<R>,
    low_cutoff: R,
    high_cutoff: R,
) -> Result<ContinuousZpk<R>, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    let bw = high_cutoff - low_cutoff;
    let w0 = (low_cutoff * high_cutoff).sqrt();
    let half_bw = bw / (R::one() + R::one());
    let degree = prototype
        .poles()
        .len()
        .saturating_sub(prototype.zeros().len());
    let j_w0 = Complex::new(R::zero(), w0);

    let mut zeros = Vec::with_capacity(prototype.zeros().len() * 2 + degree * 2);
    for &zero in prototype.zeros() {
        let root = Complex::new(half_bw, R::zero()) / zero;
        let radical = (root * root - Complex::new(w0 * w0, R::zero())).sqrt();
        zeros.push(root + radical);
        zeros.push(root - radical);
    }
    zeros.extend((0..degree).flat_map(|_| [j_w0, -j_w0]));

    let mut poles = Vec::with_capacity(prototype.poles().len() * 2);
    for &pole in prototype.poles() {
        let root = Complex::new(half_bw, R::zero()) / pole;
        let radical = (root * root - Complex::new(w0 * w0, R::zero())).sqrt();
        poles.push(root + radical);
        poles.push(root - radical);
    }

    let numerator = prototype
        .zeros()
        .iter()
        .fold(Complex::new(R::one(), R::zero()), |acc, &zero| acc * -zero);
    let denominator = prototype
        .poles()
        .iter()
        .fold(Complex::new(R::one(), R::zero()), |acc, &pole| acc * -pole);
    let gain = real_scalar(Complex::new(prototype.gain(), R::zero()) * numerator / denominator)?;
    ContinuousZpk::new(zeros, poles, gain, ContinuousTime).map_err(Into::into)
}

fn real_scalar<R>(value: Complex<R>) -> Result<R, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    let scale = R::one().max(value.re.abs()).max(value.im.abs());
    let tol = R::from(128.0).unwrap() * R::epsilon() * scale;
    if value.im.abs() > tol || !value.re.is_finite() {
        Err(FilterDesignError::Lti(
            crate::control::lti::LtiError::NonFiniteResult {
                which: "filter_transform_gain",
            },
        ))
    } else {
        Ok(value.re)
    }
}
