//! Analog and digital IIR filter design.
//!
//! The first implementation is intentionally practical and numerically
//! conservative:
//!
//! - prototype generation and frequency transformations stay in `Zpk` form
//! - `Sos` is the preferred practical digital output form
//! - raw polynomial `TransferFunction` output is treated as an interoperability
//!   view rather than the main working representation
//!
//! Supported in the first pass:
//!
//! - Butterworth
//! - Chebyshev Type I
//! - Bessel, analog only
//! - lowpass / highpass / bandpass / bandstop
//!
//! Digital Bessel design is intentionally not exposed in the digital family
//! enum. A bilinear-transformed Bessel filter is mathematically possible, but
//! the current API omits it so users are not misled into expecting classical
//! Bessel group-delay behavior from the resulting digital filter.

mod digital;
mod error;
mod prototype;
mod spec;
mod transform;

pub use error::FilterDesignError;
pub use spec::{
    AnalogFilterFamily, AnalogFilterSpec, DigitalFilterFamily, DigitalFilterSpec, FilterShape,
};

use digital::maybe_prewarp_shape;
use prototype::analog_lowpass_prototype_zpk;
use transform::{analog_shape_transform, bilinear_transform_zpk};

use super::{
    ContinuousSos, ContinuousTransferFunction, ContinuousZpk, DiscreteSos,
    DiscreteTransferFunction, DiscreteZpk,
};
use faer_traits::RealField;
use num_traits::Float;

/// Designs an analog IIR filter and returns it in `Zpk` form.
///
/// This is the numerically preferred internal representation because prototype
/// generation and frequency transformations operate directly on roots instead
/// of high-order polynomials.
pub fn design_analog_filter_zpk<R>(
    spec: &AnalogFilterSpec<R>,
) -> Result<ContinuousZpk<R>, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    let prototype = analog_lowpass_prototype_zpk(spec.order, spec.family)?;
    analog_shape_transform(&prototype, spec.shape)
}

/// Designs an analog IIR filter and returns it in `Sos` form.
///
/// `Sos` is usually the best practical representation for realized high-order
/// filters because it localizes rounding error section by section.
pub fn design_analog_filter_sos<R>(
    spec: &AnalogFilterSpec<R>,
) -> Result<ContinuousSos<R>, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    Ok(design_analog_filter_zpk(spec)?.to_sos()?)
}

/// Designs an analog IIR filter and returns it in coefficient form.
///
/// This is mainly an interoperability view. High-order direct-form
/// coefficients are more numerically fragile than `Zpk` or `Sos`.
pub fn design_analog_filter_tf<R>(
    spec: &AnalogFilterSpec<R>,
) -> Result<ContinuousTransferFunction<R>, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    Ok(design_analog_filter_zpk(spec)?.to_transfer_function()?)
}

/// Designs a digital IIR filter and returns it in `Zpk` form.
///
/// The implementation builds an analog prototype, applies the requested analog
/// frequency transformation, optionally prewarps the cutoff frequencies, then
/// maps the result through the bilinear transform.
pub fn design_digital_filter_zpk<R>(
    spec: &DigitalFilterSpec<R>,
) -> Result<DiscreteZpk<R>, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    let analog_family = match spec.family {
        DigitalFilterFamily::Butterworth => AnalogFilterFamily::Butterworth,
        DigitalFilterFamily::Chebyshev1 { ripple_db } => {
            AnalogFilterFamily::Chebyshev1 { ripple_db }
        }
    };
    let analog_spec = AnalogFilterSpec::new(
        spec.order,
        analog_family,
        maybe_prewarp_shape(spec.shape, spec.sample_rate, spec.prewarp),
    )?;
    let analog = design_analog_filter_zpk(&analog_spec)?;
    bilinear_transform_zpk(&analog, spec.sample_rate)
}

/// Designs a digital IIR filter and returns it in `Sos` form.
///
/// This is the recommended output form for practical digital filtering.
pub fn design_digital_filter_sos<R>(
    spec: &DigitalFilterSpec<R>,
) -> Result<DiscreteSos<R>, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    Ok(design_digital_filter_zpk(spec)?.to_sos()?)
}

/// Designs a digital IIR filter and returns it in coefficient form.
///
/// This is kept for interoperability and inspection. For numerically sensitive
/// high-order filters, prefer `design_digital_filter_sos`.
pub fn design_digital_filter_tf<R>(
    spec: &DigitalFilterSpec<R>,
) -> Result<DiscreteTransferFunction<R>, FilterDesignError>
where
    R: Float + Copy + RealField,
{
    Ok(design_digital_filter_zpk(spec)?.to_transfer_function()?)
}

#[cfg(test)]
mod tests {
    use super::{
        AnalogFilterFamily, AnalogFilterSpec, DigitalFilterFamily, DigitalFilterSpec, FilterShape,
        design_analog_filter_sos, design_analog_filter_tf, design_analog_filter_zpk,
        design_digital_filter_tf, design_digital_filter_zpk,
    };
    use faer::complex::Complex;

    fn assert_close(lhs: f64, rhs: f64, tol: f64) {
        let err = (lhs - rhs).abs();
        assert!(err <= tol, "lhs={lhs}, rhs={rhs}, err={err}, tol={tol}");
    }

    #[test]
    fn analog_butterworth_lowpass_has_unity_dc_gain_and_lhp_poles() {
        let spec = AnalogFilterSpec::new(
            3,
            AnalogFilterFamily::Butterworth,
            FilterShape::Lowpass { cutoff: 2.0 },
        )
        .unwrap();
        let zpk = design_analog_filter_zpk(&spec).unwrap();
        assert_close(zpk.evaluate(Complex::new(0.0, 0.0)).norm(), 1.0, 1.0e-10);
        assert!(zpk.poles().iter().all(|pole| pole.re < 0.0));
    }

    #[test]
    fn analog_chebyshev_bandstop_rejects_center_frequency() {
        let spec = AnalogFilterSpec::new(
            2,
            AnalogFilterFamily::Chebyshev1 { ripple_db: 1.0 },
            FilterShape::Bandstop {
                low_cutoff: 8.0,
                high_cutoff: 12.0,
            },
        )
        .unwrap();
        let tf = design_analog_filter_tf(&spec).unwrap();
        let dc = tf.evaluate(Complex::new(0.0, 0.0)).norm();
        let center = tf
            .evaluate(Complex::new(0.0, (8.0f64 * 12.0f64).sqrt()))
            .norm();
        assert!(center < dc);
    }

    #[test]
    fn analog_bessel_designs_and_converts_consistently() {
        let spec = AnalogFilterSpec::new(
            3,
            AnalogFilterFamily::Bessel,
            FilterShape::Highpass { cutoff: 4.0 },
        )
        .unwrap();
        let zpk = design_analog_filter_zpk(&spec).unwrap();
        let sos = design_analog_filter_sos(&spec).unwrap();
        let tf = design_analog_filter_tf(&spec).unwrap();
        let point = Complex::new(0.0, 20.0);
        let lhs = zpk.evaluate(point);
        let rhs = sos.evaluate(point);
        let ref_value = tf.evaluate(point);
        assert!((lhs - rhs).norm() <= 1.0e-8);
        assert!((lhs - ref_value).norm() <= 1.0e-8);
    }

    #[test]
    fn digital_butterworth_lowpass_is_stable_and_lowpass_like() {
        let spec = DigitalFilterSpec::new(
            4,
            DigitalFilterFamily::Butterworth,
            FilterShape::Lowpass { cutoff: 10.0 },
            100.0,
        )
        .unwrap();
        let zpk = design_digital_filter_zpk(&spec).unwrap();
        assert!(zpk.poles().iter().all(|pole| pole.norm() < 1.0));
        let dc = zpk.evaluate(Complex::new(1.0, 0.0)).norm();
        let nyquist = zpk.evaluate(Complex::new(-1.0, 0.0)).norm();
        assert!(dc > nyquist);
    }

    #[test]
    fn digital_bandpass_hits_center_more_than_dc() {
        let spec = DigitalFilterSpec::new(
            2,
            DigitalFilterFamily::Chebyshev1 { ripple_db: 1.0 },
            FilterShape::Bandpass {
                low_cutoff: 15.0,
                high_cutoff: 25.0,
            },
            100.0,
        )
        .unwrap();
        let tf = design_digital_filter_tf(&spec).unwrap();
        let dc = tf.evaluate(Complex::new(1.0, 0.0)).norm();
        let center_omega = (15.0f64 * 25.0f64).sqrt();
        let phase = center_omega / 100.0;
        let center = tf.evaluate(Complex::new(phase.cos(), phase.sin())).norm();
        assert!(center > dc);
    }
}
