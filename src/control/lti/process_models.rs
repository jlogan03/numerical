//! Continuous delay-aware process models.
//!
//! These models keep transport delay explicit instead of forcing it through the
//! rational transfer-function layer. That makes them a better fit for
//! process-control workflows and for PID tuning rules that reason directly in
//! terms of dead time.
//!
//! They are still linear and time-invariant, but they are not finite-order
//! rational transfer functions because of the explicit `exp(-Ls)` delay term.
//!
//! # Two Intuitions
//!
//! 1. **Process-control view.** These models capture the dominant gain, lag,
//!    and delay that matter for everyday tuning and process interpretation.
//! 2. **Delay-honest view.** They also serve as a reminder that pure delay is
//!    linear but not a finite-order rational transfer function, so it deserves
//!    an explicit model surface instead of being hidden in a polynomial proxy.
//!
//! # Glossary
//!
//! - **FOPDT:** First-order-plus-dead-time model.
//! - **SOPDT:** Second-order-plus-dead-time model.
//! - **Dead time / transport delay:** Pure delay factor `exp(-L s)`.
//!
//! # Mathematical Formulation
//!
//! The two model families exposed here are:
//!
//! - `FOPDT: K exp(-L s) / (tau s + 1)`
//! - `SOPDT: K exp(-L s) / ((tau_1 s + 1)(tau_2 s + 1))`
//!
//! Frequency response evaluates the delay exactly, and time-domain step
//! response helpers use the corresponding delayed closed forms.
//!
//! # Implementation Notes
//!
//! - The models are continuous-time only in this layer.
//! - Delay is preserved explicitly in analysis helpers rather than converted to
//!   a rational approximation.
//! - These public types are shared with the PID-design layer so fitted process
//!   models and hand-specified ones follow the same contract.

use super::{BodeData, LtiError, NicholsData, NyquistData, util::unwrap_phase_deg};
use crate::scalar::{mul_add, neg_mul_add, real_complex_mul_add};
use faer::complex::Complex;
use faer_traits::RealField;
use num_traits::Float;

/// First-order-plus-dead-time process model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FopdtModel<R> {
    /// Steady-state process gain.
    pub gain: R,
    /// Dominant first-order lag.
    pub time_constant: R,
    /// Input-output dead time.
    pub delay: R,
}

impl<R> FopdtModel<R>
where
    R: Float + Copy + RealField,
{
    /// Evaluates the continuous-time transfer map
    /// `K * exp(-Ls) / (tau s + 1)`.
    #[must_use]
    pub fn evaluate(&self, s: Complex<R>) -> Complex<R> {
        // Keep the transport delay exact in frequency-domain evaluation rather
        // than approximating it through a rational surrogate.
        let delay_factor = (-s * self.delay).exp();
        Complex::new(self.gain, R::zero()) * delay_factor
            / real_complex_mul_add(self.time_constant, s, Complex::new(R::one(), R::zero()))
    }

    /// Returns the steady-state gain.
    #[must_use]
    pub fn dc_gain(&self) -> R {
        self.gain
    }

    /// Evaluates the delayed unit-step response for a step of amplitude
    /// `step_amplitude` and baseline `initial_output`.
    #[must_use]
    pub fn step_response_value(
        &self,
        time_since_step: R,
        step_amplitude: R,
        initial_output: R,
    ) -> R {
        if time_since_step <= self.delay {
            initial_output
        } else {
            let theta = time_since_step - self.delay;
            mul_add(
                self.gain * step_amplitude,
                R::one() - (-(theta / self.time_constant)).exp(),
                initial_output,
            )
        }
    }

    /// Samples the delayed step response on the supplied time grid.
    pub fn step_response_values(
        &self,
        sample_times: &[R],
        step_time: R,
        step_amplitude: R,
        initial_output: R,
    ) -> Result<Vec<R>, LtiError> {
        validate_step_grid(sample_times)?;
        Ok(sample_times
            .iter()
            .map(|&time| {
                // The public step-response helper keeps the same "delay is a
                // pure dead zone" convention as the fitting and PID-tuning
                // path.
                self.step_response_value(
                    (time - step_time).max(R::zero()),
                    step_amplitude,
                    initial_output,
                )
            })
            .collect())
    }

    /// Samples Nyquist data on a continuous-time angular-frequency grid.
    pub fn nyquist_data(&self, angular_frequencies: &[R]) -> Result<NyquistData<R>, LtiError> {
        validate_frequency_grid(angular_frequencies, "nyquist_data")?;
        Ok(NyquistData {
            angular_frequencies: angular_frequencies.to_vec(),
            values: angular_frequencies
                .iter()
                .map(|&omega| self.evaluate(Complex::new(R::zero(), omega)))
                .collect(),
        })
    }

    /// Samples Nichols data on a continuous-time angular-frequency grid.
    pub fn nichols_data(&self, angular_frequencies: &[R]) -> Result<NicholsData<R>, LtiError> {
        let bode = self.bode_data(angular_frequencies)?;
        Ok(NicholsData {
            angular_frequencies: bode.angular_frequencies,
            magnitude_db: bode.magnitude_db,
            phase_deg: bode.phase_deg,
        })
    }

    /// Samples Bode magnitude and unwrapped phase on a continuous-time
    /// angular-frequency grid.
    pub fn bode_data(&self, angular_frequencies: &[R]) -> Result<BodeData<R>, LtiError> {
        validate_frequency_grid(angular_frequencies, "bode_data")?;
        let values = angular_frequencies
            .iter()
            .map(|&omega| self.evaluate(Complex::new(R::zero(), omega)))
            .collect::<Vec<_>>();
        let magnitude_db = values
            .iter()
            .map(|value| R::from(20.0).unwrap() * value.norm().log10())
            .collect::<Vec<_>>();
        let phase_deg = unwrap_phase_deg(
            &values
                .iter()
                .map(|value| {
                    value.im.atan2(value.re) * R::from(180.0 / core::f64::consts::PI).unwrap()
                })
                .collect::<Vec<_>>(),
        );
        Ok(BodeData {
            angular_frequencies: angular_frequencies.to_vec(),
            magnitude_db,
            phase_deg,
        })
    }
}

/// Second-order-plus-dead-time process model with two real lags.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SopdtModel<R> {
    /// Steady-state process gain.
    pub gain: R,
    /// Slower first-order lag.
    pub time_constant_1: R,
    /// Faster first-order lag.
    pub time_constant_2: R,
    /// Input-output dead time.
    pub delay: R,
}

impl<R> SopdtModel<R>
where
    R: Float + Copy + RealField,
{
    /// Evaluates the continuous-time transfer map
    /// `K * exp(-Ls) / ((tau1 s + 1)(tau2 s + 1))`.
    #[must_use]
    pub fn evaluate(&self, s: Complex<R>) -> Complex<R> {
        // As in the FOPDT path, preserve the explicit transport delay exactly
        // in frequency-domain evaluation.
        let delay_factor = (-s * self.delay).exp();
        let d1 = real_complex_mul_add(self.time_constant_1, s, Complex::new(R::one(), R::zero()));
        let d2 = real_complex_mul_add(self.time_constant_2, s, Complex::new(R::one(), R::zero()));
        Complex::new(self.gain, R::zero()) * delay_factor / (d1 * d2)
    }

    /// Returns the steady-state gain.
    #[must_use]
    pub fn dc_gain(&self) -> R {
        self.gain
    }

    /// Evaluates the delayed second-order step response for a step of
    /// amplitude `step_amplitude` and baseline `initial_output`.
    #[must_use]
    pub fn step_response_value(
        &self,
        time_since_step: R,
        step_amplitude: R,
        initial_output: R,
    ) -> R {
        if time_since_step <= self.delay {
            return initial_output;
        }
        let theta = time_since_step - self.delay;
        let lag = second_order_lag_step(theta, self.time_constant_1, self.time_constant_2);
        mul_add(self.gain * step_amplitude, lag, initial_output)
    }

    /// Samples the delayed step response on the supplied time grid.
    pub fn step_response_values(
        &self,
        sample_times: &[R],
        step_time: R,
        step_amplitude: R,
        initial_output: R,
    ) -> Result<Vec<R>, LtiError> {
        validate_step_grid(sample_times)?;
        Ok(sample_times
            .iter()
            .map(|&time| {
                self.step_response_value(
                    (time - step_time).max(R::zero()),
                    step_amplitude,
                    initial_output,
                )
            })
            .collect())
    }

    /// Samples Nyquist data on a continuous-time angular-frequency grid.
    pub fn nyquist_data(&self, angular_frequencies: &[R]) -> Result<NyquistData<R>, LtiError> {
        validate_frequency_grid(angular_frequencies, "nyquist_data")?;
        Ok(NyquistData {
            angular_frequencies: angular_frequencies.to_vec(),
            values: angular_frequencies
                .iter()
                .map(|&omega| self.evaluate(Complex::new(R::zero(), omega)))
                .collect(),
        })
    }

    /// Samples Nichols data on a continuous-time angular-frequency grid.
    pub fn nichols_data(&self, angular_frequencies: &[R]) -> Result<NicholsData<R>, LtiError> {
        let bode = self.bode_data(angular_frequencies)?;
        Ok(NicholsData {
            angular_frequencies: bode.angular_frequencies,
            magnitude_db: bode.magnitude_db,
            phase_deg: bode.phase_deg,
        })
    }

    /// Samples Bode magnitude and unwrapped phase on a continuous-time
    /// angular-frequency grid.
    pub fn bode_data(&self, angular_frequencies: &[R]) -> Result<BodeData<R>, LtiError> {
        validate_frequency_grid(angular_frequencies, "bode_data")?;
        let values = angular_frequencies
            .iter()
            .map(|&omega| self.evaluate(Complex::new(R::zero(), omega)))
            .collect::<Vec<_>>();
        let magnitude_db = values
            .iter()
            .map(|value| R::from(20.0).unwrap() * value.norm().log10())
            .collect::<Vec<_>>();
        let phase_deg = unwrap_phase_deg(
            &values
                .iter()
                .map(|value| {
                    value.im.atan2(value.re) * R::from(180.0 / core::f64::consts::PI).unwrap()
                })
                .collect::<Vec<_>>(),
        );
        Ok(BodeData {
            angular_frequencies: angular_frequencies.to_vec(),
            magnitude_db,
            phase_deg,
        })
    }
}

fn validate_step_grid<R>(sample_times: &[R]) -> Result<(), LtiError>
where
    R: Float + Copy + RealField,
{
    if sample_times.is_empty()
        || sample_times
            .windows(2)
            .any(|window| !window[0].is_finite() || !window[1].is_finite() || window[1] < window[0])
        || sample_times.iter().any(|&time| !time.is_finite())
    {
        return Err(LtiError::InvalidSampleGrid {
            which: "step_response_values",
        });
    }
    Ok(())
}

fn validate_frequency_grid<R>(
    angular_frequencies: &[R],
    which: &'static str,
) -> Result<(), LtiError>
where
    R: Float + Copy + RealField,
{
    if angular_frequencies.is_empty()
        || angular_frequencies
            .iter()
            .any(|&omega| !omega.is_finite() || omega < R::zero())
        || angular_frequencies
            .windows(2)
            .any(|window| window[1] < window[0])
    {
        return Err(LtiError::InvalidSampleGrid { which });
    }
    Ok(())
}

fn second_order_lag_step<R>(time: R, tau1: R, tau2: R) -> R
where
    R: Float + Copy + RealField,
{
    // Near a repeated pole, the distinct-lag formula loses accuracy through
    // subtraction. Switch to the repeated-pole limit in that regime.
    let tol = (tau1.abs() + tau2.abs() + R::one()) * R::from(1.0e-8).unwrap();
    if (tau1 - tau2).abs() <= tol {
        let tau = (tau1 + tau2) / R::from(2.0).unwrap();
        let scaled = time / tau;
        neg_mul_add(R::one() + scaled, (-scaled).exp(), R::one())
    } else {
        let e1 = (-(time / tau1)).exp();
        let e2 = (-(time / tau2)).exp();
        R::one() - (tau1 * e1 - tau2 * e2) / (tau1 - tau2)
    }
}

#[cfg(test)]
mod tests {
    use super::{FopdtModel, SopdtModel};

    fn assert_close(lhs: f64, rhs: f64, tol: f64) {
        let err = (lhs - rhs).abs();
        assert!(err <= tol, "lhs={lhs}, rhs={rhs}, err={err}, tol={tol}");
    }

    #[test]
    fn fopdt_has_expected_dc_gain_and_delay_behavior() {
        let model = FopdtModel {
            gain: 2.0,
            time_constant: 3.0,
            delay: 1.5,
        };
        assert_close(model.dc_gain(), 2.0, 1.0e-12);
        assert_close(model.step_response_value(1.0, 1.0, 0.25), 0.25, 1.0e-12);
        let late = model.step_response_value(10.0, 1.0, 0.25);
        assert!(late > 2.1);
        assert!(late < 2.25);
    }

    #[test]
    fn sopdt_has_expected_dc_gain_and_monotone_step_value() {
        let model = SopdtModel {
            gain: 1.5,
            time_constant_1: 4.0,
            time_constant_2: 1.0,
            delay: 0.5,
        };
        assert_close(model.dc_gain(), 1.5, 1.0e-12);
        assert_close(model.step_response_value(0.25, 1.0, 0.0), 0.0, 1.0e-12);
        let mid = model.step_response_value(3.0, 1.0, 0.0);
        let late = model.step_response_value(20.0, 1.0, 0.0);
        assert!(late > mid);
        assert!(late < 1.5 + 1.0e-4);
    }
}
