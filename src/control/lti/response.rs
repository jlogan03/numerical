use super::error::LtiError;
use crate::control::state_space::convert::matrix_exponential;
use crate::control::state_space::{ContinuousStateSpace, DiscreteStateSpace};
use crate::sparse::compensated::{CompensatedField, CompensatedSum};
use faer::complex::Complex;
use faer::{Mat, MatRef};
use faer_traits::ext::ComplexFieldExt;
use faer_traits::{ComplexField, RealField};
use num_traits::{Float, Zero};

/// Sampled response values evaluated on a one-dimensional grid.
///
/// For time-domain responses, `sample_points` contains times or discrete step
/// indices. For frequency-domain responses, it contains angular frequencies.
///
/// Each entry in `values` is a full output-by-input matrix evaluated at the
/// corresponding sample point, so MIMO systems retain the same block shape as
/// the original `C`/`D` maps.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledResponse<X, T> {
    /// Grid points at which the response matrices were evaluated.
    pub sample_points: Vec<X>,
    /// Response matrices at each grid point.
    pub values: Vec<Mat<T>>,
}

/// Continuous-time impulse response data.
///
/// Continuous LTI systems with nonzero `D` have an impulsive feedthrough term
/// `D δ(t)`. The `values` field stores only the regular part `C exp(A t) B`;
/// the singular feedthrough is returned separately in `direct_feedthrough`.
///
/// This keeps the API honest about the distribution-valued part of the
/// response instead of silently mixing it into the sampled regular component.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousImpulseResponse<R, T> {
    /// Sample times where the regular impulse response was evaluated.
    pub sample_times: Vec<R>,
    /// The direct-feedthrough impulse coefficient `D`.
    pub direct_feedthrough: Mat<T>,
    /// Regular impulse response samples `C exp(A t) B`.
    pub values: Vec<Mat<T>>,
}

impl<T> ContinuousStateSpace<T>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    /// Evaluates the regular part of the impulse response at the supplied
    /// nonnegative sample times.
    ///
    /// For `x' = A x + B u`, `y = C x + D u`, this returns
    ///
    /// `h_reg(t) = C exp(A t) B`
    ///
    /// together with the separate impulsive coefficient `D`.
    pub fn impulse_response(
        &self,
        sample_times: &[T::Real],
    ) -> Result<ContinuousImpulseResponse<T::Real, T>, LtiError> {
        validate_nonnegative_grid(sample_times, "continuous impulse response")?;
        let mut values = Vec::with_capacity(sample_times.len());
        for &time in sample_times {
            // For each sample time, evaluate the state-transition map exactly
            // through the dense matrix exponential and then project it through
            // `C` and `B` to obtain the regular impulse kernel.
            let scaled_a = Mat::from_fn(self.nstates(), self.nstates(), |row, col| {
                self.a()[(row, col)].mul_real(time)
            });
            let exp_at = matrix_exponential(scaled_a.as_ref())?;
            let value = dense_mul(self.c(), dense_mul(exp_at.as_ref(), self.b()).as_ref());
            values.push(value);
        }
        Ok(ContinuousImpulseResponse {
            sample_times: sample_times.to_vec(),
            direct_feedthrough: clone_mat(self.d()),
            values,
        })
    }

    /// Evaluates the unit-step response at the supplied nonnegative sample
    /// times.
    ///
    /// Each returned matrix maps an input-channel unit step to the output
    /// vector at that time. The exact dense formula is obtained from the same
    /// lifted exponential used by zero-order-hold discretization.
    pub fn step_response(
        &self,
        sample_times: &[T::Real],
    ) -> Result<SampledResponse<T::Real, T>, LtiError> {
        validate_nonnegative_grid(sample_times, "continuous step response")?;
        let n = self.nstates();
        let m = self.ninputs();
        let size = n + m;
        let mut values = Vec::with_capacity(sample_times.len());

        for &time in sample_times {
            let mut lifted = Mat::<T>::zeros(size, size);
            for row in 0..n {
                for col in 0..n {
                    lifted[(row, col)] = self.a()[(row, col)].mul_real(time);
                }
            }
            for row in 0..n {
                for col in 0..m {
                    lifted[(row, n + col)] = self.b()[(row, col)].mul_real(time);
                }
            }

            // This is the same lifted exponential used for exact ZOH
            // discretization:
            //
            // exp(t * [A B; 0 0]) = [exp(A t)   integral_0^t exp(A τ) B dτ
            //                       0          I                         ]
            //
            // The upper-right block is exactly the step-response state map.
            let exp_lifted = matrix_exponential(lifted.as_ref())?;
            let bd = Mat::from_fn(n, m, |row, col| exp_lifted[(row, n + col)]);
            let value = dense_add(dense_mul(self.c(), bd.as_ref()).as_ref(), self.d());
            values.push(value);
        }

        Ok(SampledResponse {
            sample_points: sample_times.to_vec(),
            values,
        })
    }

    /// Evaluates the frequency response on the imaginary axis.
    ///
    /// Each frequency `ω` is mapped to `s = jω`, and the returned matrices are
    /// `G(jω)`.
    pub fn frequency_response(
        &self,
        angular_frequencies: &[T::Real],
    ) -> Result<SampledResponse<T::Real, Complex<T::Real>>, LtiError> {
        validate_finite_grid(angular_frequencies, "continuous frequency response")?;
        let mut values = Vec::with_capacity(angular_frequencies.len());
        for &omega in angular_frequencies {
            // Frequency response is just transfer-function evaluation on the
            // imaginary axis; the dense solve is handled by `transfer_at`.
            values.push(self.transfer_at(Complex::new(<T::Real as Zero>::zero(), omega))?);
        }
        Ok(SampledResponse {
            sample_points: angular_frequencies.to_vec(),
            values,
        })
    }
}

impl<T> DiscreteStateSpace<T>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    /// Returns the first `n_steps` samples of the impulse response sequence.
    ///
    /// The returned matrices are indexed by step:
    ///
    /// - `k = 0`: `D`
    /// - `k >= 1`: `C A^(k-1) B`
    pub fn impulse_response(&self, n_steps: usize) -> SampledResponse<usize, T> {
        let mut values = Vec::with_capacity(n_steps);
        let mut ab = clone_mat(self.b());

        for step in 0..n_steps {
            let value = if step == 0 {
                clone_mat(self.d())
            } else {
                // After the direct-feedthrough sample, the discrete impulse
                // sequence follows the simple recurrence `C A^(k-1) B`.
                let out = dense_mul(self.c(), ab.as_ref());
                ab = dense_mul(self.a(), ab.as_ref());
                out
            };
            values.push(value);
        }

        SampledResponse {
            sample_points: (0..n_steps).collect(),
            values,
        }
    }

    /// Returns the first `n_steps` samples of the unit-step response.
    ///
    /// Each returned matrix maps a per-input unit step to the output vector at
    /// the corresponding discrete-time index.
    pub fn step_response(&self, n_steps: usize) -> SampledResponse<usize, T> {
        let mut values = Vec::with_capacity(n_steps);
        let mut state = Mat::<T>::zeros(self.nstates(), self.ninputs());

        for _step in 0..n_steps {
            // `state` stores the accumulated state response to a unit step up
            // to the current sample. Advancing it is just the usual discrete
            // affine recurrence `x_{k+1} = A x_k + B`.
            let value = dense_add(dense_mul(self.c(), state.as_ref()).as_ref(), self.d());
            values.push(value);
            state = dense_add(dense_mul(self.a(), state.as_ref()).as_ref(), self.b());
        }

        SampledResponse {
            sample_points: (0..n_steps).collect(),
            values,
        }
    }

    /// Evaluates the frequency response on the discrete-time unit circle.
    ///
    /// Each physical angular frequency `ω` is mapped to
    ///
    /// `z = exp(j ω dt)`
    ///
    /// so the returned matrices are `G(exp(j ω dt))`.
    pub fn frequency_response(
        &self,
        angular_frequencies: &[T::Real],
    ) -> Result<SampledResponse<T::Real, Complex<T::Real>>, LtiError> {
        validate_finite_grid(angular_frequencies, "discrete frequency response")?;
        let dt = self.sample_time();
        let mut values = Vec::with_capacity(angular_frequencies.len());
        for &omega in angular_frequencies {
            // Map physical angular frequency onto the sampled unit circle
            // before reusing the generic transfer-function evaluation.
            let phase = omega * dt;
            let point = Complex::new(phase.cos(), phase.sin());
            values.push(self.transfer_at(point)?);
        }
        Ok(SampledResponse {
            sample_points: angular_frequencies.to_vec(),
            values,
        })
    }
}

fn clone_mat<T: Copy>(matrix: MatRef<'_, T>) -> Mat<T> {
    Mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        matrix[(row, col)]
    })
}

fn dense_add<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: ComplexField + Copy,
{
    assert_eq!(lhs.nrows(), rhs.nrows());
    assert_eq!(lhs.ncols(), rhs.ncols());
    Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| {
        lhs[(row, col)] + rhs[(row, col)]
    })
}

fn dense_mul<T>(lhs: MatRef<'_, T>, rhs: MatRef<'_, T>) -> Mat<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    assert_eq!(lhs.ncols(), rhs.nrows());
    Mat::from_fn(lhs.nrows(), rhs.ncols(), |row, col| {
        // Keep even these small dense response-building products compensated so
        // the LTI analysis layer follows the same accumulation policy as the
        // rest of the control module.
        let mut acc = CompensatedSum::<T>::default();
        for k in 0..lhs.ncols() {
            acc.add(lhs[(row, k)] * rhs[(k, col)]);
        }
        acc.finish()
    })
}

fn validate_nonnegative_grid<R: Float + Copy>(
    sample_points: &[R],
    which: &'static str,
) -> Result<(), LtiError> {
    for &sample in sample_points {
        if !sample.is_finite() || sample < R::zero() {
            return Err(LtiError::InvalidSamplePoint { which });
        }
    }
    Ok(())
}

fn validate_finite_grid<R: Float + Copy>(
    sample_points: &[R],
    which: &'static str,
) -> Result<(), LtiError> {
    for &sample in sample_points {
        if !sample.is_finite() {
            return Err(LtiError::InvalidSamplePoint { which });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ContinuousImpulseResponse, SampledResponse};
    use crate::control::state_space::{ContinuousStateSpace, DiscreteStateSpace};
    use faer::complex::Complex;
    use faer::{Mat, MatRef};

    fn assert_close_real(lhs: MatRef<'_, f64>, rhs: MatRef<'_, f64>, tol: f64) {
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

    fn assert_close_complex(
        lhs: MatRef<'_, Complex<f64>>,
        rhs: MatRef<'_, Complex<f64>>,
        tol: f64,
    ) {
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

    #[test]
    fn continuous_impulse_response_matches_first_order_closed_form() {
        let sys = ContinuousStateSpace::new(
            Mat::from_fn(1, 1, |_, _| -2.0),
            Mat::from_fn(1, 1, |_, _| 3.0),
            Mat::from_fn(1, 1, |_, _| 4.0),
            Mat::from_fn(1, 1, |_, _| 5.0),
        )
        .unwrap();

        let response: ContinuousImpulseResponse<f64, f64> =
            sys.impulse_response(&[0.0, 1.0]).unwrap();
        assert_close_real(
            response.direct_feedthrough.as_ref(),
            Mat::from_fn(1, 1, |_, _| 5.0).as_ref(),
            1.0e-12,
        );
        assert_close_real(
            response.values[0].as_ref(),
            Mat::from_fn(1, 1, |_, _| 12.0).as_ref(),
            1.0e-12,
        );
        assert_close_real(
            response.values[1].as_ref(),
            Mat::from_fn(1, 1, |_, _| 12.0 * (-2.0f64).exp()).as_ref(),
            1.0e-12,
        );
    }

    #[test]
    fn continuous_step_response_matches_first_order_closed_form() {
        let sys = ContinuousStateSpace::new(
            Mat::from_fn(1, 1, |_, _| -2.0),
            Mat::from_fn(1, 1, |_, _| 3.0),
            Mat::from_fn(1, 1, |_, _| 4.0),
            Mat::from_fn(1, 1, |_, _| 5.0),
        )
        .unwrap();

        let response: SampledResponse<f64, f64> = sys.step_response(&[0.0, 1.0]).unwrap();
        assert_close_real(
            response.values[0].as_ref(),
            Mat::from_fn(1, 1, |_, _| 5.0).as_ref(),
            1.0e-12,
        );
        assert_close_real(
            response.values[1].as_ref(),
            Mat::from_fn(1, 1, |_, _| 5.0 + 6.0 * (1.0 - (-2.0f64).exp())).as_ref(),
            1.0e-12,
        );
    }

    #[test]
    fn discrete_impulse_and_step_responses_match_recurrence() {
        let sys = DiscreteStateSpace::new(
            Mat::from_fn(1, 1, |_, _| 0.5),
            Mat::from_fn(1, 1, |_, _| 2.0),
            Mat::from_fn(1, 1, |_, _| 3.0),
            Mat::from_fn(1, 1, |_, _| 4.0),
            0.1,
        )
        .unwrap();

        let impulse = sys.impulse_response(4);
        let expected_impulse = [4.0, 6.0, 3.0, 1.5];
        for (matrix, expected) in impulse.values.iter().zip(expected_impulse) {
            assert_close_real(
                matrix.as_ref(),
                Mat::from_fn(1, 1, |_, _| expected).as_ref(),
                1.0e-12,
            );
        }

        let step = sys.step_response(4);
        let expected_step = [4.0, 10.0, 13.0, 14.5];
        for (matrix, expected) in step.values.iter().zip(expected_step) {
            assert_close_real(
                matrix.as_ref(),
                Mat::from_fn(1, 1, |_, _| expected).as_ref(),
                1.0e-12,
            );
        }
    }

    #[test]
    fn frequency_response_matches_dc_gain_at_zero_frequency() {
        let cont = ContinuousStateSpace::new(
            Mat::from_fn(1, 1, |_, _| -2.0),
            Mat::from_fn(1, 1, |_, _| 3.0),
            Mat::from_fn(1, 1, |_, _| 4.0),
            Mat::from_fn(1, 1, |_, _| 5.0),
        )
        .unwrap();
        let disc = DiscreteStateSpace::new(
            Mat::from_fn(1, 1, |_, _| 0.25),
            Mat::from_fn(1, 1, |_, _| 3.0),
            Mat::from_fn(1, 1, |_, _| 4.0),
            Mat::from_fn(1, 1, |_, _| 5.0),
            0.1,
        )
        .unwrap();

        let cont_resp = cont.frequency_response(&[0.0]).unwrap();
        let disc_resp = disc.frequency_response(&[0.0]).unwrap();
        assert_close_complex(
            cont_resp.values[0].as_ref(),
            cont.dc_gain().unwrap().as_ref(),
            1.0e-12,
        );
        assert_close_complex(
            disc_resp.values[0].as_ref(),
            disc.dc_gain().unwrap().as_ref(),
            1.0e-12,
        );
    }
}
