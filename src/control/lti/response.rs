use super::analysis::sparse_transfer_at_points;
use super::error::LtiError;
use crate::control::state_space::convert::matrix_exponential;
use crate::control::state_space::{
    ContinuousStateSpace, DiscreteStateSpace, SparseContinuousStateSpace, SparseDiscreteStateSpace,
};
use crate::sparse::compensated::{CompensatedField, CompensatedSum, sum2};
use crate::sparse::matvec::SparseMatVec;
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

/// Dense sampled trajectory of a continuous-time system under a piecewise-
/// constant input.
///
/// `inputs[:, k]` is interpreted as the value held across the interval
/// `[sample_times[k], sample_times[k + 1])`. The final input column is retained
/// so the output at the last sample time can still be evaluated consistently.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousSimulation<R, T> {
    /// Sample times at which the state and output were recorded.
    pub sample_times: Vec<R>,
    /// Input samples aligned with `sample_times`.
    pub inputs: Mat<T>,
    /// State trajectory, one column per sample time.
    pub states: Mat<T>,
    /// Output trajectory, one column per sample time.
    pub outputs: Mat<T>,
}

/// Dense sampled trajectory of a discrete-time system.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteSimulation<T> {
    /// Input sequence, one column per discrete update.
    pub inputs: Mat<T>,
    /// State trajectory. Column `k` stores `x[k]`, so there is one more state
    /// sample than input samples.
    pub states: Mat<T>,
    /// Output sequence. Column `k` stores `y[k]`.
    pub outputs: Mat<T>,
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
        let inputs = Mat::from_fn(self.ninputs(), sample_times.len(), |_, _| T::one());
        let x0 = vec![T::zero(); self.nstates()];
        let sim = self.simulate_zoh(&x0, sample_times, inputs.as_ref())?;
        Ok(SampledResponse {
            sample_points: sim.sample_times,
            values: split_columns(sim.outputs.as_ref()),
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

    /// Simulates the dense continuous-time model on a sampling grid under a
    /// zero-order-held input sequence.
    ///
    /// `inputs[:, k]` is held across the interval
    /// `[sample_times[k], sample_times[k + 1])`. The final input column is used
    /// only to evaluate the output at the final sample time.
    pub fn simulate_zoh(
        &self,
        x0: &[T],
        sample_times: &[T::Real],
        inputs: MatRef<'_, T>,
    ) -> Result<ContinuousSimulation<T::Real, T>, LtiError> {
        validate_state_vector(self.nstates(), x0, "continuous_simulation.x0")?;
        validate_continuous_grid(sample_times, "continuous simulation")?;
        if inputs.nrows() != self.ninputs() || inputs.ncols() != sample_times.len() {
            return Err(LtiError::DimensionMismatch {
                which: "continuous_simulation.inputs",
                expected_nrows: self.ninputs(),
                expected_ncols: sample_times.len(),
                actual_nrows: inputs.nrows(),
                actual_ncols: inputs.ncols(),
            });
        }

        let inputs_owned = clone_mat(inputs);
        let mut states = Mat::<T>::zeros(self.nstates(), sample_times.len());
        let mut outputs = Mat::<T>::zeros(self.noutputs(), sample_times.len());
        let mut state = column_from_slice(x0);
        write_column(states.as_mut(), 0, state.as_ref());

        for k in 0..sample_times.len() {
            let input = column_owned(inputs_owned.as_ref(), k);
            let output = dense_add(
                dense_mul(self.c(), state.as_ref()).as_ref(),
                dense_mul(self.d(), input.as_ref()).as_ref(),
            );
            write_column(outputs.as_mut(), k, output.as_ref());

            if k + 1 < sample_times.len() {
                let dt = sample_times[k + 1] - sample_times[k];
                let (ad, bd) = continuous_interval_maps(self, dt)?;
                state = dense_add(
                    dense_mul(ad.as_ref(), state.as_ref()).as_ref(),
                    dense_mul(bd.as_ref(), input.as_ref()).as_ref(),
                );
                write_column(states.as_mut(), k + 1, state.as_ref());
            }
        }

        Ok(ContinuousSimulation {
            sample_times: sample_times.to_vec(),
            inputs: inputs_owned,
            states,
            outputs,
        })
    }
}

impl<T> SparseContinuousStateSpace<T>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    /// Evaluates the sparse frequency response on the imaginary axis.
    ///
    /// Each frequency `ω` is mapped to `s = jω`, and the returned matrices are
    /// computed through sparse shifted solves rather than a dense transfer
    /// matrix formula.
    pub fn frequency_response(
        &self,
        angular_frequencies: &[T::Real],
    ) -> Result<SampledResponse<T::Real, Complex<T::Real>>, LtiError> {
        validate_finite_grid(angular_frequencies, "sparse continuous frequency response")?;
        let points = angular_frequencies
            .iter()
            .map(|&omega| Complex::new(<T::Real as Zero>::zero(), omega))
            .collect::<Vec<_>>();
        let values = sparse_transfer_at_points(self.a(), self.b(), self.c(), self.d(), &points)?;
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
        let mut inputs = Mat::<T>::zeros(self.ninputs(), n_steps);
        if n_steps > 0 {
            for row in 0..self.ninputs() {
                inputs[(row, 0)] = T::one();
            }
        }
        let x0 = vec![T::zero(); self.nstates()];
        let sim = self
            .simulate(&x0, inputs.as_ref())
            .expect("constructed impulse inputs should be valid");
        SampledResponse {
            sample_points: (0..n_steps).collect(),
            values: split_columns(sim.outputs.as_ref()),
        }
    }

    /// Returns the first `n_steps` samples of the unit-step response.
    ///
    /// Each returned matrix maps a per-input unit step to the output vector at
    /// the corresponding discrete-time index.
    pub fn step_response(&self, n_steps: usize) -> SampledResponse<usize, T> {
        let inputs = Mat::from_fn(self.ninputs(), n_steps, |_, _| T::one());
        let x0 = vec![T::zero(); self.nstates()];
        let sim = self
            .simulate(&x0, inputs.as_ref())
            .expect("constructed step inputs should be valid");
        SampledResponse {
            sample_points: (0..n_steps).collect(),
            values: split_columns(sim.outputs.as_ref()),
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

    /// Simulates the dense discrete-time model for a supplied input sequence.
    ///
    /// The recurrence is
    ///
    /// `x[k+1] = A x[k] + B u[k]`
    ///
    /// `y[k]   = C x[k] + D u[k]`
    ///
    /// so the returned state trajectory has one more column than the input and
    /// output sequences.
    pub fn simulate(
        &self,
        x0: &[T],
        inputs: MatRef<'_, T>,
    ) -> Result<DiscreteSimulation<T>, LtiError> {
        validate_state_vector(self.nstates(), x0, "discrete_simulation.x0")?;
        if inputs.nrows() != self.ninputs() {
            return Err(LtiError::DimensionMismatch {
                which: "discrete_simulation.inputs",
                expected_nrows: self.ninputs(),
                expected_ncols: inputs.ncols(),
                actual_nrows: inputs.nrows(),
                actual_ncols: inputs.ncols(),
            });
        }

        let inputs_owned = clone_mat(inputs);
        let mut states = Mat::<T>::zeros(self.nstates(), inputs.ncols() + 1);
        let mut outputs = Mat::<T>::zeros(self.noutputs(), inputs.ncols());
        let mut state = column_from_slice(x0);
        write_column(states.as_mut(), 0, state.as_ref());

        for k in 0..inputs_owned.ncols() {
            let input = column_owned(inputs_owned.as_ref(), k);

            // Evaluate the output before the state update so the returned
            // columns match the standard discrete-time convention `y[k]`.
            let output = dense_add(
                dense_mul(self.c(), state.as_ref()).as_ref(),
                dense_mul(self.d(), input.as_ref()).as_ref(),
            );
            write_column(outputs.as_mut(), k, output.as_ref());

            state = dense_add(
                dense_mul(self.a(), state.as_ref()).as_ref(),
                dense_mul(self.b(), input.as_ref()).as_ref(),
            );
            write_column(states.as_mut(), k + 1, state.as_ref());
        }

        Ok(DiscreteSimulation {
            inputs: inputs_owned,
            states,
            outputs,
        })
    }
}

impl<T> SparseDiscreteStateSpace<T>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    /// Returns the first `n_steps` samples of the sparse discrete-time impulse
    /// response sequence.
    pub fn impulse_response(&self, n_steps: usize) -> SampledResponse<usize, T> {
        let mut inputs = Mat::<T>::zeros(self.ninputs(), n_steps);
        if n_steps > 0 {
            for row in 0..self.ninputs() {
                inputs[(row, 0)] = T::one();
            }
        }
        let x0 = vec![T::zero(); self.nstates()];
        let sim = self
            .simulate(&x0, inputs.as_ref())
            .expect("constructed sparse impulse inputs should be valid");
        SampledResponse {
            sample_points: (0..n_steps).collect(),
            values: split_columns(sim.outputs.as_ref()),
        }
    }

    /// Returns the first `n_steps` samples of the sparse discrete-time
    /// unit-step response sequence.
    pub fn step_response(&self, n_steps: usize) -> SampledResponse<usize, T> {
        let inputs = Mat::from_fn(self.ninputs(), n_steps, |_, _| T::one());
        let x0 = vec![T::zero(); self.nstates()];
        let sim = self
            .simulate(&x0, inputs.as_ref())
            .expect("constructed sparse step inputs should be valid");
        SampledResponse {
            sample_points: (0..n_steps).collect(),
            values: split_columns(sim.outputs.as_ref()),
        }
    }

    /// Evaluates the sparse discrete-time frequency response on the unit
    /// circle.
    pub fn frequency_response(
        &self,
        angular_frequencies: &[T::Real],
    ) -> Result<SampledResponse<T::Real, Complex<T::Real>>, LtiError> {
        validate_finite_grid(angular_frequencies, "sparse discrete frequency response")?;
        let dt = self.sample_time();
        let points = angular_frequencies
            .iter()
            .map(|&omega| {
                let phase = omega * dt;
                Complex::new(phase.cos(), phase.sin())
            })
            .collect::<Vec<_>>();
        let values = sparse_transfer_at_points(self.a(), self.b(), self.c(), self.d(), &points)?;
        Ok(SampledResponse {
            sample_points: angular_frequencies.to_vec(),
            values,
        })
    }

    /// Simulates the sparse discrete-time model for a supplied input sequence.
    ///
    /// The sparse state matrix is applied with compensated sparse accumulation,
    /// while the dense `B`, `C`, and `D` blocks remain ordinary small dense
    /// products.
    pub fn simulate(
        &self,
        x0: &[T],
        inputs: MatRef<'_, T>,
    ) -> Result<DiscreteSimulation<T>, LtiError> {
        validate_state_vector(self.nstates(), x0, "sparse_discrete_simulation.x0")?;
        if inputs.nrows() != self.ninputs() {
            return Err(LtiError::DimensionMismatch {
                which: "sparse_discrete_simulation.inputs",
                expected_nrows: self.ninputs(),
                expected_ncols: inputs.ncols(),
                actual_nrows: inputs.nrows(),
                actual_ncols: inputs.ncols(),
            });
        }

        let inputs_owned = clone_mat(inputs);
        let mut states = Mat::<T>::zeros(self.nstates(), inputs.ncols() + 1);
        let mut outputs = Mat::<T>::zeros(self.noutputs(), inputs.ncols());
        let mut state = x0.to_vec();
        let mut ax = vec![T::zero(); self.nstates()];
        write_column_from_slice(states.as_mut(), 0, &state);

        for k in 0..inputs_owned.ncols() {
            let input = column_owned(inputs_owned.as_ref(), k);
            let state_col = column_from_slice(&state);

            let output = dense_add(
                dense_mul(self.c(), state_col.as_ref()).as_ref(),
                dense_mul(self.d(), input.as_ref()).as_ref(),
            );
            write_column(outputs.as_mut(), k, output.as_ref());

            self.a().apply_compensated(&mut ax, &state);
            let bu = dense_mul(self.b(), input.as_ref());
            for row in 0..self.nstates() {
                state[row] = sum2(ax[row], bu[(row, 0)]);
            }
            write_column_from_slice(states.as_mut(), k + 1, &state);
        }

        Ok(DiscreteSimulation {
            inputs: inputs_owned,
            states,
            outputs,
        })
    }
}

/// Clones a matrix view into an owned dense matrix.
///
/// The response layer uses this mostly to preserve `B`, `C`, or `D` blocks in
/// returned response objects without keeping borrow ties to the original model.
fn clone_mat<T: Copy>(matrix: MatRef<'_, T>) -> Mat<T> {
    Mat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        matrix[(row, col)]
    })
}

/// Copies a dense column matrix into one column of a larger dense matrix.
fn write_column<T: Copy>(mut dst: faer::MatMut<'_, T>, col: usize, src: MatRef<'_, T>) {
    assert_eq!(src.ncols(), 1);
    assert_eq!(dst.nrows(), src.nrows());
    for row in 0..src.nrows() {
        dst[(row, col)] = src[(row, 0)];
    }
}

/// Copies a raw state slice into one column of a larger dense matrix.
///
/// The sparse simulation path keeps the evolving state in a plain vector for
/// direct sparse matvec application, then uses this helper to expose that
/// vector in the same dense trajectory layout as the dense simulator.
fn write_column_from_slice<T: Copy>(mut dst: faer::MatMut<'_, T>, col: usize, src: &[T]) {
    assert_eq!(dst.nrows(), src.len());
    for (row, &value) in src.iter().enumerate() {
        dst[(row, col)] = value;
    }
}

/// Builds an owned column matrix from a state-vector slice.
fn column_from_slice<T: Copy>(values: &[T]) -> Mat<T> {
    Mat::from_fn(values.len(), 1, |row, _| values[row])
}

/// Extracts one column of a dense matrix into an owned column matrix.
fn column_owned<T: Copy>(matrix: MatRef<'_, T>, col: usize) -> Mat<T> {
    Mat::from_fn(matrix.nrows(), 1, |row, _| matrix[(row, col)])
}

/// Splits a dense matrix into a vector of single-column owned matrices.
///
/// This is mainly used to adapt the more general simulation output shape to the
/// existing `SampledResponse` representation used by impulse/step wrappers.
fn split_columns<T: Copy>(matrix: MatRef<'_, T>) -> Vec<Mat<T>> {
    (0..matrix.ncols())
        .map(|col| column_owned(matrix, col))
        .collect()
}

/// Dense elementwise matrix addition for the response layer.
///
/// This stays local rather than reusing a broader utility because the current
/// response code only needs a tiny subset of dense matrix arithmetic.
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

/// Dense matrix product with compensated accumulation.
///
/// The response routines build small dense intermediates from `A`, `B`, and
/// `C`. Keeping these products compensated avoids losing the accuracy policy
/// established elsewhere in the control module just because the matrices are
/// small.
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

/// Validates the expected state-vector length for simulation entry points.
fn validate_state_vector<T>(nstates: usize, x0: &[T], which: &'static str) -> Result<(), LtiError> {
    if x0.len() == nstates {
        Ok(())
    } else {
        Err(LtiError::DimensionMismatch {
            which,
            expected_nrows: nstates,
            expected_ncols: 1,
            actual_nrows: x0.len(),
            actual_ncols: 1,
        })
    }
}

/// Checks that a time-domain sampling grid is finite and nonnegative.
///
/// Negative time samples are not meaningful for the causal LTI response APIs
/// exposed here, so they are rejected at the wrapper boundary.
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

/// Checks that a continuous-time simulation grid is finite, nonnegative, and
/// nondecreasing.
///
/// Zero-width intervals are allowed so callers can sample the same instant more
/// than once, but backward time steps are rejected.
fn validate_continuous_grid<R: Float + Copy>(
    sample_points: &[R],
    which: &'static str,
) -> Result<(), LtiError> {
    if sample_points.is_empty() {
        return Err(LtiError::InvalidSampleGrid { which });
    }
    validate_nonnegative_grid(sample_points, which)?;
    for window in sample_points.windows(2) {
        if window[1] < window[0] {
            return Err(LtiError::InvalidSampleGrid { which });
        }
    }
    Ok(())
}

/// Checks that a frequency grid contains only finite values.
///
/// Negative frequencies are allowed because callers may want symmetric grids,
/// but `NaN` or infinite values would produce meaningless transfer evaluations.
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

/// Computes the exact dense ZOH state-transition maps over one time interval.
///
/// This is the same lifted-exponential construction used by the public
/// continuous-time discretization path, just specialized to a single interval
/// so the simulator can propagate one ZOH segment at a time.
fn continuous_interval_maps<T>(
    system: &ContinuousStateSpace<T>,
    dt: T::Real,
) -> Result<(Mat<T>, Mat<T>), LtiError>
where
    T: CompensatedField,
    T::Real: Float + Copy + RealField,
{
    let n = system.nstates();
    let m = system.ninputs();
    let size = n + m;
    let mut lifted = Mat::<T>::zeros(size, size);
    for row in 0..n {
        for col in 0..n {
            lifted[(row, col)] = system.a()[(row, col)].mul_real(dt);
        }
    }
    for row in 0..n {
        for col in 0..m {
            lifted[(row, n + col)] = system.b()[(row, col)].mul_real(dt);
        }
    }
    let exp_lifted = matrix_exponential(lifted.as_ref())?;
    let ad = Mat::from_fn(n, n, |row, col| exp_lifted[(row, col)]);
    let bd = Mat::from_fn(n, m, |row, col| exp_lifted[(row, n + col)]);
    Ok((ad, bd))
}

#[cfg(test)]
mod tests {
    use super::{ContinuousImpulseResponse, SampledResponse};
    use crate::control::state_space::{
        ContinuousStateSpace, DiscreteStateSpace, SparseContinuousStateSpace,
        SparseDiscreteStateSpace,
    };
    use faer::complex::Complex;
    use faer::sparse::{SparseColMat, Triplet};
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
    fn discrete_simulation_matches_manual_recurrence() {
        let sys = DiscreteStateSpace::new(
            Mat::from_fn(1, 1, |_, _| 0.5),
            Mat::from_fn(1, 1, |_, _| 2.0),
            Mat::from_fn(1, 1, |_, _| 3.0),
            Mat::from_fn(1, 1, |_, _| 4.0),
            0.1,
        )
        .unwrap();

        let inputs = Mat::from_fn(1, 2, |_, col| if col == 0 { 7.0 } else { 11.0 });
        let sim = sys.simulate(&[5.0], inputs.as_ref()).unwrap();

        assert_close_real(
            sim.states.as_ref(),
            Mat::from_fn(1, 3, |_, col| match col {
                0 => 5.0,
                1 => 16.5,
                2 => 30.25,
                _ => unreachable!(),
            })
            .as_ref(),
            1.0e-12,
        );
        assert_close_real(
            sim.outputs.as_ref(),
            Mat::from_fn(1, 2, |_, col| if col == 0 { 43.0 } else { 93.5 }).as_ref(),
            1.0e-12,
        );
    }

    #[test]
    fn continuous_zoh_simulation_matches_closed_form_scalar_case() {
        let sys = ContinuousStateSpace::new(
            Mat::from_fn(1, 1, |_, _| -1.0),
            Mat::from_fn(1, 1, |_, _| 2.0),
            Mat::from_fn(1, 1, |_, _| 3.0),
            Mat::from_fn(1, 1, |_, _| 4.0),
        )
        .unwrap();

        let sample_times = [0.0, 1.0, 2.0];
        let inputs = Mat::from_fn(1, 3, |_, col| match col {
            0 => 7.0,
            1 => 11.0,
            _ => 13.0,
        });
        let sim = sys
            .simulate_zoh(&[5.0], &sample_times, inputs.as_ref())
            .unwrap();

        let x1 = (-1.0f64).exp() * 5.0 + 2.0 * (1.0 - (-1.0f64).exp()) * 7.0;
        let x2 = (-1.0f64).exp() * x1 + 2.0 * (1.0 - (-1.0f64).exp()) * 11.0;
        assert_close_real(
            sim.states.as_ref(),
            Mat::from_fn(1, 3, |_, col| match col {
                0 => 5.0,
                1 => x1,
                2 => x2,
                _ => unreachable!(),
            })
            .as_ref(),
            1.0e-12,
        );
        assert_close_real(
            sim.outputs.as_ref(),
            Mat::from_fn(1, 3, |_, col| match col {
                0 => 3.0 * 5.0 + 4.0 * 7.0,
                1 => 3.0 * x1 + 4.0 * 11.0,
                2 => 3.0 * x2 + 4.0 * 13.0,
                _ => unreachable!(),
            })
            .as_ref(),
            1.0e-12,
        );
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

    #[test]
    fn sparse_discrete_simulation_matches_dense_reference() {
        let dense = DiscreteStateSpace::new(
            Mat::from_fn(2, 2, |row, col| match (row, col) {
                (0, 0) => 0.5,
                (0, 1) => 0.25,
                (1, 1) => 0.75,
                _ => 0.0,
            }),
            Mat::from_fn(2, 1, |row, _| if row == 0 { 2.0 } else { -1.0 }),
            Mat::from_fn(1, 2, |_, col| if col == 0 { 3.0 } else { -2.0 }),
            Mat::from_fn(1, 1, |_, _| 4.0),
            0.1,
        )
        .unwrap();
        let sparse_a = SparseColMat::<usize, f64>::try_new_from_triplets(
            2,
            2,
            &[
                Triplet::new(0, 0, 0.5),
                Triplet::new(0, 1, 0.25),
                Triplet::new(1, 1, 0.75),
            ],
        )
        .unwrap();
        let sparse = SparseDiscreteStateSpace::new(
            sparse_a,
            Mat::from_fn(2, 1, |row, _| if row == 0 { 2.0 } else { -1.0 }),
            Mat::from_fn(1, 2, |_, col| if col == 0 { 3.0 } else { -2.0 }),
            Mat::from_fn(1, 1, |_, _| 4.0),
            0.1,
        )
        .unwrap();

        let inputs = Mat::from_fn(1, 3, |_, col| match col {
            0 => 1.0,
            1 => -2.0,
            _ => 0.5,
        });
        let x0 = [0.25, -0.5];
        let dense_sim = dense.simulate(&x0, inputs.as_ref()).unwrap();
        let sparse_sim = sparse.simulate(&x0, inputs.as_ref()).unwrap();
        assert_close_real(
            sparse_sim.states.as_ref(),
            dense_sim.states.as_ref(),
            1.0e-12,
        );
        assert_close_real(
            sparse_sim.outputs.as_ref(),
            dense_sim.outputs.as_ref(),
            1.0e-12,
        );
    }

    #[test]
    fn sparse_discrete_step_response_matches_dense_reference() {
        let dense = DiscreteStateSpace::new(
            Mat::from_fn(1, 1, |_, _| 0.5),
            Mat::from_fn(1, 1, |_, _| 2.0),
            Mat::from_fn(1, 1, |_, _| 3.0),
            Mat::from_fn(1, 1, |_, _| 4.0),
            0.1,
        )
        .unwrap();
        let sparse_a =
            SparseColMat::<usize, f64>::try_new_from_triplets(1, 1, &[Triplet::new(0, 0, 0.5)])
                .unwrap();
        let sparse = SparseDiscreteStateSpace::new(
            sparse_a,
            Mat::from_fn(1, 1, |_, _| 2.0),
            Mat::from_fn(1, 1, |_, _| 3.0),
            Mat::from_fn(1, 1, |_, _| 4.0),
            0.1,
        )
        .unwrap();

        let dense_resp = dense.step_response(4);
        let sparse_resp = sparse.step_response(4);
        for (sparse_value, dense_value) in sparse_resp.values.iter().zip(dense_resp.values.iter()) {
            assert_close_real(sparse_value.as_ref(), dense_value.as_ref(), 1.0e-12);
        }
    }

    #[test]
    fn sparse_frequency_response_matches_dense_reference_at_zero_frequency() {
        let dense_cont = ContinuousStateSpace::new(
            Mat::from_fn(1, 1, |_, _| -2.0),
            Mat::from_fn(1, 1, |_, _| 3.0),
            Mat::from_fn(1, 1, |_, _| 4.0),
            Mat::from_fn(1, 1, |_, _| 5.0),
        )
        .unwrap();
        let sparse_cont = SparseContinuousStateSpace::new(
            SparseColMat::<usize, f64>::try_new_from_triplets(1, 1, &[Triplet::new(0, 0, -2.0)])
                .unwrap(),
            Mat::from_fn(1, 1, |_, _| 3.0),
            Mat::from_fn(1, 1, |_, _| 4.0),
            Mat::from_fn(1, 1, |_, _| 5.0),
        )
        .unwrap();
        let dense_disc = DiscreteStateSpace::new(
            Mat::from_fn(1, 1, |_, _| 0.25),
            Mat::from_fn(1, 1, |_, _| 3.0),
            Mat::from_fn(1, 1, |_, _| 4.0),
            Mat::from_fn(1, 1, |_, _| 5.0),
            0.1,
        )
        .unwrap();
        let sparse_disc = SparseDiscreteStateSpace::new(
            SparseColMat::<usize, f64>::try_new_from_triplets(1, 1, &[Triplet::new(0, 0, 0.25)])
                .unwrap(),
            Mat::from_fn(1, 1, |_, _| 3.0),
            Mat::from_fn(1, 1, |_, _| 4.0),
            Mat::from_fn(1, 1, |_, _| 5.0),
            0.1,
        )
        .unwrap();

        let dense_cont_resp = dense_cont.frequency_response(&[0.0]).unwrap();
        let sparse_cont_resp = sparse_cont.frequency_response(&[0.0]).unwrap();
        assert_close_complex(
            sparse_cont_resp.values[0].as_ref(),
            dense_cont_resp.values[0].as_ref(),
            1.0e-12,
        );

        let dense_disc_resp = dense_disc.frequency_response(&[0.0]).unwrap();
        let sparse_disc_resp = sparse_disc.frequency_response(&[0.0]).unwrap();
        assert_close_complex(
            sparse_disc_resp.values[0].as_ref(),
            dense_disc_resp.values[0].as_ref(),
            1.0e-12,
        );
    }
}
