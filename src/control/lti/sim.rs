//! Fixed-timestep digital filter simulation helpers.
//!
//! This module sits alongside the broader response layer and provides
//! filter-shaped APIs for running discrete SISO filters over sampled data.
//! The execution kernels are intentionally limited to numerically sensible
//! representations:
//!
//! - dense discrete state-space
//! - discrete SOS
//!
//! Callers should convert `TransferFunction` or `Zpk` into one of those
//! representations first instead of running high-order coefficient recurrences
//! directly.
//!
//! # Two Intuitions
//!
//! 1. **DSP view.** This module is the runtime filtering surface for sampled
//!    scalar signals, including both ordinary causal filtering and
//!    forward-backward zero-phase filtering.
//! 2. **Execution-kernel view.** It is also a deliberate restriction of the
//!    public API to the two forms that are numerically credible for IIR
//!    execution: explicit state-space recurrences and SOS cascades.
//!
//! # Glossary
//!
//! - **`filtfilt`:** Forward-backward filtering for zero-phase response.
//! - **Odd/even reflection:** Common endpoint-padding conventions.
//! - **DF2T:** Direct-form II transposed section recurrence.
//!
//! # Mathematical Formulation
//!
//! The module implements:
//!
//! - state-space recurrence `x[k+1] = A x[k] + B u[k]`, `y[k] = C x[k] + D u[k]`
//! - sectionwise IIR recurrence for SOS cascades
//! - forward-backward application `y = reverse(F(reverse(F(x))))`
//!   with configurable endpoint padding
//!
//! # Implementation Notes
//!
//! - Padding is sampled logically instead of allocating a full padded copy of
//!   the input by default.
//! - Padding length is auto-shortened on short signals to avoid surprising
//!   error paths.
//! - `TransferFunction` and `Zpk` are intentionally excluded from direct
//!   runtime simulation to avoid endorsing unstable high-order coefficient
//!   recurrences.

use super::{DiscreteSos, DiscreteStateSpace, LtiError};
use crate::sparse::compensated::{CompensatedField, CompensatedSum, sum2, sum3};
use faer_traits::ComplexField;
use faer_traits::RealField;
use num_traits::Float;

/// Filtered scalar output sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct FilteredSignal<R> {
    /// One output sample per input sample.
    pub output: Vec<R>,
}

/// Filtered scalar output sequence plus retained final runtime state.
#[derive(Clone, Debug, PartialEq)]
pub struct StatefulFilteredSignal<R, S> {
    /// One output sample per input sample.
    pub output: Vec<R>,
    /// Final runtime state after processing the full input slice.
    pub final_state: S,
}

/// Padding mode for forward-backward zero-phase filtering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiltFiltPadMode {
    /// No endpoint padding.
    None,
    /// Odd reflection around the endpoints.
    OddReflection,
    /// Even reflection around the endpoints.
    EvenReflection,
    /// Constant extension using the endpoint values.
    Constant,
}

/// Padding-length policy for forward-backward filtering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiltFiltPadLen {
    /// Use the representation-specific default.
    Auto,
    /// Use the requested padding length, then clamp it to what the input
    /// length and selected padding mode can support.
    Exact(usize),
}

/// Configurable padding policy for forward-backward filtering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FiltFiltParams {
    /// Endpoint-padding mode.
    pub mode: FiltFiltPadMode,
    /// Padding-length policy.
    pub len: FiltFiltPadLen,
}

impl Default for FiltFiltParams {
    fn default() -> Self {
        // Odd reflection is the practical default for IIR `filtfilt` because
        // it suppresses endpoint transients better than constant extension or
        // no padding, without growing the public API too much.
        Self {
            mode: FiltFiltPadMode::OddReflection,
            len: FiltFiltPadLen::Auto,
        }
    }
}

impl FiltFiltParams {
    /// Creates the default odd-reflection / auto-length policy.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Overrides the endpoint-padding mode.
    #[must_use]
    pub fn with_mode(mut self, mode: FiltFiltPadMode) -> Self {
        self.mode = mode;
        self
    }

    /// Overrides the padding-length policy.
    #[must_use]
    pub fn with_len(mut self, len: FiltFiltPadLen) -> Self {
        self.len = len;
        self
    }
}

/// Runtime state for a cascade of SOS sections.
#[derive(Clone, Debug, PartialEq)]
pub struct SosFilterState<R> {
    /// Per-section direct-form II transposed state `[s1, s2]`.
    pub section_state: Vec<[R; 2]>,
}

impl<R> SosFilterState<R>
where
    R: Float + Copy,
{
    /// Creates a zero-initialized state for a given number of sections.
    #[must_use]
    pub fn zeros(n_sections: usize) -> Self {
        Self {
            section_state: vec![[R::zero(), R::zero()]; n_sections],
        }
    }
}

impl<R> DiscreteSos<R>
where
    R: Float + Copy + RealField + CompensatedField,
{
    /// Filters one input slice causally with zero initial section state.
    ///
    /// This is the simple one-shot path. Use
    /// [`filter_forward_stateful`](Self::filter_forward_stateful) when chunked
    /// processing needs to preserve section delay state across calls.
    pub fn filter_forward(&self, input: &[R]) -> Result<FilteredSignal<R>, LtiError> {
        let mut state = SosFilterState::zeros(self.sections().len());
        self.filter_forward_stateful(&mut state, input)
    }

    /// Filters one input slice causally while updating a caller-supplied SOS
    /// runtime state.
    ///
    /// The runtime form is direct-form II transposed per section. That keeps
    /// the state compact and matches the numerically sensible execution model
    /// for high-order IIR filters represented as SOS cascades.
    pub fn filter_forward_stateful(
        &self,
        state: &mut SosFilterState<R>,
        input: &[R],
    ) -> Result<FilteredSignal<R>, LtiError> {
        validate_sos_state_len(self, state)?;
        let mut output = Vec::with_capacity(input.len());
        for &sample in input {
            output.push(sos_step(self, &mut state.section_state, sample));
        }
        Ok(FilteredSignal { output })
    }

    /// Runs forward-backward zero-phase filtering with the default padding
    /// policy.
    pub fn filtfilt(&self, input: &[R]) -> Result<FilteredSignal<R>, LtiError> {
        self.filtfilt_with_params(input, &FiltFiltParams::default())
    }

    /// Runs forward-backward zero-phase filtering with explicit padding
    /// control.
    ///
    /// This implementation deliberately uses zero initial state on both
    /// passes and relies on endpoint padding to reduce startup transients. It
    /// also shortens the requested padding automatically when the input is too
    /// short to support the nominal length.
    pub fn filtfilt_with_params(
        &self,
        input: &[R],
        params: &FiltFiltParams,
    ) -> Result<FilteredSignal<R>, LtiError> {
        if input.is_empty() {
            return Ok(FilteredSignal { output: Vec::new() });
        }

        let pad_len = resolve_pad_len(input.len(), params, 6 * self.sections().len());
        let total_len = input.len() + 2 * pad_len;

        let mut first_pass = Vec::with_capacity(total_len);
        let mut state = SosFilterState::zeros(self.sections().len());
        for idx in 0..total_len {
            // Materialize only one padded sample at a time instead of building
            // a fully padded copy of the input signal.
            let sample = padded_sample(input, params.mode, pad_len, idx);
            first_pass.push(sos_step(self, &mut state.section_state, sample));
        }

        first_pass.reverse();

        let mut second_pass = Vec::with_capacity(total_len);
        let mut state = SosFilterState::zeros(self.sections().len());
        for &sample in &first_pass {
            second_pass.push(sos_step(self, &mut state.section_state, sample));
        }

        second_pass.reverse();
        Ok(FilteredSignal {
            output: second_pass[pad_len..(pad_len + input.len())].to_vec(),
        })
    }
}

impl<R> SosFilterState<R>
where
    R: Float + Copy + RealField + CompensatedField,
{
    /// Creates a zero-initialized state sized for a particular filter.
    ///
    /// This avoids forcing callers to reach into the filter to size the state
    /// vector manually.
    #[must_use]
    pub fn for_filter(filter: &DiscreteSos<R>) -> Self {
        Self::zeros(filter.sections().len())
    }
}

impl<R> DiscreteStateSpace<R>
where
    R: Float + Copy + RealField + CompensatedField,
{
    /// Filters one input slice causally with zero initial state.
    ///
    /// The returned `final_state` is useful for manual chunked workflows even
    /// though this module does not expose a dedicated streaming wrapper.
    pub fn filter_forward(
        &self,
        input: &[R],
    ) -> Result<StatefulFilteredSignal<R, Vec<R>>, LtiError> {
        let x0 = vec![R::zero(); self.nstates()];
        self.filter_forward_with_state(&x0, input)
    }

    /// Filters one input slice causally from a caller-supplied initial state.
    ///
    /// This path intentionally reuses the same state-update law as the broader
    /// discrete simulation layer, but returns only the SISO output sequence and
    /// final state instead of a full trajectory matrix.
    pub fn filter_forward_with_state(
        &self,
        x0: &[R],
        input: &[R],
    ) -> Result<StatefulFilteredSignal<R, Vec<R>>, LtiError> {
        ensure_siso_state_space(self)?;
        validate_initial_state_len(self, x0)?;

        let mut state = x0.to_vec();
        let output = run_state_space_filter(self, &mut state, input);
        Ok(StatefulFilteredSignal {
            output,
            final_state: state,
        })
    }

    /// Runs forward-backward zero-phase filtering with the default padding
    /// policy.
    pub fn filtfilt(&self, input: &[R]) -> Result<FilteredSignal<R>, LtiError> {
        self.filtfilt_with_params(input, &FiltFiltParams::default())
    }

    /// Runs forward-backward zero-phase filtering with explicit padding
    /// control.
    ///
    /// As in the SOS path, padding is shortened automatically on short input
    /// records instead of producing a dedicated "input too short" error.
    pub fn filtfilt_with_params(
        &self,
        input: &[R],
        params: &FiltFiltParams,
    ) -> Result<FilteredSignal<R>, LtiError> {
        ensure_siso_state_space(self)?;
        if input.is_empty() {
            return Ok(FilteredSignal { output: Vec::new() });
        }

        let pad_len = resolve_pad_len(input.len(), params, 3 * self.nstates());
        let total_len = input.len() + 2 * pad_len;

        let mut state = vec![R::zero(); self.nstates()];
        let mut first_pass =
            run_state_space_filter_with_generator(self, &mut state, total_len, |idx| {
                padded_sample(input, params.mode, pad_len, idx)
            });

        first_pass.reverse();

        state.fill(R::zero());
        let mut second_pass =
            run_state_space_filter_with_generator(self, &mut state, total_len, |idx| {
                first_pass[idx]
            });

        second_pass.reverse();
        Ok(FilteredSignal {
            output: second_pass[pad_len..(pad_len + input.len())].to_vec(),
        })
    }
}

fn ensure_siso_state_space<R>(system: &DiscreteStateSpace<R>) -> Result<(), LtiError>
where
    R: ComplexField,
{
    if system.is_siso() {
        Ok(())
    } else {
        Err(LtiError::NonSisoStateSpace {
            ninputs: system.ninputs(),
            noutputs: system.noutputs(),
        })
    }
}

fn validate_initial_state_len<R>(system: &DiscreteStateSpace<R>, x0: &[R]) -> Result<(), LtiError>
where
    R: ComplexField,
{
    if x0.len() == system.nstates() {
        Ok(())
    } else {
        Err(LtiError::DimensionMismatch {
            which: "discrete_filter_forward.x0",
            expected_nrows: system.nstates(),
            expected_ncols: 1,
            actual_nrows: x0.len(),
            actual_ncols: 1,
        })
    }
}

fn validate_sos_state_len<R>(
    system: &DiscreteSos<R>,
    state: &SosFilterState<R>,
) -> Result<(), LtiError>
where
    R: Float + Copy + RealField,
{
    if state.section_state.len() == system.sections().len() {
        Ok(())
    } else {
        Err(LtiError::InvalidFilterStateLength {
            which: "sos_filter_state",
            expected: system.sections().len(),
            actual: state.section_state.len(),
        })
    }
}

pub(crate) fn resolve_pad_len(input_len: usize, params: &FiltFiltParams, auto_len: usize) -> usize {
    // The public API exposes `Auto` or `Exact`, but the runtime always works
    // with a concrete effective padding length after representation-specific
    // defaulting and input-length clamping.
    let requested = match params.len {
        FiltFiltPadLen::Auto => auto_len,
        FiltFiltPadLen::Exact(value) => value,
    };
    clamp_pad_len(input_len, params.mode, requested)
}

pub(crate) fn clamp_pad_len(input_len: usize, mode: FiltFiltPadMode, requested: usize) -> usize {
    // Reflection needs one interior sample beyond the endpoint, so it can use
    // at most `input.len() - 1`. Constant padding can extend by the full input
    // length because it does not dereference interior reflected samples.
    let max_len = match mode {
        FiltFiltPadMode::None => 0,
        FiltFiltPadMode::OddReflection | FiltFiltPadMode::EvenReflection => {
            input_len.saturating_sub(1)
        }
        FiltFiltPadMode::Constant => input_len,
    };
    requested.min(max_len)
}

pub(crate) fn padded_sample<R>(input: &[R], mode: FiltFiltPadMode, pad_len: usize, idx: usize) -> R
where
    R: Float + Copy,
{
    debug_assert!(idx < input.len() + 2 * pad_len);
    if idx < pad_len {
        let reflected = pad_len - idx;
        match mode {
            FiltFiltPadMode::None => input[idx],
            // Odd reflection preserves endpoint value and approximately
            // preserves local slope, which is why it is the default.
            FiltFiltPadMode::OddReflection => input[0] + (input[0] - input[reflected]),
            FiltFiltPadMode::EvenReflection => input[reflected],
            FiltFiltPadMode::Constant => input[0],
        }
    } else if idx < pad_len + input.len() {
        input[idx - pad_len]
    } else {
        let reflected = idx - (pad_len + input.len());
        let last = input.len() - 1;
        match mode {
            FiltFiltPadMode::None => input[last],
            FiltFiltPadMode::OddReflection => {
                input[last] + (input[last] - input[last - 1 - reflected])
            }
            FiltFiltPadMode::EvenReflection => input[last - 1 - reflected],
            FiltFiltPadMode::Constant => input[last],
        }
    }
}

fn run_state_space_filter<R>(system: &DiscreteStateSpace<R>, state: &mut [R], input: &[R]) -> Vec<R>
where
    R: Float + Copy + RealField + CompensatedField,
{
    run_state_space_filter_with_generator(system, state, input.len(), |idx| input[idx])
}

fn run_state_space_filter_with_generator<R, F>(
    system: &DiscreteStateSpace<R>,
    state: &mut [R],
    len: usize,
    mut sample_at: F,
) -> Vec<R>
where
    R: Float + Copy + RealField + CompensatedField,
    F: FnMut(usize) -> R,
{
    // The generator-based form lets the `filtfilt` path feed logically padded
    // samples without allocating a second full input buffer.
    let mut output = Vec::with_capacity(len);
    let mut next_state = vec![R::zero(); system.nstates()];
    for idx in 0..len {
        let sample = sample_at(idx);
        output.push(state_space_step(system, state, &mut next_state, sample));
    }
    output
}

fn state_space_step<R>(
    system: &DiscreteStateSpace<R>,
    state: &mut [R],
    next_state: &mut [R],
    input: R,
) -> R
where
    R: Float + Copy + RealField + CompensatedField,
{
    // Keep the state-space runtime aligned with the general discrete
    // simulation semantics: output is evaluated from the current state before
    // the state update is applied.
    let mut output_acc = CompensatedSum::<R>::default();
    for col in 0..system.nstates() {
        output_acc.add(system.c()[(0, col)] * state[col]);
    }
    output_acc.add(system.d()[(0, 0)] * input);
    let output = output_acc.finish();

    for row in 0..system.nstates() {
        let mut acc = CompensatedSum::<R>::default();
        for col in 0..system.nstates() {
            acc.add(system.a()[(row, col)] * state[col]);
        }
        acc.add(system.b()[(row, 0)] * input);
        next_state[row] = acc.finish();
    }
    state.copy_from_slice(next_state);
    output
}

fn sos_step<R>(system: &DiscreteSos<R>, section_state: &mut [[R; 2]], input: R) -> R
where
    R: Float + Copy + RealField + CompensatedField,
{
    let mut sample = input * system.gain();
    for (section, state) in system.sections().iter().zip(section_state.iter_mut()) {
        let [b0, b1, b2] = section.numerator();
        let [_one, a1, a2] = section.denominator();
        // Direct-form II transposed:
        //
        // y[n]   = b0 x[n] + s1[n-1]
        // s1[n]  = b1 x[n] - a1 y[n] + s2[n-1]
        // s2[n]  = b2 x[n] - a2 y[n]
        //
        // The section output becomes the next section input.
        let y = sum2(b0 * sample, state[0]);
        let next_s1 = sum3(b1 * sample, -(a1 * y), state[1]);
        let next_s2 = b2 * sample - a2 * y;
        state[0] = next_s1;
        state[1] = next_s2;
        sample = y;
    }
    sample
}

#[cfg(test)]
mod tests {
    use super::{FiltFiltPadLen, FiltFiltPadMode, FiltFiltParams, SosFilterState, clamp_pad_len};
    use crate::control::lti::{
        DiscreteSos, DiscreteStateSpace, DiscreteTransferFunction, LtiError, SecondOrderSection,
    };

    fn assert_close(lhs: f64, rhs: f64, tol: f64) {
        let err = (lhs - rhs).abs();
        assert!(err <= tol, "lhs={lhs}, rhs={rhs}, err={err}, tol={tol}");
    }

    fn assert_vec_close(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for (&lhs, &rhs) in lhs.iter().zip(rhs.iter()) {
            assert_close(lhs, rhs, tol);
        }
    }

    fn first_order_sos() -> DiscreteSos<f64> {
        let section = SecondOrderSection::new([1.0, 0.0, 0.0], [1.0, -0.5, 0.0]).unwrap();
        DiscreteSos::discrete(vec![section], 1.0, 1.0).unwrap()
    }

    fn first_order_state_space() -> DiscreteStateSpace<f64> {
        DiscreteTransferFunction::discrete(vec![1.0, 0.0], vec![1.0, -0.5], 1.0)
            .unwrap()
            .to_state_space()
            .unwrap()
    }

    #[test]
    fn state_space_forward_matches_manual_scalar_recurrence() {
        let system = first_order_state_space();
        let input = [1.0, 2.0, -1.0, 0.5];
        let actual = system.filter_forward(&input).unwrap();

        let mut expected_state = 0.0;
        let mut expected_output = Vec::new();
        for &sample in &input {
            expected_output.push(0.5 * expected_state + sample);
            expected_state = 0.5 * expected_state + sample;
        }

        assert_vec_close(&actual.output, &expected_output, 1.0e-12);
        assert_close(actual.final_state[0], expected_state, 1.0e-12);
    }

    #[test]
    fn sos_forward_matches_equivalent_state_space() {
        let sos = first_order_sos();
        let state_space = first_order_state_space();
        let input = [1.0, -0.5, 2.0, 0.25, -1.0];

        let lhs = sos.filter_forward(&input).unwrap();
        let rhs = state_space.filter_forward(&input).unwrap();
        assert_vec_close(&lhs.output, &rhs.output, 1.0e-12);
    }

    #[test]
    fn sos_stateful_chunked_processing_matches_one_shot() {
        let sos = first_order_sos();
        let input = [1.0, -0.5, 2.0, 0.25, -1.0];

        let one_shot = sos.filter_forward(&input).unwrap();
        let mut state = SosFilterState::for_filter(&sos);
        let first = sos
            .filter_forward_stateful(&mut state, &input[..2])
            .unwrap();
        let second = sos
            .filter_forward_stateful(&mut state, &input[2..])
            .unwrap();

        let mut combined = first.output;
        combined.extend(second.output);
        assert_vec_close(&combined, &one_shot.output, 1.0e-12);
    }

    #[test]
    fn filtfilt_short_input_auto_shortens_padding() {
        let sos = first_order_sos();
        let params = FiltFiltParams::new().with_len(FiltFiltPadLen::Exact(16));
        let filtered = sos.filtfilt_with_params(&[1.0, 2.0], &params).unwrap();
        assert_eq!(filtered.output.len(), 2);
    }

    #[test]
    fn sos_and_state_space_filtfilt_agree() {
        let sos = first_order_sos();
        let state_space = first_order_state_space();
        let input = [0.0, 1.0, 2.0, 1.0, 0.0, -1.0, 0.0];
        let params = FiltFiltParams::new()
            .with_mode(FiltFiltPadMode::OddReflection)
            .with_len(FiltFiltPadLen::Exact(3));

        let lhs = sos.filtfilt_with_params(&input, &params).unwrap();
        let rhs = state_space.filtfilt_with_params(&input, &params).unwrap();
        assert_vec_close(&lhs.output, &rhs.output, 1.0e-12);
    }

    #[test]
    fn non_siso_state_space_filtering_is_rejected() {
        let tf = DiscreteTransferFunction::discrete(vec![1.0], vec![1.0, -0.5], 1.0).unwrap();
        let base = tf.to_state_space().unwrap();
        let system = DiscreteStateSpace::new(
            base.a().to_owned(),
            faer::Mat::from_fn(base.nstates(), 2, |row, col| {
                if col == 0 { base.b()[(row, 0)] } else { 0.0 }
            }),
            faer::Mat::from_fn(2, base.nstates(), |row, col| {
                if row == 0 { base.c()[(0, col)] } else { 0.0 }
            }),
            faer::Mat::zeros(2, 2),
            1.0,
        )
        .unwrap();

        let err = system.filter_forward(&[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, LtiError::NonSisoStateSpace { .. }));
    }

    #[test]
    fn clamp_pad_len_respects_mode_limits() {
        assert_eq!(clamp_pad_len(5, FiltFiltPadMode::OddReflection, 99), 4);
        assert_eq!(clamp_pad_len(5, FiltFiltPadMode::EvenReflection, 99), 4);
        assert_eq!(clamp_pad_len(5, FiltFiltPadMode::Constant, 99), 5);
        assert_eq!(clamp_pad_len(5, FiltFiltPadMode::None, 99), 0);
    }
}
