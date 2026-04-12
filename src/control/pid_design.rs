//! PID tuning helpers based on low-order process models.
//!
//! The first implementation focuses on the pragmatic workflow that is easiest
//! to validate and explain:
//!
//! - fit `FOPDT` / `SOPDT` process models from sampled step-response data
//! - tune `PI` / `PIDF` controllers from those fitted models with SIMC-style
//!   formulas
//!
//! More general transfer-function tuning and identification-driven PID design
//! belong to later phases once the process-model path is in place.

use super::pid::{AntiWindup, Pid, PidError};
use core::fmt;
use faer_traits::RealField;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, differentiate_numerically};
use nalgebra::storage::Owned;
use nalgebra::{Dyn, OMatrix, OVector, U1, U3, U4, VecStorage, Vector3, Vector4};
use num_traits::Float;

/// Errors produced by PID process-model fitting and SIMC-style tuning.
#[derive(Debug)]
pub enum PidDesignError {
    /// The sampled data is structurally invalid.
    InvalidData { which: &'static str },
    /// The detected input step is too small to identify a process gain.
    InvalidStepAmplitude,
    /// The observed output change is too small to identify a process gain.
    InvalidResponseAmplitude,
    /// A delay estimate could not be formed from the sampled response.
    InvalidDelayEstimate,
    /// The fitted process parameters are not physically meaningful.
    InvalidProcessFit,
    /// The requested tuning parameter is invalid.
    InvalidTuningParameter { which: &'static str },
    /// PID construction failed after the tuning rule produced gains.
    Pid(PidError),
}

impl fmt::Display for PidDesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for PidDesignError {}

impl From<PidError> for PidDesignError {
    fn from(value: PidError) -> Self {
        Self::Pid(value)
    }
}

/// Sampled SISO step-response experiment data.
///
/// The first implementation assumes one dominant input step over the record.
#[derive(Clone, Debug, PartialEq)]
pub struct StepResponseData<R> {
    time: Vec<R>,
    input: Vec<R>,
    output: Vec<R>,
}

impl<R> StepResponseData<R>
where
    R: Float + Copy,
{
    /// Creates validated step-response data.
    pub fn new(time: Vec<R>, input: Vec<R>, output: Vec<R>) -> Result<Self, PidDesignError> {
        if time.len() < 4 {
            return Err(PidDesignError::InvalidData { which: "time.len" });
        }
        if time.len() != input.len() || time.len() != output.len() {
            return Err(PidDesignError::InvalidData {
                which: "length_mismatch",
            });
        }
        if time
            .iter()
            .chain(input.iter())
            .chain(output.iter())
            .any(|value| !value.is_finite())
        {
            return Err(PidDesignError::InvalidData {
                which: "nonfinite_sample",
            });
        }
        if time.windows(2).any(|window| !(window[1] > window[0])) {
            return Err(PidDesignError::InvalidData {
                which: "nonmonotone_time",
            });
        }
        Ok(Self {
            time,
            input,
            output,
        })
    }

    /// Sample times.
    #[must_use]
    pub fn time(&self) -> &[R] {
        &self.time
    }

    /// Input samples.
    #[must_use]
    pub fn input(&self) -> &[R] {
        &self.input
    }

    /// Output samples.
    #[must_use]
    pub fn output(&self) -> &[R] {
        &self.output
    }
}

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
    R: Float + Copy,
{
    /// Evaluates the delayed unit-step kernel multiplied by `step_amplitude`
    /// and shifted by `initial_output`.
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
            initial_output
                + self.gain * step_amplitude * (R::one() - (-(theta / self.time_constant)).exp())
        }
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
    R: Float + Copy,
{
    /// Evaluates the delayed second-order step response for a step of amplitude
    /// `step_amplitude` and baseline `initial_output`.
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
        initial_output + self.gain * step_amplitude * lag
    }
}

/// Result of fitting a low-order process model to sampled data.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcessFitResult<M, R> {
    /// Structured initial estimate derived directly from response landmarks.
    pub initial: M,
    /// Refined model after local nonlinear least-squares improvement.
    pub model: M,
    /// Sum of squared residuals of the final fit.
    pub objective: R,
}

/// PID controller family produced by a tuning rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PidControllerKind {
    /// `PI` controller. The derivative path is disabled.
    Pi,
    /// `PIDF` controller with a first-order filtered derivative term.
    Pid,
}

/// SIMC-style tuning parameters shared by the low-order process-model rules.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimcPidParams<R> {
    /// Desired closed-loop time constant / aggressiveness parameter.
    pub lambda: R,
    /// Whether to return a `PI` or `PIDF` controller.
    pub controller: PidControllerKind,
    /// Derivative filter ratio `N_f`, interpreted as `derivative_filter = N_f / Td`.
    pub derivative_filter_ratio: R,
    /// Runtime anti-windup policy to attach to the returned controller.
    pub anti_windup: AntiWindup<R>,
    /// Optional direct limits on the stored integral contribution.
    pub integrator_limits: Option<(R, R)>,
}

impl<R> SimcPidParams<R>
where
    R: Float + Copy,
{
    /// Creates validated tuning parameters.
    pub fn new(
        lambda: R,
        controller: PidControllerKind,
        derivative_filter_ratio: R,
        anti_windup: AntiWindup<R>,
    ) -> Result<Self, PidDesignError> {
        if !lambda.is_finite() || lambda <= R::zero() {
            return Err(PidDesignError::InvalidTuningParameter { which: "lambda" });
        }
        if !derivative_filter_ratio.is_finite() || derivative_filter_ratio <= R::zero() {
            return Err(PidDesignError::InvalidTuningParameter {
                which: "derivative_filter_ratio",
            });
        }
        Ok(Self {
            lambda,
            controller,
            derivative_filter_ratio,
            anti_windup,
            integrator_limits: None,
        })
    }

    /// Adds explicit integrator limits to the tuned controller.
    pub fn with_integrator_limits(mut self, low: R, high: R) -> Result<Self, PidDesignError> {
        if !low.is_finite() || !high.is_finite() || low > high {
            return Err(PidDesignError::InvalidTuningParameter {
                which: "integrator_limits",
            });
        }
        self.integrator_limits = Some((low, high));
        Ok(self)
    }
}

/// Result of tuning a PID controller from a low-order process model.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcessPidDesign<R, M> {
    /// Tuned runtime controller.
    pub pid: Pid<R>,
    /// Process model the tuning rule used.
    pub model: M,
    /// Tuning parameters applied by the rule.
    pub params: SimcPidParams<R>,
}

/// Result of tuning from sampled step-response data.
#[derive(Clone, Debug, PartialEq)]
pub struct StepFitPidDesign<R, M> {
    /// Fitted process model.
    pub fit: ProcessFitResult<M, R>,
    /// Tuned controller using the fitted model.
    pub design: ProcessPidDesign<R, M>,
}

/// Fits an `FOPDT` model to sampled step-response data.
pub fn fit_fopdt_from_step_response(
    data: &StepResponseData<f64>,
) -> Result<ProcessFitResult<FopdtModel<f64>, f64>, PidDesignError> {
    let prepared = PreparedStepResponse::from_data(data)?;
    let initial = initial_fopdt_estimate(&prepared)?;
    let initial_objective = fopdt_objective(&prepared, initial);
    let model = match refine_fopdt(&prepared, initial) {
        Ok(refined) => {
            let refined_objective = fopdt_objective(&prepared, refined);
            if refined_objective <= initial_objective {
                refined
            } else {
                initial
            }
        }
        Err(_) => initial,
    };
    let objective = fopdt_objective(&prepared, model);
    Ok(ProcessFitResult {
        initial,
        model,
        objective,
    })
}

/// Fits an `SOPDT` model to sampled step-response data.
pub fn fit_sopdt_from_step_response(
    data: &StepResponseData<f64>,
) -> Result<ProcessFitResult<SopdtModel<f64>, f64>, PidDesignError> {
    let prepared = PreparedStepResponse::from_data(data)?;
    let fopdt = initial_fopdt_estimate(&prepared)?;
    let initial = initial_sopdt_estimate(&prepared, fopdt);
    let initial_objective = sopdt_objective(&prepared, initial);
    let model = match refine_sopdt(&prepared, initial) {
        Ok(refined) => {
            let refined_objective = sopdt_objective(&prepared, refined);
            if refined_objective <= initial_objective {
                refined
            } else {
                initial
            }
        }
        Err(_) => initial,
    };
    let objective = sopdt_objective(&prepared, model);
    Ok(ProcessFitResult {
        initial,
        model,
        objective,
    })
}

/// Tunes a `PI` or `PIDF` controller from an `FOPDT` process model.
pub fn design_pid_from_fopdt<R>(
    model: FopdtModel<R>,
    params: SimcPidParams<R>,
) -> Result<ProcessPidDesign<R, FopdtModel<R>>, PidDesignError>
where
    R: Float + Copy + RealField,
{
    validate_process_gain(model.gain)?;
    validate_positive(model.time_constant, "time_constant")?;
    validate_nonnegative(model.delay, "delay")?;

    let kp = model.time_constant / (model.gain * (params.lambda + model.delay));
    let (ki, kd, derivative_filter) = match params.controller {
        PidControllerKind::Pi => {
            let ti = min_value(
                model.time_constant,
                four::<R>() * (params.lambda + model.delay),
            );
            (kp / ti, R::zero(), None)
        }
        PidControllerKind::Pid => {
            let ti = model.time_constant + model.delay / two::<R>();
            let td = (model.time_constant * model.delay)
                / (two::<R>() * model.time_constant + model.delay);
            let derivative_filter = if td > R::zero() {
                Some(params.derivative_filter_ratio / td)
            } else {
                None
            };
            (kp / ti, kp * td, derivative_filter)
        }
    };

    let mut pid = Pid::new(kp, ki, kd, derivative_filter, params.anti_windup)?;
    if let Some((low, high)) = params.integrator_limits {
        pid = pid.with_integrator_limits(low, high)?;
    }

    Ok(ProcessPidDesign { pid, model, params })
}

/// Tunes a `PI` or `PIDF` controller from an `SOPDT` process model.
pub fn design_pid_from_sopdt<R>(
    model: SopdtModel<R>,
    params: SimcPidParams<R>,
) -> Result<ProcessPidDesign<R, SopdtModel<R>>, PidDesignError>
where
    R: Float + Copy + RealField,
{
    validate_process_gain(model.gain)?;
    validate_positive(model.time_constant_1, "time_constant_1")?;
    validate_positive(model.time_constant_2, "time_constant_2")?;
    validate_nonnegative(model.delay, "delay")?;

    let lag_sum = model.time_constant_1 + model.time_constant_2;
    let kp = lag_sum / (model.gain * (params.lambda + model.delay));
    let (ki, kd, derivative_filter) = match params.controller {
        PidControllerKind::Pi => {
            let ti = min_value(lag_sum, four::<R>() * (params.lambda + model.delay));
            (kp / ti, R::zero(), None)
        }
        PidControllerKind::Pid => {
            let ti = lag_sum;
            let td = (model.time_constant_1 * model.time_constant_2) / lag_sum;
            let derivative_filter = if td > R::zero() {
                Some(params.derivative_filter_ratio / td)
            } else {
                None
            };
            (kp / ti, kp * td, derivative_filter)
        }
    };

    let mut pid = Pid::new(kp, ki, kd, derivative_filter, params.anti_windup)?;
    if let Some((low, high)) = params.integrator_limits {
        pid = pid.with_integrator_limits(low, high)?;
    }

    Ok(ProcessPidDesign { pid, model, params })
}

/// Fits an `FOPDT` model from sampled step data and tunes from the result.
pub fn design_pid_from_step_response_fopdt(
    data: &StepResponseData<f64>,
    params: SimcPidParams<f64>,
) -> Result<StepFitPidDesign<f64, FopdtModel<f64>>, PidDesignError> {
    let fit = fit_fopdt_from_step_response(data)?;
    let design = design_pid_from_fopdt(fit.model, params)?;
    Ok(StepFitPidDesign { fit, design })
}

/// Fits an `SOPDT` model from sampled step data and tunes from the result.
pub fn design_pid_from_step_response_sopdt(
    data: &StepResponseData<f64>,
    params: SimcPidParams<f64>,
) -> Result<StepFitPidDesign<f64, SopdtModel<f64>>, PidDesignError> {
    let fit = fit_sopdt_from_step_response(data)?;
    let design = design_pid_from_sopdt(fit.model, params)?;
    Ok(StepFitPidDesign { fit, design })
}

#[derive(Clone, Debug)]
struct PreparedStepResponse {
    /// Time grid shifted so the dominant input step occurs near `t = 0`.
    time_relative: Vec<f64>,
    /// Measured output samples on that shifted grid.
    output: Vec<f64>,
    /// Estimated pre-step output baseline.
    initial_output: f64,
    /// Estimated input step amplitude.
    step_amplitude: f64,
    /// Coarse steady-state process gain estimate from baseline/tail averages.
    gain_guess: f64,
    /// Characteristic experiment time span used to scale positivity floors.
    time_scale: f64,
}

impl PreparedStepResponse {
    /// Normalizes raw sampled step data into the quantities needed by the
    /// landmark estimators and LM refinement.
    ///
    /// The preprocessing intentionally looks for one dominant input jump and
    /// then computes pre-step and post-step averages around that detected
    /// transition. That keeps the first fitting pass simple and robust for the
    /// intended "single experiment, single step" workflow.
    fn from_data(data: &StepResponseData<f64>) -> Result<Self, PidDesignError> {
        let n = data.time.len();
        let step_index =
            dominant_step_index(&data.input).ok_or(PidDesignError::InvalidStepAmplitude)?;
        let pre_end = (step_index + 1).max(1);
        let post_start = (step_index + 1).min(n - 1);
        let tail = tail_count(n - post_start);

        let initial_input = mean(&data.input[..pre_end]);
        let final_input = mean(&data.input[post_start..]);
        let initial_output = mean(&data.output[..pre_end]);
        let final_output = mean(&data.output[n - tail..]);

        let step_amplitude = final_input - initial_input;
        if !step_amplitude.is_finite() || step_amplitude.abs() <= 1.0e-12 {
            return Err(PidDesignError::InvalidStepAmplitude);
        }
        let output_change = final_output - initial_output;
        if !output_change.is_finite() || output_change.abs() <= 1.0e-12 {
            return Err(PidDesignError::InvalidResponseAmplitude);
        }

        let step_time = interpolate_input_step_time(&data.time, &data.input, step_index)?;

        let time_relative = data.time.iter().map(|&t| t - step_time).collect::<Vec<_>>();
        let time_scale = (data.time[n - 1] - data.time[0]).abs().max(1.0e-6);
        Ok(Self {
            time_relative,
            output: data.output.clone(),
            initial_output,
            step_amplitude,
            gain_guess: output_change / step_amplitude,
            time_scale,
        })
    }
}

/// Forms the first FOPDT estimate from standard normalized step landmarks.
///
/// The `28.3%` and `63.2%` crossings give a coarse delay/lag split that is
/// easy to compute and usually good enough to seed local nonlinear refinement.
fn initial_fopdt_estimate(data: &PreparedStepResponse) -> Result<FopdtModel<f64>, PidDesignError> {
    let final_output = *data.output.last().ok_or(PidDesignError::InvalidData {
        which: "output.last",
    })?;
    let total_change = final_output - data.initial_output;
    let gain = data.gain_guess;
    let sign = total_change.signum();
    let level_283 = data.initial_output + 0.283 * total_change;
    let level_632 = data.initial_output + 0.632 * total_change;

    let start_index = first_nonnegative_index(&data.time_relative);
    let t_283 = interpolate_crossing(
        &data.time_relative,
        &data.output,
        level_283,
        sign,
        start_index,
    )
    .ok_or(PidDesignError::InvalidDelayEstimate)?;
    let t_632 = interpolate_crossing(
        &data.time_relative,
        &data.output,
        level_632,
        sign,
        start_index,
    )
    .ok_or(PidDesignError::InvalidDelayEstimate)?;

    let tau = 1.5 * (t_632 - t_283);
    let delay = (t_632 - tau).max(0.0);
    if !tau.is_finite() || tau <= 0.0 || !delay.is_finite() {
        return Err(PidDesignError::InvalidProcessFit);
    }

    Ok(FopdtModel {
        gain,
        time_constant: tau,
        delay,
    })
}

/// Builds a physically plausible SOPDT seed from the coarser FOPDT estimate.
///
/// This does not try to solve the second-order fit in closed form. Instead it
/// searches a small deterministic grid of lag splits and delay scalings and
/// keeps the candidate with the smallest direct step-response residual.
fn initial_sopdt_estimate(data: &PreparedStepResponse, fopdt: FopdtModel<f64>) -> SopdtModel<f64> {
    let floor = positive_floor(data.time_scale);
    let total_scales = [0.5, 0.75, 1.0, 1.25, 1.5];
    let split_fractions = [0.55, 0.65, 0.75, 0.85, 0.95];
    let delay_scales = [0.5, 1.0, 1.5];

    let mut best = SopdtModel {
        gain: fopdt.gain,
        time_constant_1: (0.75 * fopdt.time_constant).max(floor),
        time_constant_2: (0.25 * fopdt.time_constant).max(floor),
        delay: fopdt.delay.max(0.0),
    };
    let mut best_objective = sopdt_objective(data, best);

    for &total_scale in &total_scales {
        let total = (fopdt.time_constant * total_scale).max(2.0 * floor);
        for &split in &split_fractions {
            let tau1 = (total * split).max(floor);
            let tau2 = (total * (1.0 - split)).max(floor);
            for &delay_scale in &delay_scales {
                let candidate = SopdtModel {
                    gain: fopdt.gain,
                    time_constant_1: tau1.max(tau2),
                    time_constant_2: tau1.min(tau2),
                    delay: (fopdt.delay * delay_scale).max(0.0),
                };
                let objective = sopdt_objective(data, candidate);
                if objective < best_objective {
                    best = candidate;
                    best_objective = objective;
                }
            }
        }
    }

    best
}

/// Runs LM refinement for the FOPDT parameters.
///
/// The time constant and delay are optimized in log coordinates so the
/// recovered model stays positive without adding explicit inequality
/// constraints to the least-squares problem.
fn refine_fopdt(
    data: &PreparedStepResponse,
    initial: FopdtModel<f64>,
) -> Result<FopdtModel<f64>, PidDesignError> {
    let floor = positive_floor(data.time_scale);
    let problem = FopdtLmProblem {
        data,
        params: Vector3::new(
            initial.gain,
            initial.time_constant.max(floor).ln(),
            initial.delay.max(floor).ln(),
        ),
        floor,
    };
    let (problem, report) = LevenbergMarquardt::new().minimize(problem);
    let model = problem.model();
    if !report.termination.was_successful()
        || !model.gain.is_finite()
        || !model.time_constant.is_finite()
        || !model.delay.is_finite()
        || model.time_constant <= 0.0
        || model.delay < 0.0
    {
        return Err(PidDesignError::InvalidProcessFit);
    }
    Ok(model)
}

/// Runs LM refinement for the SOPDT parameters.
///
/// The parameterization uses:
///
/// - one base lag
/// - one positive lag separation
/// - one positive delay
///
/// so both time constants remain positive and ordered automatically.
fn refine_sopdt(
    data: &PreparedStepResponse,
    initial: SopdtModel<f64>,
) -> Result<SopdtModel<f64>, PidDesignError> {
    let floor = positive_floor(data.time_scale);
    let base = initial.time_constant_2.max(floor);
    let extra = (initial.time_constant_1 - initial.time_constant_2)
        .abs()
        .max(floor);
    let problem = SopdtLmProblem {
        data,
        params: Vector4::new(
            initial.gain,
            base.ln(),
            extra.ln(),
            initial.delay.max(floor).ln(),
        ),
        floor,
    };
    let (problem, report) = LevenbergMarquardt::new().minimize(problem);
    let model = problem.model();
    if !report.termination.was_successful()
        || !model.gain.is_finite()
        || !model.time_constant_1.is_finite()
        || !model.time_constant_2.is_finite()
        || !model.delay.is_finite()
        || model.time_constant_1 <= 0.0
        || model.time_constant_2 <= 0.0
        || model.delay < 0.0
    {
        return Err(PidDesignError::InvalidProcessFit);
    }
    Ok(model)
}

/// Sum-of-squares objective used to compare FOPDT candidates.
fn fopdt_objective(data: &PreparedStepResponse, model: FopdtModel<f64>) -> f64 {
    data.time_relative
        .iter()
        .zip(data.output.iter())
        .map(|(&t, &y)| {
            let residual =
                model.step_response_value(t, data.step_amplitude, data.initial_output) - y;
            residual * residual
        })
        .sum()
}

/// Sum-of-squares objective used to compare SOPDT candidates.
fn sopdt_objective(data: &PreparedStepResponse, model: SopdtModel<f64>) -> f64 {
    data.time_relative
        .iter()
        .zip(data.output.iter())
        .map(|(&t, &y)| {
            let residual =
                model.step_response_value(t, data.step_amplitude, data.initial_output) - y;
            residual * residual
        })
        .sum()
}

#[derive(Clone)]
struct FopdtLmProblem<'a> {
    data: &'a PreparedStepResponse,
    /// Parameters stored as `[gain, ln(tau), ln(delay)]`.
    params: Vector3<f64>,
    floor: f64,
}

impl<'a> FopdtLmProblem<'a> {
    /// Maps the LM parameter vector back to a physical FOPDT model.
    fn model(&self) -> FopdtModel<f64> {
        FopdtModel {
            gain: self.params[0],
            time_constant: self.params[1].exp().max(self.floor),
            delay: self.params[2].exp().max(self.floor),
        }
    }

    /// Residual vector for the sampled delayed first-order step model.
    fn residual_vector(&self) -> OVector<f64, Dyn> {
        let model = self.model();
        let residuals = self
            .data
            .time_relative
            .iter()
            .zip(self.data.output.iter())
            .map(|(&t, &y)| {
                model.step_response_value(t, self.data.step_amplitude, self.data.initial_output) - y
            })
            .collect::<Vec<_>>();
        OVector::<f64, Dyn>::from_vec(residuals)
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, U3> for FopdtLmProblem<'a> {
    type ParameterStorage = Owned<f64, U3>;
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U3>;

    fn set_params(&mut self, x: &Vector3<f64>) {
        self.params = *x;
    }

    fn params(&self) -> Vector3<f64> {
        self.params
    }

    fn residuals(&self) -> Option<OVector<f64, Dyn>> {
        Some(self.residual_vector())
    }

    fn jacobian(&self) -> Option<OMatrix<f64, Dyn, U3>> {
        // A numerical Jacobian keeps the first implementation compact. The
        // fitted problems are small enough that this cost is acceptable.
        let mut clone = self.clone();
        differentiate_numerically(&mut clone)
    }
}

#[derive(Clone)]
struct SopdtLmProblem<'a> {
    data: &'a PreparedStepResponse,
    /// Parameters stored as `[gain, ln(base_tau), ln(extra_tau), ln(delay)]`.
    params: Vector4<f64>,
    floor: f64,
}

impl<'a> SopdtLmProblem<'a> {
    /// Maps the LM parameter vector back to a physical SOPDT model.
    fn model(&self) -> SopdtModel<f64> {
        let base = self.params[1].exp().max(self.floor);
        let extra = self.params[2].exp().max(self.floor);
        SopdtModel {
            gain: self.params[0],
            time_constant_1: base + extra,
            time_constant_2: base,
            delay: self.params[3].exp().max(self.floor),
        }
    }

    /// Residual vector for the sampled delayed second-order step model.
    fn residual_vector(&self) -> OVector<f64, Dyn> {
        let model = self.model();
        let residuals = self
            .data
            .time_relative
            .iter()
            .zip(self.data.output.iter())
            .map(|(&t, &y)| {
                model.step_response_value(t, self.data.step_amplitude, self.data.initial_output) - y
            })
            .collect::<Vec<_>>();
        OVector::<f64, Dyn>::from_vec(residuals)
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, U4> for SopdtLmProblem<'a> {
    type ParameterStorage = Owned<f64, U4>;
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U4>;

    fn set_params(&mut self, x: &Vector4<f64>) {
        self.params = *x;
    }

    fn params(&self) -> Vector4<f64> {
        self.params
    }

    fn residuals(&self) -> Option<OVector<f64, Dyn>> {
        Some(self.residual_vector())
    }

    fn jacobian(&self) -> Option<OMatrix<f64, Dyn, U4>> {
        // As in the FOPDT path, numerical differentiation is enough for this
        // small reference fitter and keeps the algebra out of the first pass.
        let mut clone = self.clone();
        differentiate_numerically(&mut clone)
    }
}

/// Step response of two cascaded first-order lags driven by a unit step.
///
/// When the lags nearly coincide, the ordinary distinct-pole formula suffers
/// from cancellation, so the repeated-pole limit is used instead.
fn second_order_lag_step<R>(time: R, tau1: R, tau2: R) -> R
where
    R: Float + Copy,
{
    let tol = (tau1.abs() + tau2.abs() + R::one()) * R::from(1.0e-8).unwrap();
    if (tau1 - tau2).abs() <= tol {
        let tau = (tau1 + tau2) / two::<R>();
        let scaled = time / tau;
        R::one() - (R::one() + scaled) * (-scaled).exp()
    } else {
        let e1 = (-(time / tau1)).exp();
        let e2 = (-(time / tau2)).exp();
        R::one() - (tau1 * e1 - tau2 * e2) / (tau1 - tau2)
    }
}

/// Finds the first crossing of `target` after `start_index` with the expected
/// response direction.
fn interpolate_crossing<R>(
    time: &[R],
    signal: &[R],
    target: R,
    direction: R,
    start_index: usize,
) -> Option<R>
where
    R: Float + Copy,
{
    if direction == R::zero() {
        return None;
    }
    for idx in start_index..signal.len().saturating_sub(1) {
        let lhs = signal[idx];
        let rhs = signal[idx + 1];
        let crossed = if direction > R::zero() {
            lhs <= target && rhs >= target
        } else {
            lhs >= target && rhs <= target
        };
        if crossed {
            let dy = rhs - lhs;
            if dy == R::zero() {
                return Some(time[idx]);
            }
            let alpha = (target - lhs) / dy;
            return Some(time[idx] + alpha * (time[idx + 1] - time[idx]));
        }
    }
    None
}

/// Returns the first index whose shifted time is nonnegative.
fn first_nonnegative_index(values: &[f64]) -> usize {
    values.iter().position(|&value| value >= 0.0).unwrap_or(0)
}

/// Finds the dominant single input jump in the sampled command record.
///
/// The current fitting path assumes one main step experiment, so the largest
/// adjacent jump is treated as the step used for identification.
fn dominant_step_index(signal: &[f64]) -> Option<usize> {
    let mut best_index = None;
    let mut best_jump = 0.0f64;
    for idx in 0..signal.len().saturating_sub(1) {
        let jump = (signal[idx + 1] - signal[idx]).abs();
        if jump > best_jump {
            best_jump = jump;
            best_index = Some(idx);
        }
    }
    if best_jump > 1.0e-12 {
        best_index
    } else {
        None
    }
}

/// Interpolates the detected input step to a sub-sample transition time.
fn interpolate_input_step_time(
    time: &[f64],
    input: &[f64],
    step_index: usize,
) -> Result<f64, PidDesignError> {
    let lhs_u = input[step_index];
    let rhs_u = input[step_index + 1];
    let lhs_t = time[step_index];
    let rhs_t = time[step_index + 1];
    let du = rhs_u - lhs_u;
    if du == 0.0 {
        return Err(PidDesignError::InvalidData {
            which: "input_step_time",
        });
    }
    let target = lhs_u + 0.5 * du;
    let alpha = (target - lhs_u) / du;
    Ok(lhs_t + alpha * (rhs_t - lhs_t))
}

/// Heuristic tail length used for steady-state averaging.
fn tail_count(n: usize) -> usize {
    (n / 10).max(3).min(n.max(1))
}

/// Arithmetic mean helper for baseline and tail estimates.
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Positive floor derived from the experiment time scale.
///
/// This prevents zero or negative lag/delay values in the transformed LM
/// parameters while staying small relative to the sampled experiment.
fn positive_floor(scale: f64) -> f64 {
    (1.0e-9 * scale).max(1.0e-9)
}

/// Rejects zero or nonfinite process gains before applying a tuning rule.
fn validate_process_gain<R>(gain: R) -> Result<(), PidDesignError>
where
    R: Float + Copy,
{
    if gain.is_finite() && gain != R::zero() {
        Ok(())
    } else {
        Err(PidDesignError::InvalidProcessFit)
    }
}

/// Validates a strictly positive tuning or process parameter.
fn validate_positive<R>(value: R, which: &'static str) -> Result<(), PidDesignError>
where
    R: Float + Copy,
{
    if value.is_finite() && value > R::zero() {
        Ok(())
    } else {
        Err(PidDesignError::InvalidTuningParameter { which })
    }
}

/// Validates a nonnegative tuning or process parameter.
fn validate_nonnegative<R>(value: R, which: &'static str) -> Result<(), PidDesignError>
where
    R: Float + Copy,
{
    if value.is_finite() && value >= R::zero() {
        Ok(())
    } else {
        Err(PidDesignError::InvalidTuningParameter { which })
    }
}

/// Scalar minimum helper that keeps the generic tuning formulas readable.
fn min_value<R>(lhs: R, rhs: R) -> R
where
    R: Float + Copy,
{
    if lhs < rhs { lhs } else { rhs }
}

/// Returns the scalar constant `2`.
fn two<R>() -> R
where
    R: Float + Copy,
{
    R::from(2.0).unwrap()
}

/// Returns the scalar constant `4`.
fn four<R>() -> R
where
    R: Float + Copy,
{
    R::from(4.0).unwrap()
}

#[cfg(test)]
mod tests {
    use super::{
        AntiWindup, FopdtModel, PidControllerKind, PidDesignError, SimcPidParams, SopdtModel,
        StepResponseData, design_pid_from_fopdt, design_pid_from_sopdt,
        design_pid_from_step_response_fopdt, fit_fopdt_from_step_response,
        fit_sopdt_from_step_response,
    };

    fn assert_close(lhs: f64, rhs: f64, tol: f64) {
        let err = (lhs - rhs).abs();
        assert!(err <= tol, "lhs={lhs}, rhs={rhs}, err={err}, tol={tol}");
    }

    fn fopdt_step_data(model: FopdtModel<f64>, dt: f64, duration: f64) -> StepResponseData<f64> {
        let n = (duration / dt).round() as usize + 1;
        let step_time = 5.0 * dt;
        let mut time = Vec::with_capacity(n);
        let mut input = Vec::with_capacity(n);
        let mut output = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 * dt;
            time.push(t);
            let u = if t >= step_time { 1.0 } else { 0.0 };
            input.push(u);
            output.push(model.step_response_value((t - step_time).max(0.0), 1.0, 0.0));
        }
        StepResponseData::new(time, input, output).unwrap()
    }

    fn sopdt_step_data(model: SopdtModel<f64>, dt: f64, duration: f64) -> StepResponseData<f64> {
        let n = (duration / dt).round() as usize + 1;
        let step_time = 5.0 * dt;
        let mut time = Vec::with_capacity(n);
        let mut input = Vec::with_capacity(n);
        let mut output = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 * dt;
            time.push(t);
            let u = if t >= step_time { 1.0 } else { 0.0 };
            input.push(u);
            output.push(model.step_response_value((t - step_time).max(0.0), 1.0, 0.0));
        }
        StepResponseData::new(time, input, output).unwrap()
    }

    #[test]
    fn step_response_data_rejects_invalid_inputs() {
        let err =
            StepResponseData::new(vec![0.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0]).unwrap_err();
        assert!(matches!(err, PidDesignError::InvalidData { .. }));
    }

    #[test]
    fn fit_fopdt_recovers_synthetic_step_response() {
        let model = FopdtModel {
            gain: 2.0,
            time_constant: 3.0,
            delay: 0.7,
        };
        let data = fopdt_step_data(model, 0.05, 20.0);
        let fit = fit_fopdt_from_step_response(&data).unwrap();

        assert_close(fit.model.gain, model.gain, 1.0e-2);
        assert_close(fit.model.time_constant, model.time_constant, 5.0e-2);
        assert_close(fit.model.delay, model.delay, 5.0e-2);
        assert!(fit.objective <= 1.0e-6);
    }

    #[test]
    fn fit_sopdt_recovers_synthetic_step_response() {
        let model = SopdtModel {
            gain: 1.5,
            time_constant_1: 2.5,
            time_constant_2: 0.8,
            delay: 0.4,
        };
        let data = sopdt_step_data(model, 0.05, 20.0);
        let fit = fit_sopdt_from_step_response(&data).unwrap();

        assert_close(fit.model.gain, model.gain, 1.0e-2);
        assert_close(fit.model.time_constant_1, model.time_constant_1, 7.5e-2);
        assert_close(fit.model.time_constant_2, model.time_constant_2, 7.5e-2);
        assert_close(fit.model.delay, model.delay, 5.0e-2);
        assert!(fit.objective <= 1.0e-6);
    }

    #[test]
    fn simc_fopdt_pi_matches_expected_formula() {
        let model = FopdtModel {
            gain: 2.0,
            time_constant: 4.0,
            delay: 1.0,
        };
        let params =
            SimcPidParams::new(2.0, PidControllerKind::Pi, 10.0, AntiWindup::None).unwrap();
        let design = design_pid_from_fopdt(model, params).unwrap();
        assert_close(design.pid.kp(), 4.0 / (2.0 * (2.0 + 1.0)), 1.0e-12);
        assert_close(design.pid.ki(), design.pid.kp() / 4.0, 1.0e-12);
        assert_close(design.pid.kd(), 0.0, 1.0e-12);
    }

    #[test]
    fn simc_sopdt_pid_populates_derivative_filter() {
        let model = SopdtModel {
            gain: 1.0,
            time_constant_1: 3.0,
            time_constant_2: 1.0,
            delay: 0.5,
        };
        let params =
            SimcPidParams::new(1.5, PidControllerKind::Pid, 8.0, AntiWindup::None).unwrap();
        let design = design_pid_from_sopdt(model, params).unwrap();
        assert!(design.pid.kd() > 0.0);
        assert!(design.pid.derivative_filter().unwrap() > 0.0);
    }

    #[test]
    fn step_fit_design_builds_pid_from_fopdt_fit() {
        let model = FopdtModel {
            gain: 1.2,
            time_constant: 2.0,
            delay: 0.3,
        };
        let data = fopdt_step_data(model, 0.05, 12.0);
        let params =
            SimcPidParams::new(1.0, PidControllerKind::Pid, 10.0, AntiWindup::None).unwrap();
        let design = design_pid_from_step_response_fopdt(&data, params).unwrap();
        assert_close(design.fit.model.gain, model.gain, 1.0e-3);
        assert!(design.design.pid.kp().is_finite());
    }

    #[test]
    fn fit_rejects_zero_step_amplitude() {
        let data = StepResponseData::new(
            vec![0.0, 1.0, 2.0, 3.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let err = fit_fopdt_from_step_response(&data).unwrap_err();
        assert!(matches!(err, PidDesignError::InvalidStepAmplitude));
    }
}
