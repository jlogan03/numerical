use crate::plot_helpers::{LineSeries, build_line_plot};
use crate::plotly_support::use_plotly_chart;
use faer::{Mat, MatRef};
use leptos::prelude::*;
use numerical::control::estimation::{
    DiscreteExtendedKalmanModel, DiscreteKalmanFilter, DiscreteNonlinearModel,
    ExtendedKalmanFilter, UnscentedKalmanFilter, UnscentedParams,
};
use numerical::control::lti::DiscreteStateSpace;
use plotly::Plot;
use plotly::common::DashType;

/// Interactive nonlinear-estimation page comparing a fixed-linearization
/// Kalman filter, an EKF, and a UKF on the same range-tracking problem.
#[component]
pub fn NonlinearEstimationPage() -> impl IntoView {
    let (sensor_height, set_sensor_height) = signal(8.0_f64);
    let (linear_reference, set_linear_reference) = signal(7.0_f64);
    let (true_process_noise, set_true_process_noise) = signal(0.18_f64);
    let (true_measurement_noise, set_true_measurement_noise) = signal(0.35_f64);
    let (assumed_process_noise, set_assumed_process_noise) = signal(0.18_f64);
    let (assumed_measurement_noise, set_assumed_measurement_noise) = signal(0.35_f64);
    let (pin_assumptions, set_pin_assumptions) = signal(true);

    let effective_assumed_process_noise = Memo::new(move |_| {
        if pin_assumptions.get() {
            true_process_noise.get()
        } else {
            assumed_process_noise.get()
        }
    });
    let effective_assumed_measurement_noise = Memo::new(move |_| {
        if pin_assumptions.get() {
            true_measurement_noise.get()
        } else {
            assumed_measurement_noise.get()
        }
    });

    use_plotly_chart("nonlinear-estimator-position-plot", move || {
        build_nonlinear_estimation_plot(
            sensor_height.get(),
            linear_reference.get(),
            true_process_noise.get(),
            true_measurement_noise.get(),
            effective_assumed_process_noise.get(),
            effective_assumed_measurement_noise.get(),
            NonlinearEstimationPlot::Position,
        )
    });
    use_plotly_chart("nonlinear-estimator-error-plot", move || {
        build_nonlinear_estimation_plot(
            sensor_height.get(),
            linear_reference.get(),
            true_process_noise.get(),
            true_measurement_noise.get(),
            effective_assumed_process_noise.get(),
            effective_assumed_measurement_noise.get(),
            NonlinearEstimationPlot::Error,
        )
    });

    let summary = move || {
        nonlinear_estimation_summary(
            sensor_height.get(),
            linear_reference.get(),
            true_process_noise.get(),
            true_measurement_noise.get(),
            effective_assumed_process_noise.get(),
            effective_assumed_measurement_noise.get(),
        )
    };

    view! {
        <div class="page">
            <header class="page-header">
                <p class="eyebrow">"Estimation"</p>
                <h1>"Nonlinear Tracking Workbench"</h1>
                <p>
                    "A constant-velocity target is observed through a nonlinear slant-range sensor."
                    " The comparison uses the same truth trajectory, range measurements, and assumed noise"
                    " levels for a fixed-linearization Kalman filter, an EKF, and a UKF."
                </p>
            </header>

            <div class="control-layout">
                <aside class="control-card">
                    <section>
                        <h2>"Settings"</h2>
                        <p class="section-copy">
                            "The geometry controls set the measurement nonlinearity. The truth sliders scale"
                            " the disturbance injected into the target acceleration and the noise added to the"
                            " range sensor. The filter sliders set the covariances assumed by all three estimators."
                        </p>

                        <div class="control-row">
                            <label for="nonlinear-sensor-height">"Sensor height"</label>
                            <output>{move || format!("{:.2}", sensor_height.get())}</output>
                            <input
                                id="nonlinear-sensor-height"
                                type="range"
                                min="2.0"
                                max="12.0"
                                step="0.1"
                                prop:value=move || sensor_height.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_sensor_height.set(value.max(2.0));
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="nonlinear-linear-reference">"Linearization reference"</label>
                            <output>{move || format!("{:.2}", linear_reference.get())}</output>
                            <input
                                id="nonlinear-linear-reference"
                                type="range"
                                min="1.0"
                                max="12.0"
                                step="0.1"
                                prop:value=move || linear_reference.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_linear_reference.set(value.max(1.0));
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="nonlinear-true-process-noise">"True process noise"</label>
                            <output>{move || format!("{:.3}", true_process_noise.get())}</output>
                            <input
                                id="nonlinear-true-process-noise"
                                type="range"
                                min="0.02"
                                max="2.00"
                                step="0.01"
                                prop:value=move || true_process_noise.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_true_process_noise.set(value.max(0.02));
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="nonlinear-true-measurement-noise">"True measurement noise"</label>
                            <output>{move || format!("{:.3}", true_measurement_noise.get())}</output>
                            <input
                                id="nonlinear-true-measurement-noise"
                                type="range"
                                min="0.05"
                                max="2.00"
                                step="0.01"
                                prop:value=move || true_measurement_noise.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_true_measurement_noise.set(value.max(0.05));
                                    }
                                }
                            />
                        </div>
                    </section>

                    <section>
                        <h2>"Filter assumptions"</h2>
                        <p class="section-copy">
                            "When the assumptions are pinned, all three filters use matched process and measurement"
                            " covariances. Unlock the sliders to study mismatch. The fixed-linearization KF always uses"
                            " one measurement linearization around the chosen reference position."
                        </p>

                        <div class="control-row">
                            <label for="nonlinear-pin-assumptions">"Pin assumptions to truth"</label>
                            <input
                                id="nonlinear-pin-assumptions"
                                type="checkbox"
                                prop:checked=move || pin_assumptions.get()
                                on:change=move |ev| set_pin_assumptions.set(event_target_checked(&ev))
                            />
                        </div>

                        <div class="control-row">
                            <label for="nonlinear-assumed-process-noise">"Assumed process noise"</label>
                            <output>{move || format!("{:.3}", effective_assumed_process_noise.get())}</output>
                            <input
                                id="nonlinear-assumed-process-noise"
                                type="range"
                                min="0.02"
                                max="2.00"
                                step="0.01"
                                prop:value=move || effective_assumed_process_noise.get().to_string()
                                prop:disabled=move || pin_assumptions.get()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_assumed_process_noise.set(value.max(0.02));
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="nonlinear-assumed-measurement-noise">"Assumed measurement noise"</label>
                            <output>{move || format!("{:.3}", effective_assumed_measurement_noise.get())}</output>
                            <input
                                id="nonlinear-assumed-measurement-noise"
                                type="range"
                                min="0.05"
                                max="2.00"
                                step="0.01"
                                prop:value=move || effective_assumed_measurement_noise.get().to_string()
                                prop:disabled=move || pin_assumptions.get()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_assumed_measurement_noise.set(value.max(0.05));
                                    }
                                }
                            />
                        </div>
                    </section>

                    <section>
                        <h2>"Run summary"</h2>
                        <p class="section-copy">{summary}</p>
                    </section>
                </aside>

                <div class="plots-grid wide">
                    <div class="plots-grid compact">
                        <article class="plot-card">
                            <div class="plot-header">
                                <div>
                                    <h2>"Position estimate"</h2>
                                    <p>
                                        "Truth, a positive-branch inversion of the noisy range measurement for display,"
                                        " and the three competing filters."
                                    </p>
                                </div>
                            </div>
                            <div id="nonlinear-estimator-position-plot" class="plot-surface"></div>
                        </article>

                        <article class="plot-card">
                            <div class="plot-header">
                                <div>
                                    <h2>"Absolute error"</h2>
                                    <p>"Position-estimation error on the same tracking run."</p>
                                </div>
                            </div>
                            <div id="nonlinear-estimator-error-plot" class="plot-surface"></div>
                        </article>
                    </div>

                    <section class="home-grid">
                        <article class="home-card">
                            <h2>"Tracking problem"</h2>
                            <p class="section-copy">
                                "The state is `[position, velocity]^T` with constant-velocity dynamics driven by an"
                                " acceleration command plus an unmodeled acceleration disturbance. The sensor returns"
                                " slant range `y_k = sqrt(position_k^2 + h^2) + n_k`, where `h` is the sensor-height"
                                " offset above the track."
                            </p>
                        </article>

                        <article class="home-card">
                            <h2>"Filters compared"</h2>
                            <p class="section-copy">
                                "The linear baseline uses one fixed measurement linearization around the chosen reference"
                                " position. The EKF recomputes the local Jacobian from its current state estimate each"
                                " step. The UKF pushes sigma points through the same nonlinear measurement map instead"
                                " of relying on a first-order Jacobian."
                            </p>
                        </article>

                        <article class="home-card">
                            <h2>"How To Use It"</h2>
                            <p class="section-copy">
                                "Increase sensor height or move the fixed linearization point away from the target path"
                                " to make the measurement more nonlinear from the linear KF's point of view. The EKF and"
                                " especially the UKF should then retain lower error on the same measurement sequence."
                            </p>
                        </article>
                    </section>
                </div>
            </div>
        </div>
    }
}

#[derive(Clone, Copy)]
enum NonlinearEstimationPlot {
    Position,
    Error,
}

#[derive(Clone, Copy, Debug)]
struct RangeTrackingModel {
    dt: f64,
    sensor_height: f64,
}

impl DiscreteNonlinearModel<f64> for RangeTrackingModel {
    fn nstates(&self) -> usize {
        2
    }

    fn ninputs(&self) -> usize {
        1
    }

    fn noutputs(&self) -> usize {
        1
    }

    fn transition(&self, x: MatRef<'_, f64>, u: MatRef<'_, f64>) -> Mat<f64> {
        let dt = self.dt;
        let acceleration = u[(0, 0)];
        Mat::from_fn(2, 1, |row, _| match row {
            0 => x[(0, 0)] + dt * x[(1, 0)] + 0.5 * dt * dt * acceleration,
            1 => x[(1, 0)] + dt * acceleration,
            _ => 0.0,
        })
    }

    fn output(&self, x: MatRef<'_, f64>, _u: MatRef<'_, f64>) -> Mat<f64> {
        Mat::from_fn(1, 1, |_, _| {
            (x[(0, 0)] * x[(0, 0)] + self.sensor_height * self.sensor_height).sqrt()
        })
    }
}

impl DiscreteExtendedKalmanModel<f64> for RangeTrackingModel {
    fn transition_jacobian(&self, _x: MatRef<'_, f64>, _u: MatRef<'_, f64>) -> Mat<f64> {
        let dt = self.dt;
        Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => 1.0,
            (0, 1) => dt,
            (1, 0) => 0.0,
            (1, 1) => 1.0,
            _ => 0.0,
        })
    }

    fn output_jacobian(&self, x: MatRef<'_, f64>, _u: MatRef<'_, f64>) -> Mat<f64> {
        let position = x[(0, 0)];
        let denom = (position * position + self.sensor_height * self.sensor_height).sqrt();
        let slope = if denom > 0.0 { position / denom } else { 0.0 };
        Mat::from_fn(1, 2, |_, col| if col == 0 { slope } else { 0.0 })
    }
}

struct NonlinearEstimationDemo {
    times: Vec<f64>,
    truth_position: Vec<f64>,
    measured_proxy_position: Vec<f64>,
    linear_position: Vec<f64>,
    ekf_position: Vec<f64>,
    ukf_position: Vec<f64>,
    linear_abs_error: Vec<f64>,
    ekf_abs_error: Vec<f64>,
    ukf_abs_error: Vec<f64>,
    linear_rmse: f64,
    ekf_rmse: f64,
    ukf_rmse: f64,
    best_filter: &'static str,
}

fn build_nonlinear_estimation_plot(
    sensor_height: f64,
    linear_reference: f64,
    true_process_noise: f64,
    true_measurement_noise: f64,
    assumed_process_noise: f64,
    assumed_measurement_noise: f64,
    which: NonlinearEstimationPlot,
) -> Plot {
    match run_nonlinear_estimation_demo(
        sensor_height,
        linear_reference,
        true_process_noise,
        true_measurement_noise,
        assumed_process_noise,
        assumed_measurement_noise,
    ) {
        Ok(demo) => match which {
            NonlinearEstimationPlot::Position => build_line_plot(
                "Position estimate",
                "time (s)",
                "position",
                false,
                vec![
                    LineSeries::lines("truth", demo.times.clone(), demo.truth_position),
                    LineSeries::lines(
                        "measurement proxy",
                        demo.times.clone(),
                        demo.measured_proxy_position,
                    )
                    .with_dash(DashType::Dot)
                    .with_line_width(1.0),
                    LineSeries::lines("linearized KF", demo.times.clone(), demo.linear_position)
                        .with_dash(DashType::Dash),
                    LineSeries::lines("EKF", demo.times.clone(), demo.ekf_position)
                        .with_dash(DashType::DashDot),
                    LineSeries::lines("UKF", demo.times, demo.ukf_position)
                        .with_dash(DashType::LongDash),
                ],
            ),
            NonlinearEstimationPlot::Error => build_line_plot(
                "Absolute position error",
                "time (s)",
                "|estimate - truth|",
                false,
                vec![
                    LineSeries::lines("linearized KF", demo.times.clone(), demo.linear_abs_error)
                        .with_dash(DashType::Dash),
                    LineSeries::lines("EKF", demo.times.clone(), demo.ekf_abs_error)
                        .with_dash(DashType::DashDot),
                    LineSeries::lines("UKF", demo.times, demo.ukf_abs_error)
                        .with_dash(DashType::LongDash),
                ],
            ),
        },
        Err(message) => build_line_plot(&message, "time (s)", "", false, Vec::new()),
    }
}

fn nonlinear_estimation_summary(
    sensor_height: f64,
    linear_reference: f64,
    true_process_noise: f64,
    true_measurement_noise: f64,
    assumed_process_noise: f64,
    assumed_measurement_noise: f64,
) -> String {
    match run_nonlinear_estimation_demo(
        sensor_height,
        linear_reference,
        true_process_noise,
        true_measurement_noise,
        assumed_process_noise,
        assumed_measurement_noise,
    ) {
        Ok(demo) => format!(
            "Sensor height {:.2}, fixed linearization at position {:.2}. Truth uses process {:.3} and measurement {:.3}; filters assume process {:.3} and measurement {:.3}. RMSE: linearized KF {:.3}, EKF {:.3}, UKF {:.3}. Best on this run: {}.",
            sensor_height,
            linear_reference,
            true_process_noise,
            true_measurement_noise,
            assumed_process_noise,
            assumed_measurement_noise,
            demo.linear_rmse,
            demo.ekf_rmse,
            demo.ukf_rmse,
            demo.best_filter,
        ),
        Err(err) => format!("Nonlinear tracking setup failed: {err}"),
    }
}

fn run_nonlinear_estimation_demo(
    sensor_height: f64,
    linear_reference: f64,
    true_process_noise: f64,
    true_measurement_noise: f64,
    assumed_process_noise: f64,
    assumed_measurement_noise: f64,
) -> Result<NonlinearEstimationDemo, String> {
    let dt = 0.1;
    let nonlinear_model = RangeTrackingModel { dt, sensor_height };

    let linear_measurement_slope = linear_reference
        / (linear_reference * linear_reference + sensor_height * sensor_height).sqrt();
    let linear_measurement_bias =
        (linear_reference * linear_reference + sensor_height * sensor_height).sqrt()
            - linear_measurement_slope * linear_reference;

    let linear_system = DiscreteStateSpace::new(
        Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => 1.0,
            (0, 1) => dt,
            (1, 0) => 0.0,
            (1, 1) => 1.0,
            _ => 0.0,
        }),
        Mat::from_fn(2, 1, |row, _| if row == 0 { 0.5 * dt * dt } else { dt }),
        Mat::from_fn(1, 2, |_, col| {
            if col == 0 {
                linear_measurement_slope
            } else {
                0.0
            }
        }),
        Mat::zeros(1, 1),
        dt,
    )
    .map_err(|err| err.to_string())?;

    let assumed_q = assumed_process_noise * assumed_process_noise;
    let w = Mat::from_fn(2, 2, |row, col| match (row, col) {
        (0, 0) => 0.25 * dt.powi(4) * assumed_q,
        (0, 1) | (1, 0) => 0.5 * dt.powi(3) * assumed_q,
        (1, 1) => dt.powi(2) * assumed_q,
        _ => 0.0,
    });
    let v = Mat::from_fn(1, 1, |_, _| {
        assumed_measurement_noise * assumed_measurement_noise
    });

    let x_hat0 = Mat::from_fn(2, 1, |row, _| match row {
        0 => linear_reference,
        1 => -0.15,
        _ => 0.0,
    });
    let p0 = Mat::from_fn(2, 2, |row, col| {
        if row == col {
            if row == 0 { 6.0 } else { 1.5 }
        } else {
            0.0
        }
    });

    let mut linear_kf = DiscreteKalmanFilter::from_state_space(
        &linear_system,
        w.clone(),
        v.clone(),
        x_hat0.clone(),
        p0.clone(),
    )
    .map_err(|err| err.to_string())?;
    let mut ekf = ExtendedKalmanFilter::new(
        nonlinear_model,
        w.clone(),
        v.clone(),
        x_hat0.clone(),
        p0.clone(),
    )
    .map_err(|err| err.to_string())?;
    let mut ukf = UnscentedKalmanFilter::new_standard(
        nonlinear_model,
        w,
        v,
        x_hat0,
        p0,
        UnscentedParams {
            alpha: 0.45,
            beta: 2.0,
            kappa: 0.0,
        },
    )
    .map_err(|err| err.to_string())?;

    let n_steps = 110;
    let mut truth = [6.0_f64, -0.25_f64];
    let mut times = Vec::with_capacity(n_steps);
    let mut truth_position = Vec::with_capacity(n_steps);
    let mut measured_proxy_position = Vec::with_capacity(n_steps);
    let mut linear_position = Vec::with_capacity(n_steps);
    let mut ekf_position = Vec::with_capacity(n_steps);
    let mut ukf_position = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        let t = (step as f64) * dt;
        let command = tracking_command(step);
        let disturbance = true_process_noise * colored_signal(step, 0.23);
        let measurement = (truth[0] * truth[0] + sensor_height * sensor_height).sqrt()
            + true_measurement_noise * colored_signal(step, 1.41);

        let input = Mat::from_fn(1, 1, |_, _| command);
        let measurement_mat = Mat::from_fn(1, 1, |_, _| measurement);
        let linear_measurement = Mat::from_fn(1, 1, |_, _| measurement - linear_measurement_bias);

        let linear_update = linear_kf
            .step(input.as_ref(), linear_measurement.as_ref())
            .map_err(|err| err.to_string())?;
        let ekf_update = ekf
            .step(input.as_ref(), measurement_mat.as_ref())
            .map_err(|err| err.to_string())?;
        let ukf_update = ukf
            .step(input.as_ref(), measurement_mat.as_ref())
            .map_err(|err| err.to_string())?;

        times.push(t);
        truth_position.push(truth[0]);
        measured_proxy_position.push(measurement_proxy_position(measurement, sensor_height));
        linear_position.push(linear_update.state[(0, 0)]);
        ekf_position.push(ekf_update.state[(0, 0)]);
        ukf_position.push(ukf_update.state[(0, 0)]);

        let applied_acceleration = command + disturbance;
        truth = [
            truth[0] + dt * truth[1] + 0.5 * dt * dt * applied_acceleration,
            truth[1] + dt * applied_acceleration,
        ];
    }

    let linear_abs_error = absolute_error(&linear_position, &truth_position);
    let ekf_abs_error = absolute_error(&ekf_position, &truth_position);
    let ukf_abs_error = absolute_error(&ukf_position, &truth_position);
    let linear_rmse = rmse(&linear_position, &truth_position);
    let ekf_rmse = rmse(&ekf_position, &truth_position);
    let ukf_rmse = rmse(&ukf_position, &truth_position);

    let best_filter = if linear_rmse <= ekf_rmse && linear_rmse <= ukf_rmse {
        "linearized KF"
    } else if ekf_rmse <= ukf_rmse {
        "EKF"
    } else {
        "UKF"
    };

    Ok(NonlinearEstimationDemo {
        times,
        truth_position,
        measured_proxy_position,
        linear_position,
        ekf_position,
        ukf_position,
        linear_abs_error,
        ekf_abs_error,
        ukf_abs_error,
        linear_rmse,
        ekf_rmse,
        ukf_rmse,
        best_filter,
    })
}

fn measurement_proxy_position(measurement: f64, sensor_height: f64) -> f64 {
    (measurement * measurement - sensor_height * sensor_height)
        .max(0.0)
        .sqrt()
}

fn absolute_error(estimate: &[f64], truth: &[f64]) -> Vec<f64> {
    estimate
        .iter()
        .zip(truth.iter())
        .map(|(estimate, truth)| (estimate - truth).abs())
        .collect()
}

fn rmse(estimate: &[f64], truth: &[f64]) -> f64 {
    let mse = estimate
        .iter()
        .zip(truth.iter())
        .map(|(estimate, truth)| {
            let error = estimate - truth;
            error * error
        })
        .sum::<f64>()
        / (estimate.len().max(1) as f64);
    mse.sqrt()
}

fn tracking_command(step: usize) -> f64 {
    let k = step as f64;
    0.10 * (0.09 * k).sin() + if step >= 30 { 0.10 } else { 0.0 }
        - if step >= 72 { 0.14 } else { 0.0 }
}

fn colored_signal(step: usize, phase: f64) -> f64 {
    let k = step as f64;
    0.8 * (0.17 * k + phase).sin() + 0.35 * (0.07 * k + 0.5 * phase).cos()
}
