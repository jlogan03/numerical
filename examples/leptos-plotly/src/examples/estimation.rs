use crate::plot_helpers::{LineSeries, build_line_plot};
use crate::plotly_support::use_plotly_chart;
use faer::Mat;
use leptos::prelude::*;
use numerical::control::estimation::{DiscreteKalmanFilter, SteadyStateKalmanFilter};
use numerical::control::lti::DiscreteStateSpace;
use plotly::Plot;

/// Interactive linear-estimation page comparing full and steady-state Kalman
/// filters on the same measured signal.
#[component]
pub fn EstimationPage() -> impl IntoView {
    let (process_noise, set_process_noise) = signal(0.18_f64);
    let (measurement_noise, set_measurement_noise) = signal(0.35_f64);

    use_plotly_chart("estimator-position-plot", move || {
        build_estimation_plot(
            process_noise.get(),
            measurement_noise.get(),
            EstimationPlot::State,
        )
    });
    use_plotly_chart("estimator-variance-plot", move || {
        build_estimation_plot(
            process_noise.get(),
            measurement_noise.get(),
            EstimationPlot::Variance,
        )
    });

    let summary = move || estimation_summary(process_noise.get(), measurement_noise.get());

    view! {
        <div class="page">
            <header class="page-header">
                <p class="eyebrow">"Estimation"</p>
                <h1>"Discrete Kalman Workbench"</h1>
                <p>
                    "A noisy constant-velocity sensor stream is filtered with both the full recursive"
                    " discrete Kalman filter and its fixed-gain steady-state limit. The plots are generated"
                    " directly from the runtime estimator APIs."
                </p>
            </header>

            <div class="control-layout">
                <aside class="control-card">
                    <section>
                        <h2>"Noise model"</h2>
                        <p class="section-copy">
                            "The process slider sets the assumed acceleration-disturbance standard deviation."
                            " The measurement slider sets the position-sensor noise standard deviation."
                        </p>

                        <div class="control-row">
                            <label for="kalman-process-noise">"Process noise"</label>
                            <output>{move || format!("{:.3}", process_noise.get())}</output>
                            <input
                                id="kalman-process-noise"
                                type="range"
                                min="0.02"
                                max="0.60"
                                step="0.01"
                                prop:value=move || process_noise.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_process_noise.set(value.max(0.02));
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="kalman-measurement-noise">"Measurement noise"</label>
                            <output>{move || format!("{:.3}", measurement_noise.get())}</output>
                            <input
                                id="kalman-measurement-noise"
                                type="range"
                                min="0.05"
                                max="1.00"
                                step="0.01"
                                prop:value=move || measurement_noise.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_measurement_noise.set(value.max(0.05));
                                    }
                                }
                            />
                        </div>
                    </section>

                    <section>
                        <h2>"What to look for"</h2>
                        <p class="section-copy">
                            "The full filter starts from a broad posterior covariance and converges toward the"
                            " fixed-gain observer. Higher assumed process noise keeps the estimator more reactive,"
                            " while higher measurement noise pushes both filters toward smoother state estimates."
                        </p>
                    </section>

                    <section>
                        <h2>"Run summary"</h2>
                        <p class="section-copy">{summary}</p>
                    </section>
                </aside>

                <div class="plots-grid compact">
                    <article class="plot-card">
                        <div class="plot-header">
                            <div>
                                <h2>"Position estimate"</h2>
                                <p>"Truth, noisy measurement, recursive Kalman estimate, and fixed-gain estimate."</p>
                            </div>
                        </div>
                        <div id="estimator-position-plot" class="plot-surface"></div>
                    </article>

                    <article class="plot-card">
                        <div class="plot-header">
                            <div>
                                <h2>"Position variance"</h2>
                                <p>"Recursive posterior variance converging toward the steady-state observer covariance."</p>
                            </div>
                        </div>
                        <div id="estimator-variance-plot" class="plot-surface"></div>
                    </article>
                </div>
            </div>
        </div>
    }
}

#[derive(Clone, Copy)]
enum EstimationPlot {
    State,
    Variance,
}

struct EstimationDemo {
    times: Vec<f64>,
    truth_position: Vec<f64>,
    measured_position: Vec<f64>,
    kalman_position: Vec<f64>,
    steady_position: Vec<f64>,
    kalman_variance: Vec<f64>,
    steady_variance: Vec<f64>,
    final_kalman_error: f64,
    final_steady_error: f64,
}

fn build_estimation_plot(
    process_noise: f64,
    measurement_noise: f64,
    which: EstimationPlot,
) -> Plot {
    match run_estimation_demo(process_noise, measurement_noise) {
        Ok(demo) => match which {
            EstimationPlot::State => build_line_plot(
                "Position estimate",
                "time (s)",
                "position",
                false,
                vec![
                    LineSeries::lines("truth", demo.times.clone(), demo.truth_position),
                    LineSeries::lines_markers(
                        "measurement",
                        demo.times.clone(),
                        demo.measured_position,
                    ),
                    LineSeries::lines("kalman", demo.times.clone(), demo.kalman_position),
                    LineSeries::lines("steady-state", demo.times, demo.steady_position),
                ],
            ),
            EstimationPlot::Variance => build_line_plot(
                "Posterior position variance",
                "time (s)",
                "variance",
                false,
                vec![
                    LineSeries::lines("kalman P[0,0]", demo.times.clone(), demo.kalman_variance),
                    LineSeries::lines("steady-state P[0,0]", demo.times, demo.steady_variance),
                ],
            ),
        },
        Err(message) => build_line_plot(&message, "time (s)", "", false, Vec::new()),
    }
}

fn estimation_summary(process_noise: f64, measurement_noise: f64) -> String {
    match run_estimation_demo(process_noise, measurement_noise) {
        Ok(demo) => format!(
            "Final position error: recursive {:.3}, steady-state {:.3}. Both filters use the same constant-velocity model, but only the full filter carries a transient covariance contraction phase.",
            demo.final_kalman_error, demo.final_steady_error,
        ),
        Err(err) => format!("Estimator setup failed: {err}"),
    }
}

fn run_estimation_demo(
    process_noise: f64,
    measurement_noise: f64,
) -> Result<EstimationDemo, String> {
    let dt: f64 = 0.1;
    let system = DiscreteStateSpace::new(
        Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => 1.0,
            (0, 1) => dt,
            (1, 0) => 0.0,
            (1, 1) => 1.0,
            _ => 0.0,
        }),
        Mat::from_fn(2, 1, |row, _| if row == 0 { 0.5 * dt * dt } else { dt }),
        Mat::from_fn(1, 2, |_, col| if col == 0 { 1.0 } else { 0.0 }),
        Mat::zeros(1, 1),
        dt,
    )
    .map_err(|err| err.to_string())?;

    let q = process_noise * process_noise;
    let w = Mat::from_fn(2, 2, |row, col| match (row, col) {
        (0, 0) => 0.25 * dt.powi(4) * q,
        (0, 1) | (1, 0) => 0.5 * dt.powi(3) * q,
        (1, 1) => dt.powi(2) * q,
        _ => 0.0,
    });
    let v = Mat::from_fn(1, 1, |_, _| measurement_noise * measurement_noise);
    let x0 = Mat::zeros(2, 1);
    let p0 = Mat::from_fn(2, 2, |row, col| {
        if row == col {
            if row == 0 { 4.0 } else { 1.0 }
        } else {
            0.0
        }
    });

    let mut kalman =
        DiscreteKalmanFilter::from_state_space(&system, w.clone(), v.clone(), x0.clone(), p0)
            .map_err(|err| err.to_string())?;
    let mut steady = SteadyStateKalmanFilter::from_dlqe(&system, w.as_ref(), v.as_ref(), x0)
        .map_err(|err| err.to_string())?;
    let steady_var = steady
        .steady_state_covariance()
        .map(|cov| cov[(0, 0)])
        .unwrap_or(0.0);

    let n_steps = 120;
    let mut truth = [0.0_f64, 0.0_f64];
    let mut times = Vec::with_capacity(n_steps);
    let mut truth_position = Vec::with_capacity(n_steps);
    let mut measured_position = Vec::with_capacity(n_steps);
    let mut kalman_position = Vec::with_capacity(n_steps);
    let mut steady_position = Vec::with_capacity(n_steps);
    let mut kalman_variance = Vec::with_capacity(n_steps);
    let mut steady_variance = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        let t = (step as f64) * dt;
        let command = 0.35 * (0.12 * (step as f64)).sin() + if step >= 45 { 0.18 } else { 0.0 };
        let disturbance = process_noise * colored_signal(step, 0.31);
        let measurement = truth[0] + measurement_noise * colored_signal(step, 1.17);

        let input = Mat::from_fn(1, 1, |_, _| command);
        let measurement_mat = Mat::from_fn(1, 1, |_, _| measurement);
        let kalman_update = kalman
            .step(input.as_ref(), measurement_mat.as_ref())
            .map_err(|err| err.to_string())?;
        let steady_update = steady
            .step(input.as_ref(), measurement_mat.as_ref())
            .map_err(|err| err.to_string())?;

        times.push(t);
        truth_position.push(truth[0]);
        measured_position.push(measurement);
        kalman_position.push(kalman_update.state[(0, 0)]);
        steady_position.push(steady_update.state[(0, 0)]);
        kalman_variance.push(kalman_update.covariance[(0, 0)]);
        steady_variance.push(steady_var);

        let applied_input = command + disturbance;
        let next_position = truth[0] + dt * truth[1] + 0.5 * dt * dt * applied_input;
        let next_velocity = truth[1] + dt * applied_input;
        truth = [next_position, next_velocity];
    }

    let final_truth = *truth_position.last().unwrap_or(&0.0);
    let final_kalman = *kalman_position.last().unwrap_or(&0.0);
    let final_steady = *steady_position.last().unwrap_or(&0.0);

    Ok(EstimationDemo {
        times,
        truth_position,
        measured_position,
        kalman_position,
        steady_position,
        kalman_variance,
        steady_variance,
        final_kalman_error: final_kalman - final_truth,
        final_steady_error: final_steady - final_truth,
    })
}

fn colored_signal(step: usize, phase: f64) -> f64 {
    let k = step as f64;
    0.8 * (0.17 * k + phase).sin() + 0.35 * (0.07 * k + 0.5 * phase).cos()
}
