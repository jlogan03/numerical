use crate::plot_helpers::{LineSeries, build_line_plot};
use crate::plotly_support::use_plotly_chart;
use faer::Mat;
use leptos::prelude::*;
use numerical::control::identification::{EraParams, OkidParams, era_from_markov, okid};
use numerical::control::lti::DiscreteStateSpace;
use plotly::Plot;

/// Interactive identification page demonstrating an OKID -> ERA workflow on
/// simulated sampled input/output data.
#[component]
pub fn IdentificationPage() -> impl IntoView {
    let (noise_level, set_noise_level) = signal(0.03_f64);
    let (observer_order, set_observer_order) = signal(6_usize);
    let (retained_order, set_retained_order) = signal(2_usize);

    use_plotly_chart("identification-markov-plot", move || {
        build_identification_plot(
            noise_level.get(),
            observer_order.get(),
            retained_order.get(),
            IdentificationPlot::Markov,
        )
    });
    use_plotly_chart("identification-step-plot", move || {
        build_identification_plot(
            noise_level.get(),
            observer_order.get(),
            retained_order.get(),
            IdentificationPlot::StepResponse,
        )
    });

    let summary = move || {
        identification_summary(
            noise_level.get(),
            observer_order.get(),
            retained_order.get(),
        )
    };

    view! {
        <div class="page">
            <header class="page-header">
                <p class="eyebrow">"Identification"</p>
                <h1>"OKID + ERA Recovery"</h1>
                <p>
                    "A planted discrete-time SISO system is excited with a deterministic rich input, OKID estimates"
                    " its Markov parameters from the resulting noisy input/output history, and ERA realizes a reduced"
                    " model from that recovered sequence."
                </p>
            </header>

            <div class="control-layout">
                <aside class="control-card">
                    <section>
                        <h2>"Identification controls"</h2>
                        <p class="section-copy">
                            "Observer order controls the lifted OKID regression horizon. Retained order controls the"
                            " final ERA realization size."
                        </p>

                        <div class="control-row">
                            <label for="id-noise-level">"Output noise"</label>
                            <output>{move || format!("{:.3}", noise_level.get())}</output>
                            <input
                                id="id-noise-level"
                                type="range"
                                min="0.0"
                                max="0.15"
                                step="0.005"
                                prop:value=move || noise_level.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_noise_level.set(value.max(0.0));
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="id-observer-order">"Observer order"</label>
                            <output>{move || observer_order.get().to_string()}</output>
                            <input
                                id="id-observer-order"
                                type="range"
                                min="3"
                                max="12"
                                step="1"
                                prop:value=move || observer_order.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<usize>() {
                                        set_observer_order.set(value.max(3));
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="id-retained-order">"ERA order"</label>
                            <output>{move || retained_order.get().to_string()}</output>
                            <input
                                id="id-retained-order"
                                type="range"
                                min="1"
                                max="4"
                                step="1"
                                prop:value=move || retained_order.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<usize>() {
                                        set_retained_order.set(value.max(1));
                                    }
                                }
                            />
                        </div>
                    </section>

                    <section>
                        <h2>"Interpretation"</h2>
                        <p class="section-copy">
                            "The left plot checks whether OKID recovered the planted Markov sequence. The right plot"
                            " checks whether the ERA realization built from those Markov parameters reproduces the"
                            " system-level step response."
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
                                <h2>"Markov parameters"</h2>
                                <p>"Planted versus identified impulse-response blocks."</p>
                            </div>
                        </div>
                        <div id="identification-markov-plot" class="plot-surface"></div>
                    </article>

                    <article class="plot-card">
                        <div class="plot-header">
                            <div>
                                <h2>"Step response"</h2>
                                <p>"Planted model versus the ERA realization built from the identified Markov sequence."</p>
                            </div>
                        </div>
                        <div id="identification-step-plot" class="plot-surface"></div>
                    </article>
                </div>
            </div>
        </div>
    }
}

#[derive(Clone, Copy)]
enum IdentificationPlot {
    Markov,
    StepResponse,
}

struct IdentificationDemo {
    lags: Vec<f64>,
    planted_markov: Vec<f64>,
    identified_markov: Vec<f64>,
    steps: Vec<f64>,
    planted_step: Vec<f64>,
    identified_step: Vec<f64>,
    retained_order: usize,
    rms_step_error: f64,
}

fn build_identification_plot(
    noise_level: f64,
    observer_order: usize,
    retained_order: usize,
    which: IdentificationPlot,
) -> Plot {
    match run_identification_demo(noise_level, observer_order, retained_order) {
        Ok(demo) => match which {
            IdentificationPlot::Markov => build_line_plot(
                "Markov-parameter recovery",
                "lag index",
                "H[k]",
                false,
                vec![
                    LineSeries::lines_markers("planted", demo.lags.clone(), demo.planted_markov),
                    LineSeries::lines_markers("identified", demo.lags, demo.identified_markov),
                ],
            ),
            IdentificationPlot::StepResponse => build_line_plot(
                "Step-response comparison",
                "step index",
                "output",
                false,
                vec![
                    LineSeries::lines("planted", demo.steps.clone(), demo.planted_step),
                    LineSeries::lines("identified", demo.steps, demo.identified_step),
                ],
            ),
        },
        Err(message) => build_line_plot(&message, "", "", false, Vec::new()),
    }
}

fn identification_summary(
    noise_level: f64,
    observer_order: usize,
    retained_order: usize,
) -> String {
    match run_identification_demo(noise_level, observer_order, retained_order) {
        Ok(demo) => format!(
            "ERA retained order {} with step-response RMS error {:.4}. Larger observer horizons usually improve the recovered Markov sequence until noise starts to dominate the regression.",
            demo.retained_order, demo.rms_step_error,
        ),
        Err(err) => format!("Identification failed: {err}"),
    }
}

fn run_identification_demo(
    noise_level: f64,
    observer_order: usize,
    retained_order: usize,
) -> Result<IdentificationDemo, String> {
    let dt = 0.2;
    let system = DiscreteStateSpace::new(
        Mat::from_fn(2, 2, |row, col| match (row, col) {
            (0, 0) => 0.86,
            (0, 1) => 0.18,
            (1, 0) => -0.08,
            (1, 1) => 0.82,
            _ => 0.0,
        }),
        Mat::from_fn(2, 1, |row, _| if row == 0 { 0.12 } else { 0.35 }),
        Mat::from_fn(1, 2, |_, col| if col == 0 { 1.0 } else { 0.28 }),
        Mat::zeros(1, 1),
        dt,
    )
    .map_err(|err| err.to_string())?;

    let n_samples = 140;
    let inputs = Mat::from_fn(1, n_samples, |_, col| excitation_signal(col));
    let sim = system
        .simulate(&[0.0, 0.0], inputs.as_ref())
        .map_err(|err| err.to_string())?;
    let noisy_outputs = Mat::from_fn(1, n_samples, |_, col| {
        sim.outputs[(0, col)] + noise_level * measurement_noise_signal(col)
    });

    let okid_result = okid(
        noisy_outputs.as_ref(),
        inputs.as_ref(),
        &OkidParams::new(24, observer_order),
    )
    .map_err(|err| err.to_string())?;
    let era = era_from_markov(
        &okid_result.markov,
        6,
        6,
        &EraParams::new(system.sample_time()).with_order(retained_order),
    )
    .map_err(|err| err.to_string())?;

    let lags = (0..12).map(|idx| idx as f64).collect::<Vec<_>>();
    let planted_markov = (0..12)
        .map(|idx| system.markov_parameters(12).block(idx)[(0, 0)])
        .collect::<Vec<_>>();
    let identified_markov = (0..12)
        .map(|idx| okid_result.markov.block(idx)[(0, 0)])
        .collect::<Vec<_>>();

    let planted_step_response = system.step_response(30);
    let identified_step_response = era.realized.step_response(30);
    let steps = (0..30).map(|idx| idx as f64).collect::<Vec<_>>();
    let planted_step = planted_step_response
        .values
        .iter()
        .map(|block| block[(0, 0)])
        .collect::<Vec<_>>();
    let identified_step = identified_step_response
        .values
        .iter()
        .map(|block| block[(0, 0)])
        .collect::<Vec<_>>();
    let rms_step_error = (planted_step
        .iter()
        .zip(&identified_step)
        .map(|(lhs, rhs)| {
            let err = lhs - rhs;
            err * err
        })
        .sum::<f64>()
        / (planted_step.len() as f64))
        .sqrt();

    Ok(IdentificationDemo {
        lags,
        planted_markov,
        identified_markov,
        steps,
        planted_step,
        identified_step,
        retained_order: era.retained_order,
        rms_step_error,
    })
}

fn excitation_signal(step: usize) -> f64 {
    let k = step as f64;
    let square = if (step / 10).is_multiple_of(2) {
        1.0
    } else {
        -0.65
    };
    square + 0.35 * (0.19 * k).sin() + 0.18 * (0.07 * k + 0.4).cos()
}

fn measurement_noise_signal(step: usize) -> f64 {
    let k = step as f64;
    0.7 * (0.37 * k + 0.2).sin() + 0.25 * (0.11 * k + 0.8).cos()
}
