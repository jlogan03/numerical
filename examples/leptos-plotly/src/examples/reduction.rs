use crate::plot_helpers::{LineSeries, build_line_plot, linspace};
use crate::plotly_support::use_plotly_chart;
use faer::Mat;
use leptos::prelude::*;
use numerical::control::lti::ContinuousStateSpace;
use numerical::control::reduction::BalancedParams;
use plotly::Plot;

/// Interactive balanced-truncation page comparing full and reduced models.
#[component]
pub fn ReductionPage() -> impl IntoView {
    let (retained_order, set_retained_order) = signal(3_usize);

    use_plotly_chart("reduction-step-plot", move || {
        build_reduction_plot(retained_order.get(), ReductionPlot::StepResponse)
    });
    use_plotly_chart("reduction-hsv-plot", move || {
        build_reduction_plot(retained_order.get(), ReductionPlot::Hsv)
    });

    let summary = move || reduction_summary(retained_order.get());

    view! {
        <div class="page">
            <header class="page-header">
                <p class="eyebrow">"Reduction"</p>
                <h1>"Balanced Truncation Explorer"</h1>
                <p>
                    "A stable six-state continuous model is reduced with dense balanced truncation. The first plot"
                    " overlays the full and reduced step responses, and the second shows the Hankel singular value"
                    " spectrum being truncated."
                </p>
            </header>

            <div class="control-layout">
                <aside class="control-card">
                    <section>
                        <h2>"Retained order"</h2>
                        <p class="section-copy">
                            "Balanced truncation ranks states by joint controllability and observability energy. Moving"
                            " this slider changes how many of those balanced directions are retained."
                        </p>

                        <div class="control-row">
                            <label for="reduction-order">"Retained states"</label>
                            <output>{move || retained_order.get().to_string()}</output>
                            <input
                                id="reduction-order"
                                type="range"
                                min="1"
                                max="6"
                                step="1"
                                prop:value=move || retained_order.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<usize>() {
                                        set_retained_order.set(value.clamp(1, 6));
                                    }
                                }
                            />
                        </div>
                    </section>

                    <section>
                        <h2>"Interpretation"</h2>
                        <p class="section-copy">
                            "The HSV spectrum tells you how much balanced energy each state direction contributes."
                            " When the tail decays quickly, low-order truncations preserve the dominant step-response"
                            " shape with little visible distortion."
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
                                <h2>"Step response"</h2>
                                <p>"Full model versus balanced-truncated model."</p>
                            </div>
                        </div>
                        <div id="reduction-step-plot" class="plot-surface"></div>
                    </article>

                    <article class="plot-card">
                        <div class="plot-header">
                            <div>
                                <h2>"Hankel singular values"</h2>
                                <p>"Balanced energy spectrum for the same model."</p>
                            </div>
                        </div>
                        <div id="reduction-hsv-plot" class="plot-surface"></div>
                    </article>
                </div>
            </div>
        </div>
    }
}

#[derive(Clone, Copy)]
enum ReductionPlot {
    StepResponse,
    Hsv,
}

struct ReductionDemo {
    times: Vec<f64>,
    full_step: Vec<f64>,
    reduced_step: Vec<f64>,
    hsv_indices: Vec<f64>,
    hsv_values: Vec<f64>,
    retained_order: usize,
    error_bound: Option<f64>,
    final_output_error: f64,
}

fn build_reduction_plot(retained_order: usize, which: ReductionPlot) -> Plot {
    match run_reduction_demo(retained_order) {
        Ok(demo) => match which {
            ReductionPlot::StepResponse => build_line_plot(
                "Balanced truncation step response",
                "time (s)",
                "output",
                false,
                vec![
                    LineSeries::lines("full model", demo.times.clone(), demo.full_step),
                    LineSeries::lines("reduced model", demo.times, demo.reduced_step),
                ],
            ),
            ReductionPlot::Hsv => build_line_plot(
                "Hankel singular values",
                "state index",
                "sigma",
                false,
                vec![LineSeries::lines_markers(
                    format!("retain r = {}", demo.retained_order),
                    demo.hsv_indices,
                    demo.hsv_values,
                )],
            ),
        },
        Err(message) => build_line_plot(&message, "", "", false, Vec::new()),
    }
}

fn reduction_summary(retained_order: usize) -> String {
    match run_reduction_demo(retained_order) {
        Ok(demo) => match demo.error_bound {
            Some(bound) => format!(
                "Retained order {} with balanced-truncation tail bound {:.4}. Final sampled step-response error is {:.4}.",
                demo.retained_order, bound, demo.final_output_error,
            ),
            None => format!(
                "Retained order {}. Final sampled step-response error is {:.4}.",
                demo.retained_order, demo.final_output_error,
            ),
        },
        Err(err) => format!("Reduction failed: {err}"),
    }
}

fn run_reduction_demo(retained_order: usize) -> Result<ReductionDemo, String> {
    let system = ContinuousStateSpace::new(
        Mat::from_fn(6, 6, |row, col| match (row, col) {
            (0, 0) => -0.45,
            (0, 1) => 0.14,
            (1, 1) => -0.62,
            (1, 2) => 0.12,
            (2, 2) => -0.82,
            (2, 3) => 0.11,
            (3, 3) => -1.05,
            (3, 4) => 0.09,
            (4, 4) => -1.32,
            (4, 5) => 0.08,
            (5, 5) => -1.70,
            _ => 0.0,
        }),
        Mat::from_fn(6, 1, |row, _| match row {
            0 => 1.0,
            1 => 0.92,
            2 => 0.84,
            3 => 0.75,
            4 => 0.67,
            _ => 0.58,
        }),
        Mat::from_fn(1, 6, |_, col| match col {
            0 => 1.0,
            1 => 0.92,
            2 => 0.83,
            3 => 0.72,
            4 => 0.61,
            _ => 0.52,
        }),
        Mat::zeros(1, 1),
    )
    .map_err(|err| err.to_string())?;

    let result = system
        .balanced_truncation(&BalancedParams::new().with_order(retained_order))
        .map_err(|err| err.to_string())?;
    let sample_times = linspace(0.0, 18.0, 220);
    let full_step_response = system
        .step_response(&sample_times)
        .map_err(|err| err.to_string())?;
    let reduced_step_response = result
        .reduced
        .step_response(&sample_times)
        .map_err(|err| err.to_string())?;

    let full_step = full_step_response
        .values
        .iter()
        .map(|block| block[(0, 0)])
        .collect::<Vec<_>>();
    let reduced_step = reduced_step_response
        .values
        .iter()
        .map(|block| block[(0, 0)])
        .collect::<Vec<_>>();
    let final_output_error = (full_step.last().copied().unwrap_or(0.0_f64)
        - reduced_step.last().copied().unwrap_or(0.0_f64))
    .abs();

    let hsv_values = (0..result.hankel_singular_values.nrows())
        .map(|idx| result.hankel_singular_values[idx])
        .collect::<Vec<_>>();
    let hsv_indices = (0..hsv_values.len())
        .map(|idx| (idx + 1) as f64)
        .collect::<Vec<_>>();

    Ok(ReductionDemo {
        times: sample_times,
        full_step,
        reduced_step,
        hsv_indices,
        hsv_values,
        retained_order: result.reduced_order,
        error_bound: result.error_bound,
        final_output_error,
    })
}
