use crate::plotly_support::use_plotly_chart;
use gloo_net::http::Request;
use leptos::prelude::*;
use plotly::{
    Layout, Plot, Scatter,
    common::{Mode, Title},
    layout::{Axis, AxisType},
};
use serde::{Deserialize, Serialize};

/// Interactive digital lowpass filter-design page.
#[component]
pub fn FilterDesignPage() -> impl IntoView {
    let (order, set_order) = signal(4_usize);
    let (cutoff, set_cutoff) = signal(8.0_f64);
    let (sample_rate, set_sample_rate) = signal(20.0_f64);
    let response = LocalResource::new(move || {
        let order = order.get();
        let cutoff = cutoff.get();
        let sample_rate = sample_rate.get();
        async move { fetch_butterworth_response(order, cutoff, sample_rate).await }
    });

    let mag_response = response.clone();
    use_plotly_chart("butterworth-mag-plot", move || {
        build_filter_plot(mag_response.get(), PlotKind::Magnitude)
    });
    let phase_response = response.clone();
    use_plotly_chart("butterworth-phase-plot", move || {
        build_filter_plot(phase_response.get(), PlotKind::Phase)
    });

    let design_status = move || match response.get() {
        Some(Ok(data)) => data.summary,
        Some(Err(message)) => format!("Design failed: {message}"),
        None => "Loading design...".into(),
    };

    view! {
        <div class="page">
            <header class="page-header">
                <p class="eyebrow">"Filter Design"</p>
                <h1>"Digital Butterworth Explorer"</h1>
                <p>
                    "This page calls the host-side design endpoint, keeps the designed filter in SOS form on the Rust"
                    " side, and renders the sampled Bode data with Plotly in the browser."
                </p>
            </header>

            <div class="control-layout">
                <aside class="control-card">
                    <section>
                        <h2>"Design controls"</h2>
                        <p class="section-copy">
                            "Frequencies are physical angular frequencies. The Rust API handles prewarping and bilinear"
                            " mapping internally."
                        </p>

                        <div class="control-row">
                            <label for="butterworth-order">"Order"</label>
                            <output>{move || order.get().to_string()}</output>
                            <input
                                id="butterworth-order"
                                type="range"
                                min="1"
                                max="10"
                                step="1"
                                prop:value=move || order.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<usize>() {
                                        set_order.set(value.max(1));
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="butterworth-cutoff">"Cutoff"</label>
                            <output>{move || format!("{:.2} rad/s", cutoff.get())}</output>
                            <input
                                id="butterworth-cutoff"
                                type="range"
                                min="0.5"
                                max="45.0"
                                step="0.25"
                                prop:value=move || cutoff.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_cutoff.set(value);
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="butterworth-sample-rate">"Sample rate"</label>
                            <output>{move || format!("{:.2} samples/s", sample_rate.get())}</output>
                            <input
                                id="butterworth-sample-rate"
                                type="range"
                                min="4.0"
                                max="40.0"
                                step="0.5"
                                prop:value=move || sample_rate.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_sample_rate.set(value);
                                    }
                                }
                            />
                        </div>
                    </section>

                    <section>
                        <h2>"Design summary"</h2>
                        <p class="section-copy">{design_status}</p>
                    </section>
                </aside>

                <div class="plots-grid compact">
                    <article class="plot-card">
                        <div class="plot-header">
                            <div>
                                <h2>"Magnitude"</h2>
                                <p>"Sampled from the designed digital SOS filter on the unit circle."</p>
                            </div>
                        </div>
                        <div id="butterworth-mag-plot" class="plot-surface"></div>
                    </article>

                    <article class="plot-card">
                        <div class="plot-header">
                            <div>
                                <h2>"Phase"</h2>
                                <p>"Unwrapped phase over the same monotone frequency grid."</p>
                            </div>
                        </div>
                        <div id="butterworth-phase-plot" class="plot-surface"></div>
                    </article>
                </div>
            </div>
        </div>
    }
}

#[derive(Clone, Copy)]
enum PlotKind {
    Magnitude,
    Phase,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ButterworthResponse {
    bode_frequencies: Vec<f64>,
    bode_magnitude_db: Vec<f64>,
    bode_phase_deg: Vec<f64>,
    summary: String,
}

fn build_filter_plot(
    response: Option<Result<ButterworthResponse, String>>,
    kind: PlotKind,
) -> Plot {
    match response {
        Some(Ok(data)) => {
            let (title, axis_label, values) = match kind {
                PlotKind::Magnitude => (
                    "Butterworth magnitude",
                    "magnitude (dB)",
                    data.bode_magnitude_db,
                ),
                PlotKind::Phase => ("Butterworth phase", "phase (deg)", data.bode_phase_deg),
            };
            build_line_plot(
                &data.bode_frequencies,
                &values,
                title,
                "angular frequency",
                axis_label,
                true,
            )
        }
        Some(Err(message)) => empty_plot("Butterworth design", &message),
        None => empty_plot("Butterworth design", "Loading design..."),
    }
}

fn build_line_plot(
    x: &[f64],
    y: &[f64],
    title: &str,
    x_label: &str,
    y_label: &str,
    log_x: bool,
) -> Plot {
    let trace = Scatter::new(x.to_vec(), y.to_vec())
        .mode(Mode::Lines)
        .name(title);

    let mut x_axis = Axis::new().title(Title::with_text(x_label));
    if log_x {
        x_axis = x_axis.type_(AxisType::Log);
    }

    let layout = Layout::new()
        .title(Title::with_text(title))
        .x_axis(x_axis)
        .y_axis(Axis::new().title(Title::with_text(y_label)));

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);
    plot
}

fn empty_plot(title: &str, reason: &str) -> Plot {
    let layout = Layout::new()
        .title(Title::with_text(title))
        .x_axis(Axis::new().title(Title::with_text(reason)));
    let mut plot = Plot::new();
    plot.set_layout(layout);
    plot
}

async fn fetch_butterworth_response(
    order: usize,
    cutoff: f64,
    sample_rate: f64,
) -> Result<ButterworthResponse, String> {
    let url = format!(
        "{}/api/filter-design/butterworth?order={order}&cutoff={cutoff}&sample_rate={sample_rate}",
        api_base()
    );
    let response = Request::get(&url)
        .send()
        .await
        .map_err(|err| err.to_string())?;
    if !response.ok() {
        return Err(response
            .text()
            .await
            .unwrap_or_else(|_| "request failed".into()));
    }
    response.json().await.map_err(|err| err.to_string())
}

fn api_base() -> &'static str {
    option_env!("NUMERICAL_API_BASE").unwrap_or("http://127.0.0.1:3000")
}
