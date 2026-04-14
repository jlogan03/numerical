use crate::plotly_support::use_plotly_chart;
use gloo_net::http::Request;
use leptos::prelude::*;
use plotly::{
    Layout, Plot, Scatter,
    common::{Mode, Title},
    layout::{Axis, AxisType},
};
use serde::{Deserialize, Serialize};

/// Interactive process-model page that exercises the host-side process-model
/// endpoints backed by `numerical`.
#[component]
pub fn LtiProcessModelsPage() -> impl IntoView {
    let (gain, set_gain) = signal(1.4_f64);
    let (time_constant, set_time_constant) = signal(3.5_f64);
    let (delay, set_delay) = signal(0.8_f64);
    let response = LocalResource::new(move || {
        let gain = gain.get();
        let time_constant = time_constant.get();
        let delay = delay.get();
        async move { fetch_fopdt_response(gain, time_constant, delay).await }
    });

    let step_response = response.clone();
    use_plotly_chart("fopdt-step-plot", move || {
        build_step_plot(step_response.get())
    });
    let mag_response = response.clone();
    use_plotly_chart("fopdt-bode-mag-plot", move || {
        build_bode_plot(mag_response.get(), BodeKind::Magnitude)
    });
    use_plotly_chart("fopdt-bode-phase-plot", move || {
        build_bode_plot(response.get(), BodeKind::Phase)
    });

    let reset = move |_| {
        set_gain.set(1.4);
        set_time_constant.set(3.5);
        set_delay.set(0.8);
    };

    view! {
        <div class="page">
            <header class="page-header">
                <p class="eyebrow">"LTI Analysis"</p>
                <h1>"Delayed Process Model Explorer"</h1>
                <p>
                    "The browser controls stay local, but the underlying step and Bode data come from the Rust API"
                    " server calling directly into `numerical::control::lti::FopdtModel`."
                </p>
            </header>

            <div class="control-layout">
                <aside class="control-card">
                    <section>
                        <h2>"Model parameters"</h2>
                        <p class="section-copy">
                            "These map directly onto `FOPDT: K exp(-Ls) / (tau s + 1)`."
                        </p>

                        <div class="control-row">
                            <label for="fopdt-gain">"Gain"</label>
                            <output>{move || format!("{:.3}", gain.get())}</output>
                            <input
                                id="fopdt-gain"
                                type="range"
                                min="0.2"
                                max="4.0"
                                step="0.05"
                                prop:value=move || gain.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_gain.set(value);
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="fopdt-tau">"Time constant"</label>
                            <output>{move || format!("{:.3} s", time_constant.get())}</output>
                            <input
                                id="fopdt-tau"
                                type="range"
                                min="0.3"
                                max="12.0"
                                step="0.1"
                                prop:value=move || time_constant.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_time_constant.set(value);
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="fopdt-delay">"Delay"</label>
                            <output>{move || format!("{:.3} s", delay.get())}</output>
                            <input
                                id="fopdt-delay"
                                type="range"
                                min="0.0"
                                max="4.0"
                                step="0.05"
                                prop:value=move || delay.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_delay.set(value);
                                    }
                                }
                            />
                        </div>
                    </section>

                    <section>
                        <h2>"Quick readout"</h2>
                        <div class="metric-grid">
                            <div class="metric">
                                <div class="metric-label">"DC gain"</div>
                                <div class="metric-value">
                                    {move || response_metric(response.get(), |data| format!("{:.3}", data.dc_gain))}
                                </div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">"Dominant lag"</div>
                                <div class="metric-value">
                                    {move || format!("{:.2} s", time_constant.get())}
                                </div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">"Dead time"</div>
                                <div class="metric-value">{move || format!("{:.2} s", delay.get())}</div>
                            </div>
                        </div>
                    </section>

                    <section>
                        <button class="action-button" on:click=reset>
                            "Reset to reference values"
                        </button>
                    </section>
                </aside>

                <div class="plots-grid">
                    <article class="plot-card">
                        <div class="plot-header">
                            <div>
                                <h2>"Step response"</h2>
                                <p>"Absolute-time delayed step returned by the Rust API."</p>
                            </div>
                        </div>
                        <div id="fopdt-step-plot" class="plot-surface"></div>
                    </article>

                    <div class="plots-grid compact">
                        <article class="plot-card">
                            <div class="plot-header">
                                <div>
                                    <h2>"Bode magnitude"</h2>
                                    <p>"Continuous-time response on a logarithmic angular-frequency grid."</p>
                                </div>
                            </div>
                            <div id="fopdt-bode-mag-plot" class="plot-surface"></div>
                        </article>

                        <article class="plot-card">
                            <div class="plot-header">
                                <div>
                                    <h2>"Bode phase"</h2>
                                    <p>"Unwrapped phase returned by the library plotting helpers."</p>
                                </div>
                            </div>
                            <div id="fopdt-bode-phase-plot" class="plot-surface"></div>
                        </article>
                    </div>
                </div>
            </div>
        </div>
    }
}

#[derive(Clone, Copy)]
enum BodeKind {
    Magnitude,
    Phase,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct FopdtResponse {
    step_times: Vec<f64>,
    step_values: Vec<f64>,
    bode_frequencies: Vec<f64>,
    bode_magnitude_db: Vec<f64>,
    bode_phase_deg: Vec<f64>,
    dc_gain: f64,
}

fn build_step_plot(response: Option<Result<FopdtResponse, String>>) -> Plot {
    match response {
        Some(Ok(data)) => build_line_plot(
            &data.step_times,
            &data.step_values,
            "FOPDT step response",
            "time",
            "output",
            false,
        ),
        Some(Err(message)) => empty_plot("FOPDT step response", &message),
        None => empty_plot("FOPDT step response", "Loading response data..."),
    }
}

fn build_bode_plot(response: Option<Result<FopdtResponse, String>>, kind: BodeKind) -> Plot {
    match response {
        Some(Ok(data)) => {
            let (title, axis_label, values) = match kind {
                BodeKind::Magnitude => (
                    "FOPDT Bode magnitude",
                    "magnitude (dB)",
                    data.bode_magnitude_db,
                ),
                BodeKind::Phase => ("FOPDT Bode phase", "phase (deg)", data.bode_phase_deg),
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
        Some(Err(message)) => empty_plot("FOPDT Bode", &message),
        None => empty_plot("FOPDT Bode", "Loading response data..."),
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

fn response_metric<T>(
    response: Option<Result<FopdtResponse, String>>,
    f: impl FnOnce(&FopdtResponse) -> T,
) -> String
where
    T: core::fmt::Display,
{
    match response {
        Some(Ok(data)) => f(&data).to_string(),
        Some(Err(_)) => "error".into(),
        None => "loading".into(),
    }
}

async fn fetch_fopdt_response(
    gain: f64,
    time_constant: f64,
    delay: f64,
) -> Result<FopdtResponse, String> {
    let url = format!(
        "{}/api/lti/fopdt?gain={gain}&time_constant={time_constant}&delay={delay}",
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
