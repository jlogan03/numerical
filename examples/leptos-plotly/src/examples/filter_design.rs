use crate::plotly_support::use_plotly_chart;
use leptos::prelude::*;
use numerical::control::lti::{
    DigitalFilterFamily, DigitalFilterSpec, FilterShape, design_digital_filter_sos,
};
use plotly::{
    Layout, Plot, Scatter,
    common::{Line, Mode, Title},
    layout::{Axis, AxisType},
};

/// Interactive digital lowpass filter-design page.
#[component]
pub fn FilterDesignPage() -> impl IntoView {
    let (order, set_order) = signal(4_usize);
    let (cutoff, set_cutoff) = signal(8.0_f64);
    let (sample_rate, set_sample_rate) = signal(20.0_f64);

    use_plotly_chart("butterworth-mag-plot", move || {
        build_filter_plot(
            order.get(),
            cutoff.get(),
            sample_rate.get(),
            PlotKind::Magnitude,
        )
    });
    use_plotly_chart("butterworth-phase-plot", move || {
        build_filter_plot(
            order.get(),
            cutoff.get(),
            sample_rate.get(),
            PlotKind::Phase,
        )
    });

    let design_status = move || filter_summary(order.get(), cutoff.get(), sample_rate.get());

    view! {
        <div class="page">
            <header class="page-header">
                <p class="eyebrow">"Filter Design"</p>
                <h1>"Digital Butterworth Explorer"</h1>
                <p>
                    "This page runs the actual digital IIR design path directly in the browser, keeps the designed"
                    " filter in SOS form, and renders the sampled Bode data with Plotly."
                </p>
            </header>

            <div class="control-layout">
                <aside class="control-card">
                    <section>
                        <h2>"Design controls"</h2>
                        <p class="section-copy">
                            "Frequencies are physical angular frequencies. The digital design layer handles prewarping"
                            " and bilinear mapping internally."
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

                <div class="plots-grid wide">
                    <article class="plot-card">
                        <div class="plot-header">
                            <div>
                                <h2>"Butterworth frequency response"</h2>
                                <p>"Magnitude and phase of the same designed digital SOS filter."</p>
                            </div>
                        </div>
                        <div class="plot-subsection">
                            <div class="plot-header">
                                <div>
                                    <h2>"Magnitude"</h2>
                                    <p>"Sampled from the designed digital SOS filter on the unit circle."</p>
                                </div>
                            </div>
                            <div id="butterworth-mag-plot" class="plot-surface"></div>
                        </div>

                        <div class="plot-subsection">
                            <div class="plot-header">
                                <div>
                                    <h2>"Phase"</h2>
                                    <p>"Unwrapped phase over the same monotone frequency grid."</p>
                                </div>
                            </div>
                            <div id="butterworth-phase-plot" class="plot-surface"></div>
                        </div>
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

fn build_filter_plot(order: usize, cutoff: f64, sample_rate: f64, kind: PlotKind) -> Plot {
    match make_filter_bode(order, cutoff, sample_rate, kind) {
        Ok((frequencies, values, axis_label, title)) => build_line_plot(
            &frequencies,
            &values,
            title,
            "angular frequency",
            axis_label,
            true,
        ),
        Err(message) => {
            let empty: Vec<f64> = Vec::new();
            build_line_plot(&empty, &empty, &message, "angular frequency", "", false)
        }
    }
}

fn make_filter_bode(
    order: usize,
    cutoff: f64,
    sample_rate: f64,
    kind: PlotKind,
) -> Result<(Vec<f64>, Vec<f64>, &'static str, &'static str), String> {
    let spec = DigitalFilterSpec::new(
        order,
        DigitalFilterFamily::Butterworth,
        FilterShape::Lowpass { cutoff },
        sample_rate,
    )
    .map_err(|err| err.to_string())?;
    let filter = design_digital_filter_sos(&spec).map_err(|err| err.to_string())?;

    let frequencies = logspace(
        -1.0,
        (sample_rate * core::f64::consts::PI * 0.98).log10(),
        260,
    );
    let bode = filter
        .bode_data(&frequencies)
        .map_err(|err| err.to_string())?;

    match kind {
        PlotKind::Magnitude => Ok((
            bode.angular_frequencies.clone(),
            bode.magnitude_db,
            "magnitude (dB)",
            "Butterworth magnitude",
        )),
        PlotKind::Phase => Ok((
            bode.angular_frequencies,
            bode.phase_deg,
            "phase (deg)",
            "Butterworth phase",
        )),
    }
}

fn filter_summary(order: usize, cutoff: f64, sample_rate: f64) -> String {
    match DigitalFilterSpec::new(
        order,
        DigitalFilterFamily::Butterworth,
        FilterShape::Lowpass { cutoff },
        sample_rate,
    ) {
        Ok(spec) => match design_digital_filter_sos(&spec) {
            Ok(filter) => format!(
                "Designed {} second-order sections with Nyquist {:.2} rad/s and DC gain {:.3}.",
                filter.sections().len(),
                sample_rate * core::f64::consts::PI,
                filter.dc_gain().map(|value| value.re).unwrap_or(0.0),
            ),
            Err(err) => format!("Design failed: {err}"),
        },
        Err(err) => format!("Invalid spec: {err}"),
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
        .line(Line::new().color("#000000"))
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

fn logspace(log10_start: f64, log10_stop: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![10.0_f64.powf(log10_start)];
    }
    let step = (log10_stop - log10_start) / ((n - 1) as f64);
    (0..n)
        .map(|index| 10.0_f64.powf(log10_start + (index as f64) * step))
        .collect()
}
