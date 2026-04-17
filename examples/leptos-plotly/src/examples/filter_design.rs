use crate::plotly_support::use_plotly_chart;
use leptos::prelude::*;
use numerical::control::lti::{
    DigitalFilterFamily, DigitalFilterSpec, FilterShape, design_digital_filter_sos,
};
use plotly::{
    Layout, Plot, Scatter,
    common::{DashType, Line, Mode, Title},
    layout::{Axis, AxisType},
};

/// Interactive digital lowpass filter-design page.
#[component]
pub fn FilterDesignPage() -> impl IntoView {
    let (family, set_family) = signal(FilterFamilyChoice::Butterworth);
    let (order, set_order) = signal(4_usize);
    let (cutoff_fraction, set_cutoff_fraction) = signal(0.4_f64);
    let (sample_rate, set_sample_rate) = signal(20.0_f64);
    let (ripple_db, set_ripple_db) = signal(1.0_f64);

    let inputs = move || FilterDesignInputs {
        family: family.get(),
        order: order.get(),
        cutoff_fraction: cutoff_fraction.get(),
        sample_rate: sample_rate.get(),
        ripple_db: ripple_db.get(),
    };
    let design = Memo::new(move |_| run_filter_design(inputs()));

    use_plotly_chart("filter-design-mag-plot", move || {
        build_filter_plot(design.get(), FilterPlotKind::Magnitude)
    });
    use_plotly_chart("filter-design-phase-plot", move || {
        build_filter_plot(design.get(), FilterPlotKind::Phase)
    });
    use_plotly_chart("filter-design-a-sweep-plot", move || {
        build_filter_plot(design.get(), FilterPlotKind::AEntries)
    });
    use_plotly_chart("filter-design-c-sweep-plot", move || {
        build_filter_plot(design.get(), FilterPlotKind::CEntries)
    });
    use_plotly_chart("filter-design-d-sweep-plot", move || {
        build_filter_plot(design.get(), FilterPlotKind::DEntries)
    });

    let design_status = move || filter_summary(design.get());

    view! {
        <div class="page">
            <header class="page-header">
                <p class="eyebrow">"Filter Design"</p>
                <h1>"Digital Lowpass Explorer"</h1>
                <p>
                    "This page runs the digital IIR design path directly in the browser, supports Butterworth and"
                    " Chebyshev Type I lowpass filters, and sweeps the realized state-space matrices over cutoff."
                </p>
            </header>

            <div class="control-layout">
                <aside class="control-card">
                    <section>
                        <h2>"Design controls"</h2>
                        <p class="section-copy">
                            "The main response plots show one designed digital lowpass filter. The matrix sweep below"
                            " redesigns that same family over a log-spaced cutoff sweep from `1e-6 * fs` to `0.45 * fs`."
                        </p>

                        <div class="control-row">
                            <label for="filter-family">"Family"</label>
                            <select
                                id="filter-family"
                                on:change=move |ev| {
                                    set_family
                                        .set(FilterFamilyChoice::from_form_value(&event_target_value(&ev)));
                                }
                            >
                                <option
                                    value="butterworth"
                                    selected=move || family.get() == FilterFamilyChoice::Butterworth
                                >
                                    "Butterworth"
                                </option>
                                <option
                                    value="chebyshev1"
                                    selected=move || family.get() == FilterFamilyChoice::Chebyshev1
                                >
                                    "Chebyshev I"
                                </option>
                            </select>
                        </div>

                        <div class="control-row">
                            <label for="filter-order">"Order"</label>
                            <output>{move || order.get().to_string()}</output>
                            <input
                                id="filter-order"
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
                            <label for="filter-cutoff">"Cutoff / fs"</label>
                            <output>
                                {move || {
                                    let ratio = cutoff_fraction.get();
                                    format!("{ratio:.3e} fs ({:.4})", ratio * sample_rate.get())
                                }}
                            </output>
                            <input
                                id="filter-cutoff"
                                type="range"
                                min="-6.0"
                                max=move || MAX_CUTOFF_FRACTION.log10().to_string()
                                step="0.02"
                                prop:value=move || cutoff_fraction_to_log10(cutoff_fraction.get()).to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_cutoff_fraction.set(log10_to_cutoff_fraction(value));
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="filter-sample-rate">"Sample rate"</label>
                            <output>{move || format!("{:.2} samples/s", sample_rate.get())}</output>
                            <input
                                id="filter-sample-rate"
                                type="range"
                                min="4.0"
                                max="40.0"
                                step="0.5"
                                prop:value=move || sample_rate.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_sample_rate.set(value.clamp(4.0, 40.0));
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="filter-ripple">"Passband ripple"</label>
                            <output>{move || format!("{:.2} dB", ripple_db.get())}</output>
                            <input
                                id="filter-ripple"
                                type="range"
                                min="0.10"
                                max="3.00"
                                step="0.05"
                                prop:value=move || ripple_db.get().to_string()
                                prop:disabled=move || family.get() != FilterFamilyChoice::Chebyshev1
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_ripple_db.set(value.clamp(0.10, 3.00));
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
                                <h2>"Frequency response"</h2>
                                <p>"Magnitude and phase of the currently selected digital lowpass design."</p>
                            </div>
                        </div>
                        <div class="plot-subsection">
                            <div class="plot-header">
                                <div>
                                    <h2>"Magnitude"</h2>
                                    <p>"Sampled on the unit circle from the designed digital SOS filter."</p>
                                </div>
                            </div>
                            <div id="filter-design-mag-plot" class="plot-surface"></div>
                        </div>

                        <div class="plot-subsection">
                            <div class="plot-header">
                                <div>
                                    <h2>"Phase"</h2>
                                    <p>"Unwrapped phase over the same monotone frequency grid."</p>
                                </div>
                            </div>
                            <div id="filter-design-phase-plot" class="plot-surface"></div>
                        </div>
                    </article>

                    <article class="plot-card">
                        <div class="plot-header">
                            <div>
                                <h2>"State-space sweep vs cutoff"</h2>
                                <p>"Nontrivial realized `A`, `C`, and `D` entries over a log-spaced cutoff sweep."</p>
                            </div>
                        </div>

                        <div class="plot-subsection">
                            <div class="plot-header">
                                <div>
                                    <h2>"A entries"</h2>
                                    <p>"All nontrivial `A[i, j]` traces, shown without entry labels."</p>
                                </div>
                            </div>
                            <div id="filter-design-a-sweep-plot" class="plot-surface"></div>
                        </div>

                        <div class="plot-subsection">
                            <div class="plot-header">
                                <div>
                                    <h2>"C entries"</h2>
                                    <p>"All nontrivial `C[i, j]` traces, shown without entry labels."</p>
                                </div>
                            </div>
                            <div id="filter-design-c-sweep-plot" class="plot-surface"></div>
                        </div>

                        <div class="plot-subsection">
                            <div class="plot-header">
                                <div>
                                    <h2>"D entries"</h2>
                                    <p>"All nontrivial `D[i, j]` traces, shown without entry labels."</p>
                                </div>
                            </div>
                            <div id="filter-design-d-sweep-plot" class="plot-surface"></div>
                        </div>
                    </article>
                </div>
            </div>
        </div>
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FilterFamilyChoice {
    Butterworth,
    Chebyshev1,
}

impl FilterFamilyChoice {
    fn from_form_value(value: &str) -> Self {
        match value {
            "chebyshev1" => Self::Chebyshev1,
            _ => Self::Butterworth,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Butterworth => "Butterworth",
            Self::Chebyshev1 => "Chebyshev I",
        }
    }
}

#[derive(Clone, Copy)]
struct FilterDesignInputs {
    family: FilterFamilyChoice,
    order: usize,
    cutoff_fraction: f64,
    sample_rate: f64,
    ripple_db: f64,
}

#[derive(Clone, PartialEq)]
struct FilterDesignData {
    family: FilterFamilyChoice,
    order: usize,
    cutoff_fraction: f64,
    cutoff: f64,
    sample_rate: f64,
    ripple_db: f64,
    sections: usize,
    state_order: usize,
    dc_gain: f64,
    response_frequency_over_fs: Vec<f64>,
    magnitude_db: Vec<f64>,
    phase_deg: Vec<f64>,
    sweep_cutoff_over_fs: Vec<f64>,
    a_entry_series: Vec<Vec<f64>>,
    c_entry_series: Vec<Vec<f64>>,
    d_entry_series: Vec<Vec<f64>>,
}

#[derive(Clone, Copy)]
enum FilterPlotKind {
    Magnitude,
    Phase,
    AEntries,
    CEntries,
    DEntries,
}

fn build_filter_plot(result: Result<FilterDesignData, String>, kind: FilterPlotKind) -> Plot {
    match result {
        Ok(data) => match kind {
            FilterPlotKind::Magnitude => build_multiline_plot(
                &format!("{} magnitude", data.family.label()),
                "frequency / fs",
                "magnitude (dB)",
                true,
                false,
                true,
                vec![(
                    "response".to_string(),
                    data.response_frequency_over_fs,
                    data.magnitude_db,
                )],
            ),
            FilterPlotKind::Phase => build_multiline_plot(
                &format!("{} phase", data.family.label()),
                "frequency / fs",
                "phase (deg)",
                true,
                false,
                true,
                vec![(
                    "response".to_string(),
                    data.response_frequency_over_fs,
                    data.phase_deg,
                )],
            ),
            FilterPlotKind::AEntries => build_unlabeled_entry_plot(
                "A entries vs cutoff",
                &data.sweep_cutoff_over_fs,
                &data.a_entry_series,
            ),
            FilterPlotKind::CEntries => build_unlabeled_entry_plot(
                "C entries vs cutoff",
                &data.sweep_cutoff_over_fs,
                &data.c_entry_series,
            ),
            FilterPlotKind::DEntries => build_unlabeled_entry_plot(
                "D entries vs cutoff",
                &data.sweep_cutoff_over_fs,
                &data.d_entry_series,
            ),
        },
        Err(message) => build_empty_plot(&message, "", ""),
    }
}

fn run_filter_design(inputs: FilterDesignInputs) -> Result<FilterDesignData, String> {
    let order = inputs.order.max(1);
    let sample_rate = inputs.sample_rate.clamp(4.0, 40.0);
    let cutoff_fraction = clamp_cutoff_fraction(inputs.cutoff_fraction);
    let cutoff = cutoff_fraction * sample_rate;
    let ripple_db = inputs.ripple_db.clamp(0.10, 3.00);
    let spec = make_filter_spec(inputs.family, order, cutoff, sample_rate, ripple_db)?;
    let filter = design_digital_filter_sos(&spec).map_err(|err| err.to_string())?;
    let state_space = filter.to_state_space().map_err(|err| err.to_string())?;

    let response_frequencies = logspace(
        (1.0e-6 * sample_rate).log10(),
        (sample_rate * core::f64::consts::PI * 0.98).log10(),
        260,
    );
    let bode = filter
        .bode_data(&response_frequencies)
        .map_err(|err| err.to_string())?;
    let response_frequency_over_fs = bode
        .angular_frequencies
        .iter()
        .map(|omega| *omega / sample_rate)
        .collect::<Vec<_>>();

    let sweep_cutoffs = logspace(
        (MIN_CUTOFF_FRACTION * sample_rate).log10(),
        (MAX_SWEEP_CUTOFF_FRACTION * sample_rate).log10(),
        120,
    );
    let (a_entry_series, c_entry_series, d_entry_series) = collect_state_space_entry_sweeps(
        inputs.family,
        order,
        sample_rate,
        ripple_db,
        &sweep_cutoffs,
    )?;

    Ok(FilterDesignData {
        family: inputs.family,
        order,
        cutoff_fraction,
        cutoff,
        sample_rate,
        ripple_db,
        sections: filter.sections().len(),
        state_order: state_space.nstates(),
        dc_gain: filter.dc_gain().map(|value| value.re).unwrap_or(0.0),
        response_frequency_over_fs,
        magnitude_db: bode.magnitude_db,
        phase_deg: bode.phase_deg,
        sweep_cutoff_over_fs: sweep_cutoffs
            .iter()
            .map(|cutoff| *cutoff / sample_rate)
            .collect(),
        a_entry_series,
        c_entry_series,
        d_entry_series,
    })
}

fn collect_state_space_entry_sweeps(
    family: FilterFamilyChoice,
    order: usize,
    sample_rate: f64,
    ripple_db: f64,
    sweep_cutoffs: &[f64],
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>), String> {
    let first_spec = make_filter_spec(family, order, sweep_cutoffs[0], sample_rate, ripple_db)?;
    let first_filter = design_digital_filter_sos(&first_spec).map_err(|err| err.to_string())?;
    let first_ss = first_filter
        .to_state_space()
        .map_err(|err| err.to_string())?;
    let a = first_ss.a();
    let c = first_ss.c();
    let d = first_ss.d();

    let a_count = a.nrows() * a.ncols();
    let c_count = c.nrows() * c.ncols();
    let d_count = d.nrows() * d.ncols();
    let mut a_series = vec![Vec::with_capacity(sweep_cutoffs.len()); a_count];
    let mut c_series = vec![Vec::with_capacity(sweep_cutoffs.len()); c_count];
    let mut d_series = vec![Vec::with_capacity(sweep_cutoffs.len()); d_count];

    for &cutoff in sweep_cutoffs {
        let spec = make_filter_spec(family, order, cutoff, sample_rate, ripple_db)?;
        let filter = design_digital_filter_sos(&spec).map_err(|err| err.to_string())?;
        let ss = filter.to_state_space().map_err(|err| err.to_string())?;
        if ss.a().nrows() != a.nrows()
            || ss.a().ncols() != a.ncols()
            || ss.c().nrows() != c.nrows()
            || ss.c().ncols() != c.ncols()
            || ss.d().nrows() != d.nrows()
            || ss.d().ncols() != d.ncols()
        {
            return Err(
                "state-space realization dimension changed across cutoff sweep".to_string(),
            );
        }

        for row in 0..a.nrows() {
            for col in 0..a.ncols() {
                a_series[row * a.ncols() + col].push(ss.a()[(row, col)]);
            }
        }
        for row in 0..c.nrows() {
            for col in 0..c.ncols() {
                c_series[row * c.ncols() + col].push(ss.c()[(row, col)]);
            }
        }
        for row in 0..d.nrows() {
            for col in 0..d.ncols() {
                d_series[row * d.ncols() + col].push(ss.d()[(row, col)]);
            }
        }
    }

    Ok((
        retain_nontrivial_series(a_series),
        retain_nontrivial_series(c_series),
        retain_nontrivial_series(d_series),
    ))
}

fn retain_nontrivial_series(series: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    series
        .into_iter()
        .filter(|values| {
            values
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max)
                > 1.0e-10
        })
        .collect()
}

fn make_filter_spec(
    family: FilterFamilyChoice,
    order: usize,
    cutoff: f64,
    sample_rate: f64,
    ripple_db: f64,
) -> Result<DigitalFilterSpec<f64>, String> {
    let family = match family {
        FilterFamilyChoice::Butterworth => DigitalFilterFamily::Butterworth,
        FilterFamilyChoice::Chebyshev1 => DigitalFilterFamily::Chebyshev1 { ripple_db },
    };
    DigitalFilterSpec::new(order, family, FilterShape::Lowpass { cutoff }, sample_rate)
        .map_err(|err| err.to_string())
}

fn filter_summary(result: Result<FilterDesignData, String>) -> String {
    match result {
        Ok(data) => {
            let ripple_text = match data.family {
                FilterFamilyChoice::Butterworth => String::new(),
                FilterFamilyChoice::Chebyshev1 => format!(", ripple {:.2} dB", data.ripple_db),
            };
            format!(
                "{} order-{} lowpass at cutoff {:.3e} fs ({:.4}) with fs {:.2}{}. Designed {} second-order sections with {} states, DC gain {:.3}, and {} / {} / {} nontrivial A/C/D sweep traces.",
                data.family.label(),
                data.order,
                data.cutoff_fraction,
                data.cutoff,
                data.sample_rate,
                ripple_text,
                data.sections,
                data.state_order,
                data.dc_gain,
                data.a_entry_series.len(),
                data.c_entry_series.len(),
                data.d_entry_series.len(),
            )
        }
        Err(message) => format!("Design failed: {message}"),
    }
}

fn build_unlabeled_entry_plot(title: &str, x: &[f64], series: &[Vec<f64>]) -> Plot {
    if series.is_empty() {
        return build_empty_plot(
            &format!("{title}: no nontrivial entries"),
            "cutoff / fs",
            "|entry|",
        );
    }

    let traces = series
        .iter()
        .cloned()
        .map(|values| {
            (
                String::new(),
                x.to_vec(),
                values
                    .into_iter()
                    .map(|value| value.abs().max(1.0e-16))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    build_multiline_plot(title, "cutoff / fs", "|entry|", true, true, false, traces)
}

fn build_multiline_plot(
    title: &str,
    x_label: &str,
    y_label: &str,
    log_x: bool,
    log_y: bool,
    show_legend: bool,
    series: Vec<(String, Vec<f64>, Vec<f64>)>,
) -> Plot {
    let mut plot = Plot::new();
    for (index, (name, x, y)) in series.into_iter().enumerate() {
        let line = Line::new()
            .color("#000000")
            .width(if show_legend { 2.0 } else { 1.0 })
            .dash(dash_style(index));
        let mut trace = Scatter::new(x, y).mode(Mode::Lines).line(line).name(name);
        if !show_legend {
            trace = trace.show_legend(false);
        }
        plot.add_trace(trace);
    }

    let mut x_axis = Axis::new().title(Title::with_text(x_label));
    if log_x {
        x_axis = x_axis.type_(AxisType::Log);
    }
    let mut y_axis = Axis::new().title(Title::with_text(y_label));
    if log_y {
        y_axis = y_axis.type_(AxisType::Log);
    }

    plot.set_layout(
        Layout::new()
            .title(Title::with_text(title))
            .show_legend(show_legend)
            .x_axis(x_axis)
            .y_axis(y_axis),
    );
    plot
}

fn build_empty_plot(title: &str, x_label: &str, y_label: &str) -> Plot {
    build_multiline_plot(title, x_label, y_label, false, false, false, Vec::new())
}

fn dash_style(index: usize) -> DashType {
    match index % 6 {
        0 => DashType::Solid,
        1 => DashType::Dash,
        2 => DashType::Dot,
        3 => DashType::DashDot,
        4 => DashType::LongDash,
        _ => DashType::LongDashDot,
    }
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

const MIN_CUTOFF_FRACTION: f64 = 1.0e-6;
const MAX_CUTOFF_FRACTION: f64 = 0.49;
const MAX_SWEEP_CUTOFF_FRACTION: f64 = 0.45;

fn clamp_cutoff_fraction(value: f64) -> f64 {
    value.clamp(MIN_CUTOFF_FRACTION, MAX_CUTOFF_FRACTION)
}

fn cutoff_fraction_to_log10(value: f64) -> f64 {
    clamp_cutoff_fraction(value).log10()
}

fn log10_to_cutoff_fraction(value: f64) -> f64 {
    clamp_cutoff_fraction(10.0_f64.powf(value))
}

#[cfg(test)]
mod tests {
    use super::{FilterDesignInputs, FilterFamilyChoice, run_filter_design};

    #[test]
    fn filter_design_demo_runs_for_butterworth_and_chebyshev() {
        let butterworth = run_filter_design(FilterDesignInputs {
            family: FilterFamilyChoice::Butterworth,
            order: 4,
            cutoff_fraction: 0.4,
            sample_rate: 20.0,
            ripple_db: 1.0,
        })
        .unwrap();
        assert!(!butterworth.magnitude_db.is_empty());
        assert!(!butterworth.a_entry_series.is_empty());

        let chebyshev = run_filter_design(FilterDesignInputs {
            family: FilterFamilyChoice::Chebyshev1,
            order: 4,
            cutoff_fraction: 0.4,
            sample_rate: 20.0,
            ripple_db: 1.0,
        })
        .unwrap();
        assert!(!chebyshev.phase_deg.is_empty());
        assert!(!chebyshev.c_entry_series.is_empty());
    }
}
