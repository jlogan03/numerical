use crate::plot_helpers::{LineSeries, build_line_plot};
use crate::plotly_support::use_plotly_chart;
use faer::Mat;
use leptos::prelude::*;
use numerical::control::lti::DiscreteStateSpace;
use plotly::Plot;
use plotly::common::DashType;

/// Interactive controller-synthesis page using a discrete LQR design on a
/// lightly unstable second-order plant.
#[component]
pub fn SynthesisPage() -> impl IntoView {
    let (q_position, set_q_position) = signal(12.0_f64);
    let (q_velocity, set_q_velocity) = signal(1.5_f64);
    let (r_control, set_r_control) = signal(0.8_f64);

    use_plotly_chart("synthesis-position-plot", move || {
        build_synthesis_plot(
            q_position.get(),
            q_velocity.get(),
            r_control.get(),
            SynthesisPlot::Position,
        )
    });
    use_plotly_chart("synthesis-control-plot", move || {
        build_synthesis_plot(
            q_position.get(),
            q_velocity.get(),
            r_control.get(),
            SynthesisPlot::Control,
        )
    });

    let summary = move || synthesis_summary(q_position.get(), q_velocity.get(), r_control.get());

    view! {
        <div class="page">
            <header class="page-header">
                <p class="eyebrow">"Synthesis"</p>
                <h1>"Discrete LQR Regulator"</h1>
                <p>
                    "A lightly unstable sampled plant is regulated with a discrete LQR gain. The controls map"
                    " directly onto the state and control weights in the DLQR problem."
                </p>
            </header>

            <div class="control-layout">
                <aside class="control-card">
                    <section>
                        <h2>"Cost weights"</h2>
                        <p class="section-copy">
                            "Increasing state weights makes the controller more aggressive. Increasing control weight"
                            " penalizes effort and produces a slower, gentler response."
                        </p>

                        <div class="control-row">
                            <label for="lqr-q-position">"Q position"</label>
                            <output>{move || format!("{:.2}", q_position.get())}</output>
                            <input
                                id="lqr-q-position"
                                type="range"
                                min="0.5"
                                max="40.0"
                                step="0.5"
                                prop:value=move || q_position.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_q_position.set(value.max(0.5));
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="lqr-q-velocity">"Q velocity"</label>
                            <output>{move || format!("{:.2}", q_velocity.get())}</output>
                            <input
                                id="lqr-q-velocity"
                                type="range"
                                min="0.1"
                                max="12.0"
                                step="0.1"
                                prop:value=move || q_velocity.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_q_velocity.set(value.max(0.1));
                                    }
                                }
                            />
                        </div>

                        <div class="control-row">
                            <label for="lqr-r-control">"R control"</label>
                            <output>{move || format!("{:.2}", r_control.get())}</output>
                            <input
                                id="lqr-r-control"
                                type="range"
                                min="0.1"
                                max="5.0"
                                step="0.1"
                                prop:value=move || r_control.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        set_r_control.set(value.max(0.1));
                                    }
                                }
                            />
                        </div>
                    </section>

                    <section>
                        <h2>"Interpretation"</h2>
                        <p class="section-copy">
                            "The left plot compares the open-loop and closed-loop position trajectories from the same"
                            " perturbed initial condition. The right plot shows the control action generated by the"
                            " state-feedback law `u = -Kx`."
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
                                <h2>"Position trajectory"</h2>
                                <p>"Open loop versus closed loop from the same initial state."</p>
                            </div>
                        </div>
                        <div id="synthesis-position-plot" class="plot-surface"></div>
                    </article>

                    <article class="plot-card">
                        <div class="plot-header">
                            <div>
                                <h2>"Control effort"</h2>
                                <p>"State-feedback actuation commanded by the DLQR gain."</p>
                            </div>
                        </div>
                        <div id="synthesis-control-plot" class="plot-surface"></div>
                    </article>
                </div>
            </div>
        </div>
    }
}

#[derive(Clone, Copy)]
enum SynthesisPlot {
    Position,
    Control,
}

struct SynthesisDemo {
    times: Vec<f64>,
    open_loop_position: Vec<f64>,
    closed_loop_position: Vec<f64>,
    control_effort: Vec<f64>,
    gain: [f64; 2],
    spectral_radius: f64,
}

fn build_synthesis_plot(
    q_position: f64,
    q_velocity: f64,
    r_control: f64,
    which: SynthesisPlot,
) -> Plot {
    match run_synthesis_demo(q_position, q_velocity, r_control) {
        Ok(demo) => match which {
            SynthesisPlot::Position => build_line_plot(
                "Open-loop vs closed-loop position",
                "time (s)",
                "position",
                false,
                vec![
                    LineSeries::lines("open loop", demo.times.clone(), demo.open_loop_position)
                        .with_dash(DashType::Dash),
                    LineSeries::lines("closed loop", demo.times, demo.closed_loop_position)
                        .with_dash(DashType::Solid),
                ],
            ),
            SynthesisPlot::Control => build_line_plot(
                "Closed-loop control effort",
                "time (s)",
                "u[k]",
                false,
                vec![LineSeries::lines(
                    "u = -Kx",
                    demo.times,
                    demo.control_effort,
                )],
            ),
        },
        Err(message) => build_line_plot(&message, "", "", false, Vec::new()),
    }
}

fn synthesis_summary(q_position: f64, q_velocity: f64, r_control: f64) -> String {
    match run_synthesis_demo(q_position, q_velocity, r_control) {
        Ok(demo) => format!(
            "DLQR gain K = [{:.3}, {:.3}] with closed-loop spectral radius {:.3}. Lower `R` or higher state weights move the poles deeper inside the unit disk and increase control effort.",
            demo.gain[0], demo.gain[1], demo.spectral_radius,
        ),
        Err(err) => format!("Synthesis failed: {err}"),
    }
}

fn run_synthesis_demo(
    q_position: f64,
    q_velocity: f64,
    r_control: f64,
) -> Result<SynthesisDemo, String> {
    let dt = 0.12;
    let a = Mat::from_fn(2, 2, |row, col| match (row, col) {
        (0, 0) => 1.05,
        (0, 1) => dt,
        (1, 0) => 0.0,
        (1, 1) => 1.02,
        _ => 0.0,
    });
    let b = Mat::from_fn(2, 1, |row, _| if row == 0 { 0.5 * dt * dt } else { dt });
    let c = Mat::from_fn(1, 2, |_, col| if col == 0 { 1.0 } else { 0.0 });
    let d = Mat::zeros(1, 1);
    let system = DiscreteStateSpace::new(a.clone(), b.clone(), c.clone(), d.clone(), dt)
        .map_err(|err| err.to_string())?;

    let q = Mat::from_fn(2, 2, |row, col| {
        if row == col {
            if row == 0 { q_position } else { q_velocity }
        } else {
            0.0
        }
    });
    let r = Mat::from_fn(1, 1, |_, _| r_control);
    let solve = system
        .dlqr(q.as_ref(), r.as_ref())
        .map_err(|err| err.to_string())?;

    let zero_inputs = Mat::zeros(1, 80);
    let x0 = [1.0, -0.3];
    let open_sim = system
        .simulate(&x0, zero_inputs.as_ref())
        .map_err(|err| err.to_string())?;
    let closed_system =
        DiscreteStateSpace::new(solve.closed_loop_a.clone(), Mat::zeros(2, 1), c, d, dt)
            .map_err(|err| err.to_string())?;
    let closed_sim = closed_system
        .simulate(&x0, zero_inputs.as_ref())
        .map_err(|err| err.to_string())?;
    let poles = closed_system.poles().map_err(|err| err.to_string())?;
    let spectral_radius = poles.iter().map(|pole| pole.norm()).fold(0.0_f64, f64::max);

    let times = (0..zero_inputs.ncols())
        .map(|idx| (idx as f64) * dt)
        .collect::<Vec<_>>();
    let open_loop_position = (0..open_sim.outputs.ncols())
        .map(|idx| open_sim.outputs[(0, idx)])
        .collect::<Vec<_>>();
    let closed_loop_position = (0..closed_sim.outputs.ncols())
        .map(|idx| closed_sim.outputs[(0, idx)])
        .collect::<Vec<_>>();
    let control_effort = (0..zero_inputs.ncols())
        .map(|idx| {
            -(solve.gain[(0, 0)] * closed_sim.states[(0, idx)]
                + solve.gain[(0, 1)] * closed_sim.states[(1, idx)])
        })
        .collect::<Vec<_>>();

    Ok(SynthesisDemo {
        times,
        open_loop_position,
        closed_loop_position,
        control_effort,
        gain: [solve.gain[(0, 0)], solve.gain[(0, 1)]],
        spectral_radius,
    })
}
