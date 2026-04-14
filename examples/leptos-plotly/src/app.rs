use crate::catalog::{EXAMPLE_GROUPS, ExampleEntry, ExampleId};
use crate::examples::{FilterDesignPage, HomePage, LtiProcessModelsPage, PlaceholderPage};
use leptos::prelude::*;

/// Top-level application shell for the interactive example browser.
#[component]
pub fn App() -> impl IntoView {
    let (selected, set_selected) = signal(ExampleId::Home);

    let current_entry = move || selected.get().entry();
    let current_page = move || {
        match selected.get() {
        ExampleId::Home => view! { <HomePage set_selected /> }.into_any(),
        ExampleId::LtiProcessModels => view! { <LtiProcessModelsPage /> }.into_any(),
        ExampleId::FilterDesign => view! { <FilterDesignPage /> }.into_any(),
        ExampleId::Estimation => view! {
            <PlaceholderPage
                eyebrow="Estimation"
                title="Estimator workbench"
                summary="Use this page for linear Kalman, steady-state observer, EKF, and UKF demos. A good first example would compare split-step and monolithic filtering on the same signal."
                next_steps=&[
                    "Start with a linear discrete Kalman example that overlays truth, measurement, and estimate.",
                    "Add a covariance or residual subplot so the demo exposes estimator confidence, not just state traces.",
                    "Then layer in EKF/UKF variants with the same plotting contract."
                ]
            />
        }
        .into_any(),
        ExampleId::Identification => view! {
            <PlaceholderPage
                eyebrow="Identification"
                title="ERA / OKID workbench"
                summary="Use this area for interactive identification from sampled data. The right pattern is a small input/output dataset generator, an identification run, and response comparison against the planted system."
                next_steps=&[
                    "Start with a simulated SISO system and noisy impulse or step data.",
                    "Plot identified versus planted Markov parameters and step responses.",
                    "Add controls for horizon length, assumed order, and noise level."
                ]
            />
        }
        .into_any(),
        ExampleId::Reduction => view! {
            <PlaceholderPage
                eyebrow="Reduction"
                title="Model-reduction workbench"
                summary="Use this page for balanced truncation and HSV exploration. The useful interactive comparison is full versus reduced response, not just a table of singular values."
                next_steps=&[
                    "Show Hankel singular values and the chosen truncation index together.",
                    "Overlay full and reduced step or Bode responses.",
                    "Expose retained order and error trends in a single screen."
                ]
            />
        }
        .into_any(),
        ExampleId::Synthesis => view! {
            <PlaceholderPage
                eyebrow="Synthesis"
                title="Controller-design workbench"
                summary="Use this area for PID, LQR, LQG, and pole-placement examples. The first slice should compare controller choices on a single plant and show the resulting loop or time-domain behavior."
                next_steps=&[
                    "Start with one plant and a closed-loop step response comparison.",
                    "Add gain or weighting controls that map directly onto the library APIs.",
                    "Keep analysis plots next to time-domain plots so tuning changes stay interpretable."
                ]
            />
        }
        .into_any(),
    }
    };

    view! {
        <div class="app-shell">
            <aside class="sidebar">
                <h1 class="sidebar-title">"numerical examples"</h1>
                <p class="sidebar-copy">
                    "A browser-hosted workbench for interactive demos built directly on top of the Rust library."
                </p>

                <button
                    class=move || nav_button_class(current_entry(), ExampleId::Home)
                    on:click=move |_| set_selected.set(ExampleId::Home)
                >
                    <span class="nav-button-title">
                        <span>"Overview"</span>
                        <StatusBadge entry=ExampleId::Home.entry() />
                    </span>
                    <p class="nav-button-copy">{ExampleId::Home.entry().summary}</p>
                </button>

                {EXAMPLE_GROUPS
                    .iter()
                    .copied()
                    .map(|group| {
                        view! {
                            <section class="nav-group">
                                <h2>{group.title}</h2>
                                <div class="nav-list">
                                    {group
                                        .entries
                                        .iter()
                                        .copied()
                                        .map(|entry| {
                                            view! {
                                                <button
                                                    class=move || nav_button_class(current_entry(), entry.id)
                                                    on:click=move |_| set_selected.set(entry.id)
                                                >
                                                    <span class="nav-button-title">
                                                        <span>{entry.title}</span>
                                                        <StatusBadge entry />
                                                    </span>
                                                    <p class="nav-button-copy">{entry.summary}</p>
                                                </button>
                                            }
                                        })
                                        .collect_view()}
                                </div>
                            </section>
                        }
                    })
                    .collect_view()}
            </aside>

            <main class="content">{current_page}</main>
        </div>
    }
}

fn nav_button_class(current: ExampleEntry, selected_id: ExampleId) -> &'static str {
    if current.id == selected_id {
        "nav-button active"
    } else {
        "nav-button"
    }
}

#[component]
fn StatusBadge(entry: ExampleEntry) -> impl IntoView {
    view! {
        <span class=format!("status-badge {}", entry.status.class_name())>
            {entry.status.label()}
        </span>
    }
}
