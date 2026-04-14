use leptos::prelude::*;

/// Generic scaffold page for feature areas that do not yet have a concrete
/// interactive example.
#[component]
pub fn PlaceholderPage(
    eyebrow: &'static str,
    title: &'static str,
    summary: &'static str,
    next_steps: &'static [&'static str],
) -> impl IntoView {
    view! {
        <div class="page">
            <header class="page-header">
                <p class="eyebrow">{eyebrow}</p>
                <h1>{title}</h1>
                <p>{summary}</p>
            </header>

            <div class="placeholder-actions">
                <article class="placeholder-card">
                    <h2>"Suggested next slice"</h2>
                    <p class="placeholder-copy">
                        "Keep new pages small and focused. Each page should expose one task-shaped workflow with a few"
                        " controls, one or more plots, and a concise textual summary of what changed."
                    </p>
                    <ul>
                        {next_steps
                            .iter()
                            .map(|step| view! { <li>{*step}</li> })
                            .collect_view()}
                    </ul>
                </article>

                <article class="placeholder-card">
                    <h2>"Implementation pattern"</h2>
                    <p class="placeholder-copy">
                        "Use Leptos signals for controls, derive the numerical outputs in ordinary Rust, and send the"
                        " resulting figure specs through the shared Plotly bridge."
                    </p>
                    <ul>
                        <li>"Add one focused page component under `src/examples/`."</li>
                        <li>"Register the page in `src/examples/mod.rs` and `src/app.rs`."</li>
                        <li>"Prefer one working plot over several half-finished panels."</li>
                    </ul>
                </article>
            </div>
        </div>
    }
}
