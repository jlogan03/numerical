# numerical interactive examples

This is a standalone `Leptos` + `plotly.rs` client-side application for
building interactive examples directly on top of the `numerical` crate.

The scaffold is intentionally simple:

- `Leptos` manages app state and controls.
- `plotly.rs` builds figure specs in Rust.
- Plot rendering is delegated to Plotly.js in the browser.
- Each example lives in plain Rust and can call directly into `numerical`.

## Prerequisites

1. Install the wasm target:

   ```bash
   rustup target add wasm32-unknown-unknown
   ```

2. Install `trunk`:

   ```bash
   cargo install trunk
   ```

## Run

```bash
cd examples/leptos-plotly
trunk serve
```

Then open the local URL reported by `trunk`.

## Layout

- `src/app.rs`: overall shell, example catalog, and page switching
- `src/catalog.rs`: major-feature catalog used by the sidebar and home screen
- `src/examples/`: concrete example pages
- `src/plotly_support.rs`: thin bridge from Leptos effects to Plotly.js

## Adding a new example

1. Add a new `ExampleId` + catalog entry in `src/catalog.rs`.
2. Add a page component under `src/examples/`.
3. Register the page in `src/examples/mod.rs`.
4. Add a `match` arm in `src/app.rs`.

The existing `LTI` and `Filter Design` pages are the reference patterns for:

- control-panel signals
- derived plot generation
- calling `numerical` directly from UI logic
- rendering Plotly figures from Rust
