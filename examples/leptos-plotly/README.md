# numerical interactive examples

This is a standalone `Leptos` + `plotly.rs` frontend paired with a small Rust
API server for building interactive examples on top of the `numerical` crate.

The scaffold is intentionally simple:

- `Leptos` manages app state and controls in the browser.
- `plotly.rs` builds figure specs in Rust on the frontend.
- Plot rendering is delegated to Plotly.js in the browser.
- A small host-side Rust API runs `numerical` and returns plot-ready JSON.

This split is deliberate. The current `numerical` dependency stack is not yet
fully wasm-ready because some core linear-algebra dependencies still assume a
host platform. A local API server preserves the interactive workflow without
pretending the full control stack can already run in the browser.

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

1. Start the local API server:

   ```bash
   cargo run --manifest-path server/Cargo.toml
   ```

2. In another shell, start the frontend:

   ```bash
   trunk serve
   ```

Then open the local URL reported by `trunk`. By default the frontend calls the
API at `http://127.0.0.1:3000`.

## Layout

- `src/app.rs`: overall shell, example catalog, and page switching
- `src/catalog.rs`: major-feature catalog used by the sidebar and home screen
- `src/examples/`: concrete example pages
- `src/plotly_support.rs`: thin bridge from Leptos effects to Plotly.js
- `server/`: Axum API that evaluates `numerical` on the host

## Adding a new example

1. Add a new `ExampleId` + catalog entry in `src/catalog.rs`.
2. Add a page component under `src/examples/`.
3. Register the page in `src/examples/mod.rs`.
4. Add a `match` arm in `src/app.rs`.

The existing `LTI` and `Filter Design` pages are the reference patterns for:

- control-panel signals
- derived plot generation
- calling the local Rust API from UI logic
- rendering Plotly figures from Rust
