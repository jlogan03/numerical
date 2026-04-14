use std::time::Instant;

/// Formats millisecond timings for compact diagnostics readouts.
#[must_use]
pub fn format_millis(millis: f64) -> String {
    if millis >= 1_000.0 {
        format!("{:.2} s", millis / 1_000.0)
    } else {
        format!("{millis:.1} ms")
    }
}

/// Times one closure and returns the value plus elapsed milliseconds.
pub fn measure<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let value = f();
    let millis = start.elapsed().as_secs_f64() * 1_000.0;
    (value, millis)
}
