use plotly::{
    Layout, Plot, Scatter,
    common::{Mode, Title},
    layout::{Axis, AxisType},
};

/// One line-oriented Plotly trace.
pub struct LineSeries {
    /// Legend label for the trace.
    pub name: String,
    /// X-axis samples.
    pub x: Vec<f64>,
    /// Y-axis samples.
    pub y: Vec<f64>,
    /// Plotly mode for the trace.
    pub mode: Mode,
}

impl LineSeries {
    /// Creates a line-only series.
    #[must_use]
    pub fn lines(name: impl Into<String>, x: Vec<f64>, y: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            x,
            y,
            mode: Mode::Lines,
        }
    }

    /// Creates a line-and-marker series.
    #[must_use]
    pub fn lines_markers(name: impl Into<String>, x: Vec<f64>, y: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            x,
            y,
            mode: Mode::LinesMarkers,
        }
    }
}

/// Builds a multi-trace line plot with optional logarithmic x-axis scaling.
#[must_use]
pub fn build_line_plot(
    title: &str,
    x_label: &str,
    y_label: &str,
    log_x: bool,
    series: Vec<LineSeries>,
) -> Plot {
    let mut plot = Plot::new();
    for trace in series {
        plot.add_trace(
            Scatter::new(trace.x, trace.y)
                .mode(trace.mode)
                .name(trace.name),
        );
    }

    let mut x_axis = Axis::new().title(Title::with_text(x_label));
    if log_x {
        x_axis = x_axis.type_(AxisType::Log);
    }

    plot.set_layout(
        Layout::new()
            .title(Title::with_text(title))
            .x_axis(x_axis)
            .y_axis(Axis::new().title(Title::with_text(y_label))),
    );
    plot
}

/// Creates an evenly spaced linear grid.
#[must_use]
pub fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![start];
    }
    let step = (stop - start) / ((n - 1) as f64);
    (0..n).map(|index| start + (index as f64) * step).collect()
}
