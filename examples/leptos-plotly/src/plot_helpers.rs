use plotly::{
    Layout, Plot, Scatter,
    common::{DashType, Line, Marker, MarkerSymbol, Mode, Title},
    layout::{Axis, AxisConstrain, AxisType},
};

/// One line-oriented Plotly trace.
#[derive(Clone)]
pub struct LineSeries {
    /// Legend label for the trace.
    pub name: String,
    /// X-axis samples.
    pub x: Vec<f64>,
    /// Y-axis samples.
    pub y: Vec<f64>,
    /// Plotly mode for the trace.
    pub mode: Mode,
    /// Overall trace opacity.
    pub opacity: f64,
    /// Line width in CSS pixels.
    pub line_width: f64,
    /// Optional explicit dash style override.
    pub dash: Option<DashType>,
    /// Optional explicit marker symbol override.
    pub marker_symbol: Option<MarkerSymbol>,
    /// Optional explicit marker size override in CSS pixels.
    pub marker_size: Option<usize>,
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
            opacity: 1.0,
            line_width: 2.0,
            dash: None,
            marker_symbol: None,
            marker_size: None,
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
            opacity: 1.0,
            line_width: 2.0,
            dash: None,
            marker_symbol: None,
            marker_size: None,
        }
    }

    /// Creates a marker-only series.
    #[must_use]
    pub fn markers(name: impl Into<String>, x: Vec<f64>, y: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            x,
            y,
            mode: Mode::Markers,
            opacity: 1.0,
            line_width: 2.0,
            dash: None,
            marker_symbol: None,
            marker_size: None,
        }
    }

    /// Overrides the trace opacity.
    #[must_use]
    pub fn with_opacity(mut self, opacity: f64) -> Self {
        self.opacity = opacity;
        self
    }

    /// Overrides the line width.
    #[must_use]
    pub fn with_line_width(mut self, line_width: f64) -> Self {
        self.line_width = line_width;
        self
    }

    /// Overrides the dash pattern.
    #[must_use]
    pub fn with_dash(mut self, dash: DashType) -> Self {
        self.dash = Some(dash);
        self
    }

    /// Overrides the marker symbol.
    #[must_use]
    pub fn with_marker_symbol(mut self, marker_symbol: MarkerSymbol) -> Self {
        self.marker_symbol = Some(marker_symbol);
        self
    }

    /// Overrides the marker size.
    #[must_use]
    pub fn with_marker_size(mut self, marker_size: usize) -> Self {
        self.marker_size = Some(marker_size);
        self
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
    for (index, trace) in series.into_iter().enumerate() {
        let line_style = Line::new()
            .color("#000000")
            .dash(trace.dash.unwrap_or_else(|| dash_style(index)))
            .width(trace.line_width);
        let marker_size = trace.marker_size.unwrap_or(11);
        let marker_style = match trace.marker_symbol {
            Some(symbol) => Marker::new()
                .color("#000000")
                .symbol(symbol)
                .size(marker_size),
            None => Marker::new().color("#000000"),
        };
        plot.add_trace(
            Scatter::new(trace.x, trace.y)
                .mode(trace.mode)
                .name(trace.name)
                .opacity(trace.opacity)
                .line(line_style)
                .marker(marker_style),
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

/// Builds a plot on the complex plane with symmetric real and imaginary axes.
#[must_use]
pub fn build_complex_plane_plot(title: &str, series: Vec<LineSeries>) -> Plot {
    let (x_radius, y_radius) = symmetric_complex_plane_half_ranges(&series);
    let mut plot = build_line_plot(title, "real", "imaginary", false, series);

    plot.set_layout(
        Layout::new()
            .title(Title::with_text(title))
            .x_axis(
                Axis::new()
                    .title(Title::with_text("real"))
                    .range(vec![-x_radius, x_radius])
                    .constrain(AxisConstrain::Range),
            )
            .y_axis(
                Axis::new()
                    .title(Title::with_text("imaginary"))
                    .range(vec![-y_radius, y_radius])
                    .constrain(AxisConstrain::Range),
            ),
    );
    plot
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

fn symmetric_complex_plane_half_ranges(series: &[LineSeries]) -> (f64, f64) {
    let x_max_abs = series
        .iter()
        .flat_map(|trace| trace.x.iter())
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let y_max_abs = series
        .iter()
        .flat_map(|trace| trace.y.iter())
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);

    (
        if x_max_abs > 0.0 {
            x_max_abs * 1.1
        } else {
            1.0
        },
        if y_max_abs > 0.0 {
            y_max_abs * 1.1
        } else {
            1.0
        },
    )
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

/// Creates an evenly spaced logarithmic grid in base-10 exponent space.
#[must_use]
pub fn logspace(log10_start: f64, log10_stop: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![10.0_f64.powf(log10_start)];
    }
    let step = (log10_stop - log10_start) / ((n - 1) as f64);
    (0..n)
        .map(|index| 10.0_f64.powf(log10_start + (index as f64) * step))
        .collect()
}
