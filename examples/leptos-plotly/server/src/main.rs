use axum::{Json, Router, extract::Query, http::StatusCode, routing::get};
use numerical::control::lti::{
    BodeData, DigitalFilterFamily, DigitalFilterSpec, FilterShape, FopdtModel,
    design_digital_filter_sos,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/api/lti/fopdt", get(fopdt_endpoint))
        .route("/api/filter-design/butterworth", get(butterworth_endpoint))
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("bind example API");
    println!("numerical example API listening on http://127.0.0.1:3000");
    axum::serve(listener, app).await.expect("serve example API");
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct FopdtQuery {
    gain: f64,
    time_constant: f64,
    delay: f64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct ButterworthQuery {
    order: usize,
    cutoff: f64,
    sample_rate: f64,
}

#[derive(Clone, Debug, Serialize)]
struct FopdtResponse {
    step_times: Vec<f64>,
    step_values: Vec<f64>,
    bode_frequencies: Vec<f64>,
    bode_magnitude_db: Vec<f64>,
    bode_phase_deg: Vec<f64>,
    dc_gain: f64,
}

#[derive(Clone, Debug, Serialize)]
struct ButterworthResponse {
    bode_frequencies: Vec<f64>,
    bode_magnitude_db: Vec<f64>,
    bode_phase_deg: Vec<f64>,
    section_count: usize,
    dc_gain: f64,
    nyquist: f64,
    summary: String,
}

async fn fopdt_endpoint(
    Query(query): Query<FopdtQuery>,
) -> Result<Json<FopdtResponse>, (StatusCode, String)> {
    if !query.gain.is_finite()
        || !query.time_constant.is_finite()
        || !query.delay.is_finite()
        || query.time_constant <= 0.0
        || query.delay < 0.0
    {
        return Err((StatusCode::BAD_REQUEST, "invalid FOPDT parameters".into()));
    }

    let model = FopdtModel {
        gain: query.gain,
        time_constant: query.time_constant,
        delay: query.delay,
    };
    let step_times = linspace(0.0, 24.0, 320);
    let step_values = model
        .step_response_values(&step_times, 0.0, 1.0, 0.0)
        .map_err(invalid_request)?;
    let bode_frequencies = logspace(-2.0, 1.7, 240);
    let bode = model
        .bode_data(&bode_frequencies)
        .map_err(invalid_request)?;

    Ok(Json(FopdtResponse {
        step_times,
        step_values,
        bode_frequencies: bode.angular_frequencies,
        bode_magnitude_db: bode.magnitude_db,
        bode_phase_deg: bode.phase_deg,
        dc_gain: model.dc_gain(),
    }))
}

async fn butterworth_endpoint(
    Query(query): Query<ButterworthQuery>,
) -> Result<Json<ButterworthResponse>, (StatusCode, String)> {
    let spec = DigitalFilterSpec::new(
        query.order,
        DigitalFilterFamily::Butterworth,
        FilterShape::Lowpass {
            cutoff: query.cutoff,
        },
        query.sample_rate,
    )
    .map_err(invalid_request)?;
    let filter = design_digital_filter_sos(&spec).map_err(invalid_request)?;

    let nyquist = query.sample_rate * core::f64::consts::PI;
    let frequencies = logspace(-1.0, (nyquist * 0.98).log10(), 260);
    let BodeData {
        angular_frequencies,
        magnitude_db,
        phase_deg,
    } = filter.bode_data(&frequencies).map_err(invalid_request)?;
    let section_count = filter.sections().len();
    let dc_gain = filter.dc_gain().map_err(invalid_request)?.re;

    Ok(Json(ButterworthResponse {
        bode_frequencies: angular_frequencies,
        bode_magnitude_db: magnitude_db,
        bode_phase_deg: phase_deg,
        section_count,
        dc_gain,
        nyquist,
        summary: format!(
            "Designed {section_count} second-order sections with Nyquist {nyquist:.2} rad/s and DC gain {dc_gain:.3}."
        ),
    }))
}

fn invalid_request(err: impl core::fmt::Display) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, err.to_string())
}

fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![start];
    }
    let step = (stop - start) / ((n - 1) as f64);
    (0..n).map(|index| start + (index as f64) * step).collect()
}

fn logspace(log10_start: f64, log10_stop: f64, n: usize) -> Vec<f64> {
    linspace(log10_start, log10_stop, n)
        .into_iter()
        .map(|value| 10.0_f64.powf(value))
        .collect()
}
