use std::f64::consts::TAU;

/// Smooth deterministic colored signal used for process-disturbance examples.
#[must_use]
pub fn colored_signal(step: usize, phase: f64) -> f64 {
    let k = step as f64;
    0.8 * (0.17 * k + phase).sin() + 0.35 * (0.07 * k + 0.5 * phase).cos()
}

/// Deterministic Gaussian-like jitter based on a fixed hashed stream.
///
/// This uses Box-Muller with a stable integer hash instead of runtime RNG so
/// the interactive examples remain repeatable across runs and screenshots.
#[must_use]
pub fn gaussianish_signal(step: usize, stream: u64) -> f64 {
    let u1 = unit_uniform(step * 2, stream).clamp(1.0e-12, 1.0 - 1.0e-12);
    let u2 =
        unit_uniform(step * 2 + 1, stream ^ 0x9e37_79b9_7f4a_7c15).clamp(1.0e-12, 1.0 - 1.0e-12);
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

fn unit_uniform(index: usize, stream: u64) -> f64 {
    let mut x = (index as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(stream.wrapping_mul(0xbf58_476d_1ce4_e5b9))
        .wrapping_add(0x94d0_49bb_1331_11eb);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    ((x >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}
