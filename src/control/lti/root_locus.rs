//! Sampled root-locus data helpers.
//!
//! The first pass is intentionally plotting-oriented:
//!
//! - it samples closed-loop poles on a caller-supplied gain grid
//! - it tracks branches by nearest-neighbor continuation between adjacent gain
//!   samples
//! - it does not attempt exact breakaway, asymptote, or branch-point solves

use super::util::{poly_add_aligned, poly_roots};
use super::{
    ContinuousSos, ContinuousStateSpace, ContinuousTransferFunction, ContinuousZpk, DiscreteSos,
    DiscreteStateSpace, DiscreteTransferFunction, DiscreteZpk, LtiError,
};
use faer::complex::Complex;
use faer_traits::RealField;
use num_traits::Float;

/// One tracked root-locus branch across a sampled gain grid.
#[derive(Clone, Debug, PartialEq)]
pub struct RootLocusBranch<R> {
    /// Closed-loop poles assigned to this branch at each sampled gain.
    pub poles: Vec<Complex<R>>,
}

/// Sampled root-locus data for a SISO open-loop transfer.
#[derive(Clone, Debug, PartialEq)]
pub struct RootLocusData<R> {
    /// Nonnegative scalar gains used to close the unity-feedback loop.
    pub gains: Vec<R>,
    /// Closed-loop poles at each sampled gain before any branch interpretation.
    pub poles: Vec<Vec<Complex<R>>>,
    /// Heuristically tracked pole branches across the sampled gain grid.
    pub branches: Vec<RootLocusBranch<R>>,
}

impl<R> ContinuousTransferFunction<R>
where
    R: Float + Copy + RealField,
{
    /// Samples the unity negative-feedback root locus of the open-loop
    /// transfer on the supplied gain grid.
    pub fn root_locus_data(&self, gains: &[R]) -> Result<RootLocusData<R>, LtiError> {
        root_locus_from_transfer(self.numerator(), self.denominator(), gains)
    }
}

impl<R> DiscreteTransferFunction<R>
where
    R: Float + Copy + RealField,
{
    /// Samples the unity negative-feedback root locus of the open-loop
    /// transfer on the supplied gain grid.
    pub fn root_locus_data(&self, gains: &[R]) -> Result<RootLocusData<R>, LtiError> {
        root_locus_from_transfer(self.numerator(), self.denominator(), gains)
    }
}

impl<R> ContinuousZpk<R>
where
    R: Float + Copy + RealField,
{
    /// Samples the unity negative-feedback root locus on the supplied gain
    /// grid.
    pub fn root_locus_data(&self, gains: &[R]) -> Result<RootLocusData<R>, LtiError> {
        self.to_transfer_function()?.root_locus_data(gains)
    }
}

impl<R> DiscreteZpk<R>
where
    R: Float + Copy + RealField,
{
    /// Samples the unity negative-feedback root locus on the supplied gain
    /// grid.
    pub fn root_locus_data(&self, gains: &[R]) -> Result<RootLocusData<R>, LtiError> {
        self.to_transfer_function()?.root_locus_data(gains)
    }
}

impl<R> ContinuousSos<R>
where
    R: Float + Copy + RealField,
{
    /// Samples the unity negative-feedback root locus on the supplied gain
    /// grid.
    pub fn root_locus_data(&self, gains: &[R]) -> Result<RootLocusData<R>, LtiError> {
        self.to_transfer_function()?.root_locus_data(gains)
    }
}

impl<R> DiscreteSos<R>
where
    R: Float + Copy + RealField,
{
    /// Samples the unity negative-feedback root locus on the supplied gain
    /// grid.
    pub fn root_locus_data(&self, gains: &[R]) -> Result<RootLocusData<R>, LtiError> {
        self.to_transfer_function()?.root_locus_data(gains)
    }
}

impl<R> ContinuousStateSpace<R>
where
    R: Float + Copy + RealField,
{
    /// Samples the unity negative-feedback root locus of the represented SISO
    /// loop transfer.
    pub fn root_locus_data(&self, gains: &[R]) -> Result<RootLocusData<R>, LtiError> {
        self.to_transfer_function()?.root_locus_data(gains)
    }
}

impl<R> DiscreteStateSpace<R>
where
    R: Float + Copy + RealField,
{
    /// Samples the unity negative-feedback root locus of the represented SISO
    /// loop transfer.
    pub fn root_locus_data(&self, gains: &[R]) -> Result<RootLocusData<R>, LtiError> {
        self.to_transfer_function()?.root_locus_data(gains)
    }
}

fn root_locus_from_transfer<R>(
    numerator: &[R],
    denominator: &[R],
    gains: &[R],
) -> Result<RootLocusData<R>, LtiError>
where
    R: Float + Copy + RealField,
{
    validate_gain_grid(gains)?;
    let mut poles = Vec::with_capacity(gains.len());
    let mut branches = Vec::<RootLocusBranch<R>>::new();

    for &gain in gains {
        // For unity negative feedback, the closed-loop poles solve
        // `D(s) + k N(s) = 0`.
        let closed_loop_denominator = poly_add_aligned(denominator, &scale_poly(numerator, gain));
        let roots = sort_roots(poly_roots(&closed_loop_denominator)?);

        if branches.is_empty() {
            branches = roots
                .iter()
                .map(|&pole| RootLocusBranch { poles: vec![pole] })
                .collect();
        } else if branches.len() == roots.len() {
            assign_roots_to_branches(&mut branches, &roots);
        }

        poles.push(roots);
    }

    Ok(RootLocusData {
        gains: gains.to_vec(),
        poles,
        branches,
    })
}

fn validate_gain_grid<R>(gains: &[R]) -> Result<(), LtiError>
where
    R: Float + Copy + RealField,
{
    if gains
        .iter()
        .any(|&gain| !gain.is_finite() || gain < R::zero())
    {
        return Err(LtiError::InvalidSamplePoint {
            which: "root_locus_data",
        });
    }
    if gains.windows(2).any(|window| window[1] < window[0]) {
        return Err(LtiError::InvalidSampleGrid {
            which: "root_locus_data",
        });
    }
    Ok(())
}

fn scale_poly<R>(coeffs: &[R], gain: R) -> Vec<R>
where
    R: Float + Copy + RealField,
{
    coeffs.iter().map(|&value| value * gain).collect()
}

fn sort_roots<R>(mut roots: Vec<Complex<R>>) -> Vec<Complex<R>>
where
    R: Float + Copy + RealField,
{
    // Keep the per-gain spectrum deterministic before the branch tracker does
    // its nearest-neighbor assignment.
    roots.sort_by(|lhs, rhs| {
        rhs.norm()
            .partial_cmp(&lhs.norm())
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| {
                rhs.re
                    .partial_cmp(&lhs.re)
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .then_with(|| {
                rhs.im
                    .partial_cmp(&lhs.im)
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
    });
    roots
}

fn assign_roots_to_branches<R>(branches: &mut [RootLocusBranch<R>], roots: &[Complex<R>])
where
    R: Float + Copy + RealField,
{
    let mut used = vec![false; roots.len()];
    for branch in branches {
        let prev = *branch.poles.last().unwrap();
        let mut best_idx = None;
        let mut best_dist = R::infinity();
        for (idx, &root) in roots.iter().enumerate() {
            if used[idx] {
                continue;
            }
            // This is intentionally heuristic: branch continuity comes from the
            // supplied gain grid, not from an exact eigenvalue-continuation
            // solve.
            let dist = (root - prev).norm();
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(idx);
            }
        }
        if let Some(idx) = best_idx {
            used[idx] = true;
            branch.poles.push(roots[idx]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RootLocusData;
    use crate::control::lti::ContinuousTransferFunction;

    fn assert_close(lhs: f64, rhs: f64, tol: f64) {
        let err = (lhs - rhs).abs();
        assert!(err <= tol, "lhs={lhs}, rhs={rhs}, err={err}, tol={tol}");
    }

    fn assert_root_locus_close(lhs: &RootLocusData<f64>, rhs: &RootLocusData<f64>, tol: f64) {
        assert_eq!(lhs.gains, rhs.gains);
        assert_eq!(lhs.poles.len(), rhs.poles.len());
        for (lhs_roots, rhs_roots) in lhs.poles.iter().zip(rhs.poles.iter()) {
            assert_eq!(lhs_roots.len(), rhs_roots.len());
            for (&lhs_root, &rhs_root) in lhs_roots.iter().zip(rhs_roots.iter()) {
                assert!((lhs_root - rhs_root).norm() <= tol);
            }
        }
    }

    #[test]
    fn root_locus_matches_scalar_closed_loop_poles() {
        let loop_tf =
            ContinuousTransferFunction::continuous(vec![1.0], vec![1.0, 1.0, 0.0]).unwrap();
        let gains = vec![0.0, 1.0, 4.0];
        let locus = loop_tf.root_locus_data(&gains).unwrap();

        assert_eq!(locus.poles[0].len(), 2);
        assert_close(locus.poles[1][0].re, -0.5, 1.0e-8);
        assert_close(locus.poles[1][1].re, -0.5, 1.0e-8);
        assert_close(locus.poles[1][0].im.abs(), (3.0f64).sqrt() / 2.0, 1.0e-8);
        assert_close(locus.poles[2][0].re, -0.5, 1.0e-8);
        assert_close(locus.poles[2][1].re, -0.5, 1.0e-8);
        assert_close(locus.poles[2][0].im.abs(), (15.0f64).sqrt() / 2.0, 1.0e-8);
    }

    #[test]
    fn root_locus_matches_across_representations() {
        let tf = ContinuousTransferFunction::continuous(vec![2.0], vec![1.0, 3.0, 2.0]).unwrap();
        let zpk = tf.to_zpk().unwrap();
        let sos = tf.to_sos().unwrap();
        let ss = tf.to_state_space().unwrap();
        let gains = vec![0.0, 0.5, 1.0, 2.0, 5.0];

        let tf_data = tf.root_locus_data(&gains).unwrap();
        let zpk_data = zpk.root_locus_data(&gains).unwrap();
        let sos_data = sos.root_locus_data(&gains).unwrap();
        let ss_data = ss.root_locus_data(&gains).unwrap();

        assert_root_locus_close(&tf_data, &zpk_data, 1.0e-10);
        assert_root_locus_close(&tf_data, &sos_data, 1.0e-10);
        assert_root_locus_close(&tf_data, &ss_data, 1.0e-10);
    }
}
