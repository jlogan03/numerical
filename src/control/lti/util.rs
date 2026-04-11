use super::error::LtiError;
use faer::Mat;
use faer::complex::Complex;
use faer_traits::RealField;
use faer_traits::math_utils::{eps, from_f64};
use num_traits::Float;

/// Validates the positive finite sample-time invariant shared by discrete LTI
/// representations.
pub(crate) fn validate_sample_time<R: Float + RealField>(sample_time: R) -> Result<(), LtiError> {
    if sample_time.is_finite() && sample_time > R::zero() {
        Ok(())
    } else {
        Err(LtiError::InvalidSampleTime)
    }
}

/// Drops leading zeros from a real polynomial coefficient vector.
///
/// The polynomial utilities normalize coefficients to descending-power form
/// without redundant leading zeros so degree calculations stay well-defined.
pub(crate) fn trim_leading_zeros<R: Float + Copy + RealField>(coeffs: &[R]) -> Vec<R> {
    let first_nz = coeffs.iter().position(|&value| value != R::zero());
    match first_nz {
        Some(idx) => coeffs[idx..].to_vec(),
        None => vec![R::zero()],
    }
}

/// Normalizes a rational polynomial pair so the denominator is monic.
///
/// This gives transfer-function coefficient form a stable canonicalization
/// rule, which simplifies equality checks and conversion round trips.
pub(crate) fn normalize_ratio<R: Float + Copy + RealField>(
    numerator: &[R],
    denominator: &[R],
) -> Result<(Vec<R>, Vec<R>), LtiError> {
    let numerator = trim_leading_zeros(numerator);
    let denominator = trim_leading_zeros(denominator);
    if numerator.is_empty() {
        return Err(LtiError::EmptyPolynomial { which: "numerator" });
    }
    if denominator.is_empty() {
        return Err(LtiError::EmptyPolynomial {
            which: "denominator",
        });
    }
    let leading = denominator[0];
    if leading == R::zero() {
        return Err(LtiError::ZeroLeadingCoefficient {
            which: "denominator",
        });
    }
    if numerator[0].is_nan() || leading.is_nan() {
        return Err(LtiError::NonFiniteResult {
            which: "normalize_ratio",
        });
    }
    let scale = leading.recip();
    Ok((
        numerator.into_iter().map(|value| value * scale).collect(),
        denominator.into_iter().map(|value| value * scale).collect(),
    ))
}

/// Evaluates a real-coefficient polynomial at a complex point via Horner's
/// method.
pub(crate) fn poly_eval<R: Float + Copy + RealField>(
    coeffs: &[R],
    point: Complex<R>,
) -> Complex<R> {
    coeffs
        .iter()
        .fold(Complex::new(R::zero(), R::zero()), |acc, &coef| {
            acc * point + Complex::new(coef, R::zero())
        })
}

/// Multiplies two real-coefficient polynomials in descending-power form.
pub(crate) fn poly_mul<R: Float + Copy + RealField>(lhs: &[R], rhs: &[R]) -> Vec<R> {
    let mut out = vec![R::zero(); lhs.len() + rhs.len() - 1];
    for (i, &lhs_value) in lhs.iter().enumerate() {
        for (j, &rhs_value) in rhs.iter().enumerate() {
            out[i + j] = out[i + j] + lhs_value * rhs_value;
        }
    }
    trim_leading_zeros(&out)
}

/// Computes the roots of a real-coefficient polynomial through its companion
/// matrix.
///
/// This is the dense reference implementation used by the first TF/ZPK/SOS
/// conversion layer. It is appropriate for modest SISO polynomials and keeps
/// the implementation on top of already-available dense eigenvalue routines.
pub(crate) fn poly_roots<R: Float + Copy + RealField>(
    coeffs: &[R],
) -> Result<Vec<Complex<R>>, LtiError> {
    let coeffs = trim_leading_zeros(coeffs);
    if coeffs.is_empty() {
        return Err(LtiError::EmptyPolynomial {
            which: "polynomial",
        });
    }
    if coeffs[0] == R::zero() {
        return Err(LtiError::ZeroLeadingCoefficient {
            which: "polynomial",
        });
    }
    if coeffs.len() == 1 {
        return Ok(Vec::new());
    }

    let lead_inv = coeffs[0].recip();
    let degree = coeffs.len() - 1;
    let mut companion = Mat::<R>::zeros(degree, degree);
    for row in 1..degree {
        companion[(row, row - 1)] = R::one();
    }
    for row in 0..degree {
        companion[(row, degree - 1)] = -(coeffs[degree - row] * lead_inv);
    }
    let mut roots = companion.eigenvalues()?;
    roots.sort_by(compare_roots);
    Ok(roots)
}

/// Rebuilds a real polynomial from a root list.
///
/// The result is only accepted when the coefficients are numerically real,
/// which in practice means the supplied roots are closed under complex
/// conjugation to within tolerance.
pub(crate) fn real_poly_from_roots<R: Float + Copy + RealField>(
    roots: &[Complex<R>],
    which: &'static str,
) -> Result<Vec<R>, LtiError> {
    let mut coeffs = vec![Complex::new(R::one(), R::zero())];
    for &root in roots {
        let mut next = vec![Complex::new(R::zero(), R::zero()); coeffs.len() + 1];
        for (i, &coef) in coeffs.iter().enumerate() {
            next[i] += coef;
            next[i + 1] -= coef * root;
        }
        coeffs = next;
    }
    complex_coeffs_to_real(&coeffs, which)
}

/// Groups roots into first- or second-order real polynomial sections.
///
/// Real roots become first-order sections padded to the SOS width, while
/// complex roots must appear in conjugate pairs so each section still has real
/// coefficients.
///
/// The padded first-order convention is:
///
/// `s - r` or `z - r`  ->  `[0, 1, -r]`
///
/// rather than `[1, -r, 0]`, which would represent `s^2 - r s`.
pub(crate) fn root_sections<R: Float + Copy + RealField>(
    roots: &[Complex<R>],
    which: &'static str,
) -> Result<Vec<[R; 3]>, LtiError> {
    let mut roots = roots.to_vec();
    roots.sort_by(compare_roots);

    let mut used = vec![false; roots.len()];
    let mut sections = Vec::new();
    let tol = root_tol(&roots);

    for i in 0..roots.len() {
        if used[i] {
            continue;
        }

        let root = roots[i];
        used[i] = true;

        if root.im.abs() <= tol {
            sections.push([R::zero(), R::one(), -root.re]);
            continue;
        }

        let conj = Complex::new(root.re, -root.im);
        let mut partner = None;
        for j in (i + 1)..roots.len() {
            if !used[j] && (roots[j] - conj).norm() <= tol {
                partner = Some(j);
                break;
            }
        }
        let Some(j) = partner else {
            return Err(LtiError::NotConjugateClosed { which });
        };
        used[j] = true;
        let pair_sum = root + roots[j];
        let pair_prod = root * roots[j];
        sections.push([R::one(), -pair_sum.re, pair_prod.re]);
    }

    Ok(sections)
}

/// Returns the multiplicative identity polynomial `1` in section storage.
///
/// This is used to pad the shorter side when a ZPK representation has more
/// pole than zero sections or vice versa.
///
/// In descending-power section storage, the identity is `[0, 0, 1]`, not
/// `[1, 0, 0]`.
pub(crate) fn identity_section<R: Float + Copy + RealField>() -> [R; 3] {
    [R::zero(), R::zero(), R::one()]
}

/// Converts numerically real complex coefficients back into a real vector.
///
/// A small tolerance is allowed so roundoff from root reconstruction does not
/// falsely reject an otherwise valid conjugate-closed polynomial.
fn complex_coeffs_to_real<R: Float + Copy + RealField>(
    coeffs: &[Complex<R>],
    which: &'static str,
) -> Result<Vec<R>, LtiError> {
    let tol = coeff_tol(coeffs);
    let mut out = Vec::with_capacity(coeffs.len());
    for &coef in coeffs {
        if coef.im.abs() > tol {
            return Err(LtiError::NotConjugateClosed { which });
        }
        out.push(coef.re);
    }
    Ok(trim_leading_zeros(&out))
}

/// Heuristic matching tolerance for pairing complex-conjugate roots.
fn root_tol<R: Float + Copy + RealField>(roots: &[Complex<R>]) -> R {
    let scale = roots
        .iter()
        .fold(R::one(), |acc: R, root: &Complex<R>| acc.max(root.norm()));
    scale * from_f64::<R>(128.0) * eps::<R>()
}

/// Heuristic tolerance for deciding whether reconstructed coefficients are
/// numerically real.
fn coeff_tol<R: Float + Copy + RealField>(coeffs: &[Complex<R>]) -> R {
    let scale = coeffs
        .iter()
        .fold(R::one(), |acc: R, coeff: &Complex<R>| acc.max(coeff.norm()));
    scale * from_f64::<R>(128.0) * eps::<R>()
}

/// Stable sort order for roots: descending magnitude, then real part, then
/// imaginary part.
///
/// This keeps conversions and tests deterministic without imposing a stronger
/// mathematical meaning on root ordering than the underlying eigenvalue solve
/// actually provides.
fn compare_roots<R: Float + Copy + RealField>(
    lhs: &Complex<R>,
    rhs: &Complex<R>,
) -> core::cmp::Ordering {
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
}
