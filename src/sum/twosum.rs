use num_traits::Float;

/// Computes an error-free addition `a + b`.
///
/// Returns `(sum, err)` such that `sum` is the rounded floating-point sum and
/// `err` is the rounding error.
#[inline]
pub fn twosum<T: Float>(a: T, b: T) -> (T, T) {
    let sum = a + b;
    let b_rounded = sum - a;
    let a_rounded = sum - b_rounded;
    let a_roundoff = a - a_rounded;
    let b_roundoff = b - b_rounded;
    let err = a_roundoff + b_roundoff;

    (sum, err)
}

/// `std` is required for tests, but is not a default feature.
/// To allow the library to compile with default features,
/// tests that require `std` are feature-gated.
/// This test makes sure we do not skip the real tests.
#[cfg(test)]
#[cfg(not(feature = "std"))]
mod test {
    #[test]
    fn require_std_for_tests() {
        panic!("`std` feature is required for tests")
    }
}

#[cfg(feature = "std")]
#[cfg(test)]
mod test {
    use super::twosum;
    use num_traits::Float;

    fn assert_twosum_residual<T: Float + core::fmt::Debug>() {
        let one = T::one();
        let quarter = T::from(0.25).unwrap();
        let tiny = T::epsilon() * quarter;

        let (sum, err) = twosum(tiny, one);

        assert_eq!(sum, one);
        assert_eq!(err, tiny);
    }

    #[test]
    fn twosum_returns_zero_error_for_exact_sum_f32() {
        let (sum, err) = twosum(1.0f32, 0.5f32);

        assert_eq!(sum, 1.5);
        assert_eq!(err, 0.0);
    }

    #[test]
    fn twosum_returns_zero_error_for_exact_sum_f64() {
        let (sum, err) = twosum(1.0f64, 0.5f64);

        assert_eq!(sum, 1.5);
        assert_eq!(err, 0.0);
    }

    #[test]
    fn twosum_captures_rounded_off_bits_without_ordering_f32() {
        assert_twosum_residual::<f32>();
    }

    #[test]
    fn twosum_captures_rounded_off_bits_without_ordering_f64() {
        assert_twosum_residual::<f64>();
    }
}
