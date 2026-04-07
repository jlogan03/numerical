use num_traits::Float;

/// Sums a mutable slice in place with a simple looping pairwise reduction.
#[inline]
#[must_use]
pub fn sum_inplace<T: Float>(values: &mut [T]) -> T {
    if values.is_empty() {
        return T::zero();
    }

    let mut len = values.len();
    while len > 1 {
        let pairs = len / 2;

        for i in 0..pairs {
            values[i] = values[2 * i] + values[(2 * i) + 1];
        }

        if len % 2 == 1 {
            values[pairs] = values[len - 1];
            len = pairs + 1;
        } else {
            len = pairs;
        }
    }

    values[0]
}

/// Sums a slice with a simple looping pairwise reduction using caller storage.
#[inline]
#[must_use]
pub fn sum<T: Float>(values: &[T], storage: &mut [T]) -> T {
    if values.is_empty() {
        return T::zero();
    }

    assert!(
        storage.len() >= values.len(),
        "Pairwise sum_slice storage is smaller than the input slice"
    );

    storage[..values.len()].copy_from_slice(values);
    sum_inplace(&mut storage[..values.len()])
}

/// Alias for `sum`.
#[inline]
#[must_use]
pub fn sum_slice<T: Float>(values: &[T], storage: &mut [T]) -> T {
    sum(values, storage)
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
    use super::{sum, sum_inplace, sum_slice};

    #[test]
    fn sum_handles_odd_node_carries() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut storage = [0.0; 5];
        let sum = sum(&values, &mut storage);

        assert_eq!(sum, 15.0);
    }

    #[test]
    fn sum_uses_pairwise_grouping_for_tail_values() {
        let values = [1.0e16, 1.0, -1.0e16, 1.0, 1.0];
        let mut storage = [0.0; 5];
        let sum = sum(&values, &mut storage);

        assert_eq!(sum, 1.0);
    }

    #[test]
    fn sum_handles_larger_exact_sequences() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut storage = [0.0; 9];
        let sum = sum(&values, &mut storage);

        assert_eq!(sum, 45.0);
    }

    #[test]
    fn sum_handles_longer_sequences() {
        let values = [1.0f64; 100];
        let mut storage = [0.0; 100];
        let sum = sum(&values, &mut storage);

        assert_eq!(sum, 100.0);
    }

    #[test]
    fn top_level_sum_uses_caller_storage() {
        let values = [1.0f64, 2.0, 3.0];
        let mut storage = [0.0; 3];
        let sum = sum(&values, &mut storage);

        assert_eq!(sum, 6.0);
    }

    #[test]
    fn top_level_sum_handles_empty_input() {
        let mut storage = [0.0f64; 1];
        let sum = sum(&[], &mut storage);

        assert_eq!(sum, 0.0);
    }

    #[test]
    fn sum_inplace_handles_odd_lengths() {
        let mut values = [1.0e16, 1.0, -1.0e16, 1.0, 1.0];
        let sum = sum_inplace(&mut values);

        assert_eq!(sum, 1.0);
    }

    #[test]
    fn sum_slice_uses_caller_storage() {
        let values = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mut storage = [0.0; 5];
        let sum = sum_slice(&values, &mut storage);

        assert_eq!(sum, 15.0);
    }

    #[test]
    fn sum_slice_handles_empty_input() {
        let mut storage = [0.0f64; 1];
        let sum = sum_slice(&[], &mut storage);

        assert_eq!(sum, 0.0);
    }
}
