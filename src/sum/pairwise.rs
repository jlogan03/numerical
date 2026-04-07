use core::borrow::Borrow;
use num_traits::Float;

const PAIRWISE_STORAGE: usize = 8;

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
pub fn sum_slice<T: Float>(values: &[T], storage: &mut [T]) -> T {
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

/// Sums an iterator of values with the `Pairwise` accumulator.
#[inline]
#[must_use]
pub fn sum<I, T>(values: I) -> T
where
    I: IntoIterator,
    I::Item: Borrow<T>,
    T: Float,
{
    let mut values = values.into_iter();
    let Some(value) = values.next() else {
        return T::zero();
    };

    let mut acc = Pairwise::new(*value.borrow());
    for value in values {
        acc.add(*value.borrow());
    }

    acc.finish()
}

/// Accumulates values with a fixed-storage pairwise summation tree.
///
/// Values are first buffered in a small fresh-value storage. When that storage
/// fills, it is pairwise-reduced to one chunk sum and pushed into a second
/// storage reserved for accumulated values. When the accumulated-value storage
/// fills, it is pairwise-reduced in place and starts over with that reduced
/// chunk as its first element.
#[derive(Clone, Copy, Debug)]
pub struct Pairwise<T: Float> {
    fresh: [T; PAIRWISE_STORAGE],
    fresh_len: usize,
    accum: [T; PAIRWISE_STORAGE],
    accum_len: usize,
}

impl<T: Float> Pairwise<T> {
    /// Creates a new accumulator with an initial value.
    #[inline]
    #[must_use]
    pub fn new(value: T) -> Self {
        let mut fresh = [T::zero(); PAIRWISE_STORAGE];
        fresh[0] = value;

        Self {
            fresh,
            fresh_len: 1,
            accum: [T::zero(); PAIRWISE_STORAGE],
            accum_len: 0,
        }
    }

    /// Adds a value into the pairwise tree.
    #[inline]
    pub fn add(&mut self, value: T) {
        self.fresh[self.fresh_len] = value;
        self.fresh_len += 1;

        if self.fresh_len == PAIRWISE_STORAGE {
            self.flush_fresh();
        }
    }

    /// Finishes the accumulation and returns the final value.
    #[inline]
    #[must_use]
    pub fn finish(mut self) -> T {
        if self.fresh_len > 0 {
            if self.fresh_len == 1 && self.accum_len == 0 {
                return self.fresh[0];
            }

            let reduced = sum_inplace(&mut self.fresh[..self.fresh_len]);
            self.push_accum(reduced);
            self.fresh_len = 0;
        }

        if self.accum_len == 0 {
            T::zero()
        } else {
            sum_inplace(&mut self.accum[..self.accum_len])
        }
    }

    #[inline]
    fn flush_fresh(&mut self) {
        let reduced = sum_inplace(&mut self.fresh);
        self.fresh_len = 0;
        self.push_accum(reduced);
    }

    #[inline]
    fn push_accum(&mut self, value: T) {
        self.accum[self.accum_len] = value;
        self.accum_len += 1;

        if self.accum_len == PAIRWISE_STORAGE {
            self.accum[0] = sum_inplace(&mut self.accum);
            self.accum_len = 1;
        }
    }
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
    use super::{Pairwise, sum, sum_inplace, sum_slice};

    fn pairwise_sum(values: &[f64]) -> f64 {
        let mut acc = Pairwise::new(values[0]);

        for &value in &values[1..] {
            acc.add(value);
        }

        acc.finish()
    }

    #[test]
    fn accumulator_handles_odd_node_carries() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = pairwise_sum(&values);

        assert_eq!(sum, 15.0);
    }

    #[test]
    fn accumulator_uses_pairwise_grouping_for_tail_values() {
        let values = [1.0e16, 1.0, -1.0e16, 1.0, 1.0];
        let sum = pairwise_sum(&values);

        assert_eq!(sum, 1.0);
    }

    #[test]
    fn accumulator_handles_larger_exact_sequences() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sum = pairwise_sum(&values);

        assert_eq!(sum, 45.0);
    }

    #[test]
    fn accumulator_handles_longer_sequences_without_overflowing_tree_storage() {
        let values = [1.0f64; 100];
        let sum = sum::<_, f64>(values.iter());

        assert_eq!(sum, 100.0);
    }

    #[test]
    fn top_level_sum_accepts_owned_items() {
        let sum = sum([1.0f64, 2.0, 3.0]);

        assert_eq!(sum, 6.0);
    }

    #[test]
    fn top_level_sum_accepts_borrowed_items() {
        let values = [1.0f64, 2.0, 3.0];
        let sum = sum::<_, f64>(values.iter());

        assert_eq!(sum, 6.0);
    }

    #[test]
    fn top_level_sum_handles_empty_input() {
        let sum = sum(core::iter::empty::<f64>());

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
