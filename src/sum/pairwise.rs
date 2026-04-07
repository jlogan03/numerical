use core::borrow::Borrow;
use num_traits::Float;

const PAIRWISE_LEVELS: usize = 64;

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
/// Values are inserted as leaves into an unrealized binary tree. Internally,
/// this stores the active frontier of that tree: one realized partial sum per
/// level, plus a bitmask tracking which levels are occupied. This keeps the
/// implementation allocation-free while preserving pairwise grouping.
#[derive(Clone, Copy, Debug)]
pub struct Pairwise<T: Float> {
    partials: [T; PAIRWISE_LEVELS],
    occupied: u64,
}

impl<T: Float> Pairwise<T> {
    /// Creates a new accumulator with an initial value.
    #[inline]
    #[must_use]
    pub fn new(value: T) -> Self {
        let mut partials = [T::zero(); PAIRWISE_LEVELS];
        partials[0] = value;

        Self {
            partials,
            occupied: 1,
        }
    }

    /// Adds a value into the pairwise tree.
    #[inline]
    pub fn add(&mut self, value: T) {
        let mut carry = value;

        for level in 0..PAIRWISE_LEVELS {
            let bit = 1_u64 << level;

            if (self.occupied & bit) == 0 {
                self.partials[level] = carry;
                self.occupied |= bit;
                return;
            }

            carry = self.partials[level] + carry;
            self.occupied &= !bit;
        }

        panic!("Pairwise overflowed its fixed tree storage");
    }

    /// Finishes the accumulation and returns the final value.
    #[inline]
    #[must_use]
    pub fn finish(self) -> T {
        let mut acc = T::zero();
        let mut started = false;

        for level in (0..PAIRWISE_LEVELS).rev() {
            let bit = 1_u64 << level;
            if (self.occupied & bit) == 0 {
                continue;
            }

            if started {
                acc = acc + self.partials[level];
            } else {
                acc = self.partials[level];
                started = true;
            }
        }

        acc
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
    use super::{Pairwise, sum};

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
}
