use num_traits::Float;

const PAIRWISE_LEVELS: usize = 64;
const PAIRWISE_NODE_CAPACITY: usize = 3;

/// Accumulates values with a fixed-storage pairwise summation tree.
///
/// Values are inserted as leaves into an unrealized binary tree. Each tree
/// level stores up to three nodes: two nodes to combine and one carry when the
/// level has an odd node count. This keeps the implementation allocation-free
/// while preserving pairwise grouping.
#[derive(Clone, Copy, Debug)]
pub struct Pairwise<T: Float> {
    nodes: [[T; PAIRWISE_NODE_CAPACITY]; PAIRWISE_LEVELS],
    counts: [u8; PAIRWISE_LEVELS],
}

impl<T: Float> Pairwise<T> {
    /// Creates a new accumulator with an initial value.
    #[inline]
    #[must_use]
    pub fn new(value: T) -> Self {
        let mut nodes = [[T::zero(); PAIRWISE_NODE_CAPACITY]; PAIRWISE_LEVELS];
        let mut counts = [0; PAIRWISE_LEVELS];

        nodes[0][0] = value;
        counts[0] = 1;

        Self { nodes, counts }
    }

    /// Adds a value into the pairwise tree.
    #[inline]
    pub fn add(&mut self, value: T) {
        self.push_node(0, value);
    }

    /// Finishes the accumulation and returns the final value.
    #[inline]
    #[must_use]
    pub fn finish(mut self) -> T {
        for level in 0..PAIRWISE_LEVELS - 1 {
            while self.counts[level] > 0 {
                let carry = match self.counts[level] {
                    1 => {
                        self.counts[level] = 0;
                        self.nodes[level][0]
                    }
                    2 => {
                        self.counts[level] = 0;
                        self.nodes[level][0] + self.nodes[level][1]
                    }
                    3 => {
                        let carry = self.nodes[level][0] + self.nodes[level][1];
                        self.nodes[level][0] = self.nodes[level][2];
                        self.counts[level] = 1;
                        carry
                    }
                    _ => unreachable!("pairwise levels store at most three nodes"),
                };

                self.append_node(level + 1, carry);
            }
        }

        let top = PAIRWISE_LEVELS - 1;
        while self.counts[top] > 1 {
            match self.counts[top] {
                2 => {
                    self.nodes[top][0] = self.nodes[top][0] + self.nodes[top][1];
                    self.counts[top] = 1;
                }
                3 => {
                    self.nodes[top][0] = self.nodes[top][0] + self.nodes[top][1];
                    self.nodes[top][1] = self.nodes[top][2];
                    self.counts[top] = 2;
                }
                _ => unreachable!("pairwise levels store at most three nodes"),
            }
        }

        self.nodes[top][0]
    }

    #[inline]
    fn push_node(&mut self, level: usize, value: T) {
        self.append_node(level, value);

        if self.counts[level] == PAIRWISE_NODE_CAPACITY as u8 {
            let carry = self.nodes[level][0] + self.nodes[level][1];
            self.nodes[level][0] = self.nodes[level][2];
            self.counts[level] = 1;
            self.push_node(level + 1, carry);
        }
    }

    #[inline]
    fn append_node(&mut self, level: usize, value: T) {
        assert!(
            level < PAIRWISE_LEVELS,
            "Pairwise overflowed its fixed tree storage"
        );

        let count = self.counts[level] as usize;
        debug_assert!(count < PAIRWISE_NODE_CAPACITY);
        self.nodes[level][count] = value;
        self.counts[level] += 1;
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
    use super::Pairwise;

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
}
