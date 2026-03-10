#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]



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
    /// Test initialization to a given input value
    #[test]
    fn test_initialize() {
    }
}
