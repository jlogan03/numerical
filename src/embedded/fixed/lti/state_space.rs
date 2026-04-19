//! Fixed-size discrete-time state-space models.

use crate::embedded::error::EmbeddedError;
use crate::embedded::fixed::linalg::{
    Matrix as MatrixStorage, Vector as VectorStorage, identity_matrix, invert_matrix, mat_add,
    mat_mul, mat_sub, mat_vec_mul, vec_add,
};
use num_traits::Float;

/// Fixed-size row-major matrix storage.
pub type Matrix<T, const R: usize, const C: usize> = MatrixStorage<T, R, C>;

/// Fixed-size column vector storage.
pub type Vector<T, const N: usize> = VectorStorage<T, N>;

/// Fixed-size discrete-time state-space realization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiscreteStateSpace<T, const NX: usize, const NU: usize, const NY: usize> {
    a: Matrix<T, NX, NX>,
    b: Matrix<T, NX, NU>,
    c: Matrix<T, NY, NX>,
    d: Matrix<T, NY, NU>,
    sample_time: T,
}

impl<T, const NX: usize, const NU: usize, const NY: usize> DiscreteStateSpace<T, NX, NU, NY>
where
    T: Float + Copy,
{
    /// Creates a discrete-time state-space realization.
    pub fn new(
        a: Matrix<T, NX, NX>,
        b: Matrix<T, NX, NU>,
        c: Matrix<T, NY, NX>,
        d: Matrix<T, NY, NU>,
        sample_time: T,
    ) -> Result<Self, EmbeddedError> {
        if !sample_time.is_finite() || sample_time <= T::zero() {
            return Err(EmbeddedError::InvalidSampleTime);
        }
        Ok(Self {
            a,
            b,
            c,
            d,
            sample_time,
        })
    }

    /// Returns the state matrix.
    #[must_use]
    pub fn a(&self) -> &Matrix<T, NX, NX> {
        &self.a
    }

    /// Returns the input matrix.
    #[must_use]
    pub fn b(&self) -> &Matrix<T, NX, NU> {
        &self.b
    }

    /// Returns the output matrix.
    #[must_use]
    pub fn c(&self) -> &Matrix<T, NY, NX> {
        &self.c
    }

    /// Returns the feedthrough matrix.
    #[must_use]
    pub fn d(&self) -> &Matrix<T, NY, NU> {
        &self.d
    }

    /// Returns the stored sample interval.
    #[must_use]
    pub fn sample_time(&self) -> T {
        self.sample_time
    }

    /// Evaluates the current output `C x + D u`.
    #[must_use]
    pub fn output(&self, x: &Vector<T, NX>, u: &Vector<T, NU>) -> Vector<T, NY> {
        vec_add(&mat_vec_mul(&self.c, x), &mat_vec_mul(&self.d, u))
    }

    /// Evaluates the next state `A x + B u`.
    #[must_use]
    pub fn next_state(&self, x: &Vector<T, NX>, u: &Vector<T, NU>) -> Vector<T, NX> {
        vec_add(&mat_vec_mul(&self.a, x), &mat_vec_mul(&self.b, u))
    }

    /// Advances one discrete timestep in place and returns the output.
    pub fn step(&self, x: &mut Vector<T, NX>, u: Vector<T, NU>) -> Vector<T, NY> {
        let y = self.output(x, &u);
        *x = self.next_state(x, &u);
        y
    }

    /// Returns the steady-state gain `G(1) = C (I - A)^-1 B + D`.
    pub fn dc_gain(&self) -> Result<Matrix<T, NY, NU>, EmbeddedError> {
        let lhs = mat_sub(&identity_matrix::<T, NX>(), &self.a);
        let solve = invert_matrix(&lhs, "state_space.dc_gain")?;
        Ok(mat_add(
            &mat_mul(&mat_mul(&self.c, &solve), &self.b),
            &self.d,
        ))
    }
}

#[cfg(feature = "std")]
impl<T, const NX: usize, const NU: usize, const NY: usize>
    TryFrom<&crate::control::lti::DiscreteStateSpace<T>> for DiscreteStateSpace<T, NX, NU, NY>
where
    T: Float + Copy + faer_traits::RealField,
{
    type Error = EmbeddedError;

    /// Converts a dynamic control-side state-space model into a fixed-size
    /// embedded representation when the dimensions match.
    fn try_from(value: &crate::control::lti::DiscreteStateSpace<T>) -> Result<Self, Self::Error> {
        if value.nstates() != NX {
            return Err(EmbeddedError::DimensionMismatch {
                which: "embedded.fixed.state_space.nstates",
                expected_rows: NX,
                expected_cols: 1,
                actual_rows: value.nstates(),
                actual_cols: 1,
            });
        }
        if value.ninputs() != NU {
            return Err(EmbeddedError::DimensionMismatch {
                which: "embedded.fixed.state_space.ninputs",
                expected_rows: NU,
                expected_cols: 1,
                actual_rows: value.ninputs(),
                actual_cols: 1,
            });
        }
        if value.noutputs() != NY {
            return Err(EmbeddedError::DimensionMismatch {
                which: "embedded.fixed.state_space.noutputs",
                expected_rows: NY,
                expected_cols: 1,
                actual_rows: value.noutputs(),
                actual_cols: 1,
            });
        }

        let mut a = [[T::zero(); NX]; NX];
        let mut b = [[T::zero(); NU]; NX];
        let mut c = [[T::zero(); NX]; NY];
        let mut d = [[T::zero(); NU]; NY];

        let mut i = 0usize;
        while i < NX {
            let mut j = 0usize;
            while j < NX {
                a[i][j] = value.a()[(i, j)];
                j += 1;
            }
            let mut j = 0usize;
            while j < NU {
                b[i][j] = value.b()[(i, j)];
                j += 1;
            }
            i += 1;
        }
        let mut i = 0usize;
        while i < NY {
            let mut j = 0usize;
            while j < NX {
                c[i][j] = value.c()[(i, j)];
                j += 1;
            }
            let mut j = 0usize;
            while j < NU {
                d[i][j] = value.d()[(i, j)];
                j += 1;
            }
            i += 1;
        }

        Self::new(a, b, c, d, value.sample_time())
    }
}
