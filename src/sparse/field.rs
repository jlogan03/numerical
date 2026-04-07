use faer::{c32, c64};
use faer_traits::Conjugate;
use num_traits::{Float, Zero};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, Sub};

/// Field types supported by sparse iterative solvers and compensated kernels.
pub trait Field:
    Conjugate<Canonical = Self>
    + Copy
    + Debug
    + Add<Output = Self>
    + AddAssign
    + Div<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
{
    /// Underlying real scalar type used for norms and tolerances.
    type Real: Float + Debug;

    /// The additive identity.
    fn zero_value() -> Self;

    /// Builds a scalar from real and imaginary parts.
    fn from_parts(real: Self::Real, imag: Self::Real) -> Self;

    /// Returns the real part.
    fn real_part(self) -> Self::Real;

    /// Returns the imaginary part.
    fn imag_part(self) -> Self::Real;

    /// Returns the complex conjugate, or `self` for real scalars.
    fn conj_value(self) -> Self;

    /// Returns the magnitude.
    fn abs_value(self) -> Self::Real;

    /// Returns the squared magnitude.
    fn abs2_value(self) -> Self::Real;

    /// Multiplies by a real scalar.
    fn scale_real(self, rhs: Self::Real) -> Self;

    #[inline]
    fn real_zero() -> Self::Real {
        <Self::Real as Zero>::zero()
    }

    #[inline]
    fn real_epsilon() -> Self::Real {
        <Self::Real as Float>::epsilon()
    }

    /// Builds a real scalar from an `f64` literal used in configuration.
    fn real_from_f64(value: f64) -> Self::Real;
}

impl Field for f32 {
    type Real = f32;

    #[inline]
    fn zero_value() -> Self {
        0.0
    }

    #[inline]
    fn from_parts(real: Self::Real, _imag: Self::Real) -> Self {
        real
    }

    #[inline]
    fn real_part(self) -> Self::Real {
        self
    }

    #[inline]
    fn imag_part(self) -> Self::Real {
        0.0
    }

    #[inline]
    fn conj_value(self) -> Self {
        self
    }

    #[inline]
    fn abs_value(self) -> Self::Real {
        self.abs()
    }

    #[inline]
    fn abs2_value(self) -> Self::Real {
        self * self
    }

    #[inline]
    fn scale_real(self, rhs: Self::Real) -> Self {
        self * rhs
    }

    #[inline]
    fn real_from_f64(value: f64) -> Self::Real {
        value as f32
    }
}

impl Field for f64 {
    type Real = f64;

    #[inline]
    fn zero_value() -> Self {
        0.0
    }

    #[inline]
    fn from_parts(real: Self::Real, _imag: Self::Real) -> Self {
        real
    }

    #[inline]
    fn real_part(self) -> Self::Real {
        self
    }

    #[inline]
    fn imag_part(self) -> Self::Real {
        0.0
    }

    #[inline]
    fn conj_value(self) -> Self {
        self
    }

    #[inline]
    fn abs_value(self) -> Self::Real {
        self.abs()
    }

    #[inline]
    fn abs2_value(self) -> Self::Real {
        self * self
    }

    #[inline]
    fn scale_real(self, rhs: Self::Real) -> Self {
        self * rhs
    }

    #[inline]
    fn real_from_f64(value: f64) -> Self::Real {
        value
    }
}

impl Field for c32 {
    type Real = f32;

    #[inline]
    fn zero_value() -> Self {
        Self::new(0.0, 0.0)
    }

    #[inline]
    fn from_parts(real: Self::Real, imag: Self::Real) -> Self {
        Self::new(real, imag)
    }

    #[inline]
    fn real_part(self) -> Self::Real {
        self.re
    }

    #[inline]
    fn imag_part(self) -> Self::Real {
        self.im
    }

    #[inline]
    fn conj_value(self) -> Self {
        self.conj()
    }

    #[inline]
    fn abs_value(self) -> Self::Real {
        self.re.hypot(self.im)
    }

    #[inline]
    fn abs2_value(self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    fn scale_real(self, rhs: Self::Real) -> Self {
        self * Self::new(rhs, 0.0)
    }

    #[inline]
    fn real_from_f64(value: f64) -> Self::Real {
        value as f32
    }
}

impl Field for c64 {
    type Real = f64;

    #[inline]
    fn zero_value() -> Self {
        Self::new(0.0, 0.0)
    }

    #[inline]
    fn from_parts(real: Self::Real, imag: Self::Real) -> Self {
        Self::new(real, imag)
    }

    #[inline]
    fn real_part(self) -> Self::Real {
        self.re
    }

    #[inline]
    fn imag_part(self) -> Self::Real {
        self.im
    }

    #[inline]
    fn conj_value(self) -> Self {
        self.conj()
    }

    #[inline]
    fn abs_value(self) -> Self::Real {
        self.re.hypot(self.im)
    }

    #[inline]
    fn abs2_value(self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    fn scale_real(self, rhs: Self::Real) -> Self {
        self * Self::new(rhs, 0.0)
    }

    #[inline]
    fn real_from_f64(value: f64) -> Self::Real {
        value
    }
}
