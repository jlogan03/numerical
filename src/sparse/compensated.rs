use super::field::Field;
use crate::sum::twosum::TwoSum;
use num_traits::Float;

#[derive(Clone, Copy, Debug)]
struct RealCompensatedSum<R: Float> {
    acc: Option<TwoSum<R>>,
}

impl<R: Float> Default for RealCompensatedSum<R> {
    #[inline]
    fn default() -> Self {
        Self { acc: None }
    }
}

impl<R: Float> RealCompensatedSum<R> {
    #[inline]
    fn add(&mut self, value: R) {
        match self.acc.as_mut() {
            Some(acc) => acc.add(value),
            None => self.acc = Some(TwoSum::new(value)),
        }
    }

    #[inline]
    fn finish(self) -> R {
        match self.acc {
            Some(acc) => {
                let (sum, residual) = acc.finish();
                sum + residual
            }
            None => R::zero(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CompensatedSum<T: Field> {
    real: RealCompensatedSum<T::Real>,
    imag: RealCompensatedSum<T::Real>,
}

impl<T: Field> Default for CompensatedSum<T> {
    #[inline]
    fn default() -> Self {
        Self {
            real: RealCompensatedSum::default(),
            imag: RealCompensatedSum::default(),
        }
    }
}

impl<T: Field> CompensatedSum<T> {
    #[inline]
    pub(crate) fn add(&mut self, value: T) {
        self.real.add(value.real_part());
        self.imag.add(value.imag_part());
    }

    #[inline]
    pub(crate) fn finish(self) -> T {
        T::from_parts(self.real.finish(), self.imag.finish())
    }
}

#[inline]
pub fn dotc<T: Field>(lhs: &[T], rhs: &[T]) -> T {
    assert_eq!(lhs.len(), rhs.len());

    let mut acc = CompensatedSum::<T>::default();
    for (&lhs, &rhs) in lhs.iter().zip(rhs.iter()) {
        acc.add(lhs.conj_value() * rhs);
    }

    acc.finish()
}

#[inline]
pub fn norm2_sq<T: Field>(values: &[T]) -> T::Real {
    let mut acc = RealCompensatedSum::<T::Real>::default();
    for &value in values {
        acc.add(value.abs2_value());
    }

    acc.finish()
}

#[inline]
pub fn norm2<T: Field>(values: &[T]) -> T::Real {
    norm2_sq::<T>(values).sqrt()
}

#[cfg(test)]
mod test {
    use super::dotc;
    use faer::c64;

    #[test]
    fn dotc_handles_real_cancellation() {
        let lhs = [1.0e16f64, 1.0, -1.0e16];
        let rhs = [1.0f64, 1.0, 1.0];

        assert_eq!(dotc(&lhs, &rhs), 1.0);
    }

    #[test]
    fn dotc_uses_conjugation_for_complex_inputs() {
        let lhs = [c64::new(1.0, 2.0), c64::new(-3.0, 4.0)];
        let rhs = [c64::new(5.0, -1.0), c64::new(2.0, 3.0)];
        let dot = dotc(&lhs, &rhs);

        let expected = lhs[0].conj() * rhs[0] + lhs[1].conj() * rhs[1];
        assert_eq!(dot, expected);
    }
}
