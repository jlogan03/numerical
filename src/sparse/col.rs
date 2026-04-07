use super::field::Field;
use faer::{Col, Unbind};

#[inline]
pub(crate) fn col_from_slice<T: Copy>(values: &[T]) -> Col<T> {
    Col::from_fn(values.len(), |i| values[i.unbound()])
}

#[inline]
pub(crate) fn zero_col<T: Field>(len: usize) -> Col<T> {
    Col::from_fn(len, |_| T::zero_value())
}

#[inline]
pub(crate) fn col_slice<T>(col: &Col<T>) -> &[T] {
    col.try_as_col_major().unwrap().as_slice()
}

#[inline]
pub(crate) fn col_slice_mut<T>(col: &mut Col<T>) -> &mut [T] {
    col.try_as_col_major_mut().unwrap().as_slice_mut()
}

#[inline]
pub(crate) fn copy_col<T: Copy>(dst: &mut Col<T>, src: &Col<T>) {
    col_slice_mut(dst).copy_from_slice(col_slice(src));
}
