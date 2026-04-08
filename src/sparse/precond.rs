use super::col::copy_col;
use faer::Col;
use faer::Par;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer_traits::ComplexField;

pub use faer::matrix_free::{IdentityPrecond, Precond};

#[inline]
pub(crate) fn precond_buffer<T, P>(precond: &P) -> MemBuffer
where
    T: ComplexField,
    P: Precond<T>,
{
    MemBuffer::new(precond.apply_in_place_scratch(1, Par::Seq))
}

#[inline]
pub(crate) fn apply_precond_to_col<T, P>(
    precond: &P,
    out: &mut Col<T>,
    rhs: &Col<T>,
    buffer: &mut MemBuffer,
) where
    T: ComplexField + Copy,
    P: Precond<T>,
{
    copy_col(out, rhs);
    let mut stack = MemStack::new(buffer);
    precond.apply_in_place(out.as_mat_mut(), Par::Seq, &mut stack);
}
