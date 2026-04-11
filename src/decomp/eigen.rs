//! Self-adjoint eigendecomposition front-ends.
//!
//! These wrappers target Hermitian / symmetric problems specifically. That is
//! a deliberate narrower scope than general nonnormal eigendecomposition, since
//! the self-adjoint case has much stronger numerical structure and clearer
//! ordering semantics.

use super::{
    DecompError, DecompInfo, DenseDecompParams, PartialEigen, SparseDecompParams,
    normalized_start_vector, orthogonality_error, partial_eigen_params, permute_col,
    permute_mat_cols, sorted_order_descending_by_abs,
};
use crate::sparse::col::{col_slice, col_slice_mut, zero_col};
use crate::sparse::compensated::{CompensatedField, norm2, sum2};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::get_global_parallelism;
use faer::matrix_free::LinOp;
use faer::matrix_free::eigen::{partial_self_adjoint_eigen, partial_self_adjoint_eigen_scratch};
use faer::{Col, Mat, MatRef, Side};
use faer_traits::math_utils::zero;
use num_traits::{Float, Zero};

fn dense_full_self_adjoint_eigen<T>(a: MatRef<'_, T>) -> Result<PartialEigen<T>, DecompError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    if a.nrows() != a.ncols() {
        return Err(DecompError::DimensionMismatch {
            which: "matrix.ncols",
            expected: a.nrows(),
            actual: a.ncols(),
        });
    }

    let full = a.self_adjoint_eigen(Side::Lower)?;
    let vectors_ref = full.U();
    let values_ref = full.S().column_vector();
    let values = Col::from_fn(values_ref.nrows(), |i| values_ref[i]);
    let order = sorted_order_descending_by_abs(values.as_ref());
    let values = permute_col(values.as_ref(), &order);
    let vectors = permute_mat_cols(vectors_ref, &order);
    let info = eigen_info::<T, _>(
        &a,
        &values,
        &vectors,
        values.nrows(),
        values.nrows(),
        &mut MemBuffer::new(StackReq::EMPTY),
    );

    Ok(PartialEigen {
        values,
        vectors,
        info,
    })
}

fn validated_sparse_target<T, A>(
    op: &A,
    params: &SparseDecompParams<T>,
) -> Result<usize, DecompError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
    A: LinOp<T>,
{
    if op.nrows() != op.ncols() {
        return Err(DecompError::DimensionMismatch {
            which: "operator.ncols",
            expected: op.nrows(),
            actual: op.ncols(),
        });
    }

    // The same restarted-window constraint that applies to sparse partial SVD
    // also applies here. Reject incompatible requests before calling into the
    // backend.
    let max_requested = if op.nrows() > 64 {
        (op.nrows() - 1) / 2
    } else {
        0
    };
    if params.n_components == 0 || params.n_components > max_requested {
        return Err(DecompError::InvalidTarget {
            requested: params.n_components,
            max: max_requested,
        });
    }

    Ok(params.n_components)
}

fn resolved_partial_dims(
    n_requested: usize,
    dim: usize,
    min_dim: Option<usize>,
    max_dim: Option<usize>,
) -> (usize, usize) {
    let max_allowed = dim - 1;
    // Keep the Krylov window within the range expected by `faer`'s restarted
    // self-adjoint eigensolver, while still honoring explicit caller
    // overrides.
    let min_dim = min_dim.unwrap_or(32usize.max(n_requested)).min(max_allowed);
    let max_dim = max_dim
        .unwrap_or(64usize.max(2 * n_requested))
        .max(min_dim)
        .min(max_allowed);
    (min_dim, max_dim)
}

fn partial_self_adjoint_eigen_impl<T, A>(
    op: &A,
    n_requested: usize,
    tol: T::Real,
    min_dim: Option<usize>,
    max_dim: Option<usize>,
    max_restarts: usize,
    start_vector: Option<&Col<T>>,
    scratch: &mut MemBuffer,
) -> Result<PartialEigen<T>, DecompError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
    A: LinOp<T>,
{
    let par = get_global_parallelism();
    let (min_dim, max_dim) = resolved_partial_dims(n_requested, op.nrows(), min_dim, max_dim);
    let start = normalized_start_vector(start_vector, op.nrows())?;
    let mut vectors = Mat::zeros(op.nrows(), n_requested);
    let mut values = vec![zero::<T>(); n_requested];
    let mut stack = MemStack::new(scratch);
    let info = partial_self_adjoint_eigen(
        vectors.as_mut(),
        &mut values,
        op,
        start.as_ref(),
        tol,
        par,
        &mut stack,
        partial_eigen_params::<T>(Some(min_dim), Some(max_dim), max_restarts),
    );

    let n_converged = info.n_converged_eigen.min(n_requested);
    // Reorder the converged Ritz pairs into the public descending-magnitude
    // order used consistently throughout this module.
    let values = Col::from_fn(n_converged, |i| values[i]);
    let vectors = Mat::from_fn(op.nrows(), n_converged, |i, j| vectors[(i, j)]);
    let order = sorted_order_descending_by_abs(values.as_ref());
    let values = permute_col(values.as_ref(), &order);
    let vectors = permute_mat_cols(vectors.as_ref(), &order);
    let info = eigen_info(op, &values, &vectors, n_requested, n_converged, scratch);

    Ok(PartialEigen {
        values,
        vectors,
        info,
    })
}

fn eigen_info<T, A>(
    op: &A,
    values: &Col<T>,
    vectors: &Mat<T>,
    n_requested: usize,
    n_converged: usize,
    scratch: &mut MemBuffer,
) -> DecompInfo<T::Real>
where
    T: CompensatedField,
    T::Real: Float + Copy,
    A: LinOp<T>,
{
    let par = get_global_parallelism();
    let mut residual_vec = zero_col::<T>(op.nrows());
    let mut max_residual_norm = T::Real::zero();

    for j in 0..values.nrows() {
        // For self-adjoint eigenpairs, one residual relation is enough:
        // `A v - lambda v`.
        let mut stack = MemStack::new(scratch);
        op.apply(
            residual_vec.as_mut().as_mat_mut(),
            vectors.col(j).as_mat(),
            par,
            &mut stack,
        );
        for (dst, &value) in col_slice_mut(&mut residual_vec)
            .iter_mut()
            .zip(vectors.col(j).try_as_col_major().unwrap().as_slice())
        {
            *dst = sum2(*dst, -(values[j] * value));
        }
        let residual = norm2(col_slice(&residual_vec));
        if residual > max_residual_norm {
            max_residual_norm = residual;
        }
    }

    DecompInfo {
        n_requested,
        n_converged,
        max_residual_norm,
        max_orthogonality_error: orthogonality_error(vectors.as_ref()),
    }
}

fn truncate_eigen<T, A>(op: &A, eig: PartialEigen<T>, n_requested: usize) -> PartialEigen<T>
where
    T: CompensatedField,
    T::Real: Float + Copy,
    A: LinOp<T>,
{
    let values = Col::from_fn(n_requested, |i| eig.values[i]);
    let vectors = Mat::from_fn(op.nrows(), n_requested, |i, j| eig.vectors[(i, j)]);
    let info = eigen_info(
        op,
        &values,
        &vectors,
        n_requested,
        n_requested,
        &mut MemBuffer::new(StackReq::EMPTY),
    );

    PartialEigen {
        values,
        vectors,
        info,
    }
}

/// Computes the scratch requirement for
/// [`sparse_self_adjoint_eigen_with_scratch`].
///
/// This is the expert entry point for callers that want to manage reusable
/// `MemBuffer` storage themselves.
pub fn sparse_self_adjoint_eigen_scratch_req<T, A>(
    op: &A,
    params: &SparseDecompParams<T>,
) -> Result<StackReq, DecompError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
    A: LinOp<T>,
{
    let n_requested = validated_sparse_target(op, params)?;
    let (min_dim, max_dim) =
        resolved_partial_dims(n_requested, op.nrows(), params.min_dim, params.max_dim);
    Ok(partial_self_adjoint_eigen_scratch(
        op,
        n_requested,
        get_global_parallelism(),
        partial_eigen_params::<T>(Some(min_dim), Some(max_dim), params.max_restarts),
    ))
}

/// Computes a sparse / matrix-free partial self-adjoint eigendecomposition with
/// caller-provided scratch space.
///
/// This is the lowest-allocation path when the same operator size will be
/// decomposed repeatedly.
pub fn sparse_self_adjoint_eigen_with_scratch<T, A>(
    op: &A,
    params: &SparseDecompParams<T>,
    scratch: &mut MemBuffer,
) -> Result<PartialEigen<T>, DecompError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
    A: LinOp<T>,
{
    let n_requested = validated_sparse_target(op, params)?;
    partial_self_adjoint_eigen_impl(
        op,
        n_requested,
        params.tol,
        params.min_dim,
        params.max_dim,
        params.max_restarts,
        params.start_vector.as_ref(),
        scratch,
    )
}

/// Computes a sparse / matrix-free partial self-adjoint eigendecomposition.
///
/// The sparse / matrix-free self-adjoint API is partial-only. Callers must
/// provide the number of dominant eigenpairs they want through
/// [`SparseDecompParams`].
pub fn sparse_self_adjoint_eigen<T, A>(
    op: &A,
    params: &SparseDecompParams<T>,
) -> Result<PartialEigen<T>, DecompError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
    A: LinOp<T>,
{
    let req = sparse_self_adjoint_eigen_scratch_req(op, params)?;
    let mut scratch = MemBuffer::new(req);
    sparse_self_adjoint_eigen_with_scratch(op, params, &mut scratch)
}

/// Computes a dense self-adjoint eigendecomposition.
///
/// `params.n_components = None` uses `faer`'s dense self-adjoint eigensolver.
/// `Some(k)` routes through the partial self-adjoint eigensolver when that is a
/// good fit for the matrix size, and otherwise falls back to a full dense
/// self-adjoint eigendecomposition followed by truncation.
pub fn dense_self_adjoint_eigen<T>(
    a: MatRef<'_, T>,
    params: &DenseDecompParams<T>,
) -> Result<PartialEigen<T>, DecompError>
where
    T: CompensatedField,
    T::Real: Float + Copy,
{
    if a.nrows() != a.ncols() {
        return Err(DecompError::DimensionMismatch {
            which: "matrix.ncols",
            expected: a.nrows(),
            actual: a.ncols(),
        });
    }

    match params.n_components {
        None => dense_full_self_adjoint_eigen(a),
        Some(0) => Ok(PartialEigen {
            values: Col::zeros(0),
            vectors: Mat::zeros(a.nrows(), 0),
            info: DecompInfo {
                n_requested: 0,
                n_converged: 0,
                max_residual_norm: T::Real::zero(),
                max_orthogonality_error: T::Real::zero(),
            },
        }),
        Some(k) if k >= a.nrows() => dense_full_self_adjoint_eigen(a),
        Some(k) if a.nrows() > 64 && 2 * k < a.nrows() => {
            let req = partial_self_adjoint_eigen_scratch(&a, k, get_global_parallelism(), {
                let (min_dim, max_dim) =
                    resolved_partial_dims(k, a.nrows(), params.min_dim, params.max_dim);
                partial_eigen_params::<T>(Some(min_dim), Some(max_dim), params.max_restarts)
            });
            let mut scratch = MemBuffer::new(req);
            partial_self_adjoint_eigen_impl(
                &a,
                k,
                params.tol,
                params.min_dim,
                params.max_dim,
                params.max_restarts,
                params.start_vector.as_ref(),
                &mut scratch,
            )
        }
        Some(k) => dense_full_self_adjoint_eigen(a).map(|eig| truncate_eigen(&a, eig, k)),
    }
}

#[cfg(test)]
mod tests {
    use super::{dense_self_adjoint_eigen, sparse_self_adjoint_eigen};
    use crate::decomp::operator::CompensatedLinOp;
    use crate::decomp::{DenseDecompParams, SparseDecompParams};
    use faer::sparse::{SparseColMat, Triplet};
    use faer::{Mat, Unbind};

    #[test]
    fn dense_self_adjoint_eigen_orders_by_magnitude() {
        let a = Mat::from_fn(3, 3, |i, j| match (i.unbound(), j.unbound()) {
            (0, 0) => -5.0,
            (1, 1) => 2.0,
            (2, 2) => 1.0,
            _ => 0.0,
        });

        let eig = dense_self_adjoint_eigen(a.as_ref(), &DenseDecompParams::<f64>::new()).unwrap();
        assert_eq!(eig.values.nrows(), 3);
        assert!(eig.values[0].abs() >= eig.values[1].abs());
        assert!(eig.values[1].abs() >= eig.values[2].abs());
    }

    #[test]
    fn sparse_self_adjoint_eigen_accepts_compensated_wrapper() {
        let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(
            70,
            70,
            &(0..70)
                .map(|i| {
                    let value = if i % 2 == 0 {
                        (70 - i) as f64
                    } else {
                        -((70 - i) as f64)
                    };
                    Triplet::new(i, i, value)
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let eig = sparse_self_adjoint_eigen(
            &CompensatedLinOp::new(matrix.as_ref()),
            &SparseDecompParams::<f64>::new(4),
        )
        .unwrap();
        assert_eq!(eig.values.nrows(), 4);
        assert!(eig.values[0].abs() >= eig.values[1].abs());
        assert!(eig.info.n_converged <= 4);
    }
}
