//! Sparse linear algebra wrappers and preconditioning utilities.

/// BiCGSTAB iterative solver.
pub mod bicgstab;
/// Sparse Cholesky factorizations.
pub mod cholesky;
pub(crate) mod col;
/// Compensated complex and real accumulation helpers.
pub mod compensated;
/// Two-sided sparse matrix equilibration.
pub mod equilibration;
/// Sparse LU factorization and iterative refinement helpers.
pub mod lu;
/// Sparse matrix-vector multiplication traits and wrappers.
pub mod matvec;
/// Preconditioner traits and concrete sparse preconditioners.
pub mod precond;
/// Schur-complement operators for block sparse systems.
pub mod schur;

pub use bicgstab::BiCGSTAB;
pub use cholesky::{SparseLdlt, SparseLdltError, SparseLlt, SparseLltError};
pub use compensated::CompensatedField;
pub use equilibration::{Equilibration, EquilibrationError, EquilibrationParams};
pub use lu::{LuRefinementParams, RefinedLuSolve, SparseLu, SparseLuError};
pub use matvec::SparseMatVec;
pub use precond::{
    BiPrecond, BlockDiagonalPrecond2, BlockPrecondError, BlockSplit2, BlockUpperTriangularPrecond2,
    DiagonalPrecond, DiagonalPrecondError, IdentityPrecond, Precond, SchurPrecond2,
};
pub use schur::{SchurComplement2, SchurComplementError};
