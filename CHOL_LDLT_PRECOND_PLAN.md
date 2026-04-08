# Cholesky / LDLT Preconditioner Plan

## Goal

Add reusable sparse symmetric-factorization wrappers that can be used in two
roles:

- as direct solvers for symmetric or Hermitian sparse systems
- as reusable `Precond` implementations for iterative solvers

The immediate motivation is to extend the current sparse-solver surface beyond
unsymmetric LU in a way that:

- preserves exact factorization quality
- supports same-pattern numeric refactorization
- matches the staged workflow we already use for [`SparseLu`](src/sparse/lu.rs)

This is useful on its own for symmetric problems and is also an important
building block for later block and Schur-complement machinery.

## Why Both LLT And LDLT

We should plan both variants together, but implement them as distinct concrete
types.

`LLT` is the natural exact factorization for:

- symmetric positive definite real systems
- Hermitian positive definite complex systems

`LDLT` is the more flexible sibling for:

- symmetric or Hermitian systems that are not guaranteed positive definite
- cases where diagonal regularization is needed to make the factorization usable

So the split should be:

- [`SparseLlt`](src/sparse/cholesky.rs): exact positive-definite path
- [`SparseLdlt`](src/sparse/cholesky.rs): exact indefinite or regularized path

The user-facing story is then simple:

- use `SparseLlt` when the matrix class guarantees definiteness
- use `SparseLdlt` when that guarantee is unavailable or known to fail

## Recommendation

Add a new sparse module:

- [`src/sparse/cholesky.rs`](src/sparse/cholesky.rs)

and re-export the public wrappers from:

- [`src/sparse/mod.rs`](src/sparse/mod.rs)

That module should wrap `faer`'s symbolic and numeric sparse Cholesky APIs:

- `factorize_symbolic_cholesky`
- `SymbolicCholesky<I>`
- `factorize_numeric_llt`
- `factorize_numeric_ldlt`
- `LltRef<'_, I, T>`
- `LdltRef<'_, I, T>`

The wrapper should follow the same staged pattern as `SparseLu`:

1. analyze the symbolic structure once
2. store an owned numeric value buffer for the factors
3. refactor numerically when values change but the pattern stays fixed
4. expose both direct-solve and `Precond` application paths

## Why A Wrapper Is Still Useful

`faer` already has the factorization kernels, ordering choices, and solve
paths. We still want a local wrapper because we need project-local behavior:

- strict same-pattern refactorization checks
- a stable public API aligned with our LU wrapper
- easy direct-solve and preconditioner reuse from the same stored factors
- room for compensated residual refinement later if needed

This is the same rationale that led us to add [`SparseLu`](src/sparse/lu.rs)
instead of wiring `faer` calls directly into every solver.

## Matrix Class And Storage Assumptions

The first implementation should be CSC-only and square-only.

That matches `faer`'s sparse-direct APIs and keeps the first invariants
defensible.

The first implementation should also require the caller to choose and maintain
one stored triangular half consistently:

- lower-triangular storage with `Side::Lower`, or
- upper-triangular storage with `Side::Upper`

The wrapper should not try to guess or canonicalize that choice.

Why:

- symbolic Cholesky depends on the chosen triangle
- same-pattern numeric refactorization is only simple if the stored half and
  entry order are identical across updates

## Public API

Suggested public types:

```rust
pub struct SparseLlt<I: Index, T: ComplexField> {
    symbolic: SymbolicCholesky<I>,
    l_values: Vec<T>,
    pattern_col_ptr: Vec<I>,
    pattern_row_idx: Vec<I>,
    side: Side,
    ready: bool,
}

pub struct SparseLdlt<I: Index, T: ComplexField> {
    symbolic: SymbolicCholesky<I>,
    l_values: Vec<T>,
    pattern_col_ptr: Vec<I>,
    pattern_row_idx: Vec<I>,
    side: Side,
    ready: bool,
}
```

Suggested error split:

```rust
pub enum SparseLltError {
    NonSquare { nrows: usize, ncols: usize },
    PatternMismatch,
    NotReady,
    Symbolic(FaerError),
    Numeric(LltError),
}

pub enum SparseLdltError {
    NonSquare { nrows: usize, ncols: usize },
    PatternMismatch,
    NotReady,
    Symbolic(FaerError),
    Numeric(LdltError),
}
```

Suggested staged API:

```rust
impl<I: Index, T: ComplexField> SparseLlt<I, T> {
    pub fn analyze<ViewT>(
        matrix: SparseColMatRef<'_, I, ViewT>,
        side: Side,
        ordering: SymmetricOrdering<'_, I>,
        symbolic_params: CholeskySymbolicParams<'_>,
    ) -> Result<Self, SparseLltError>
    where
        ViewT: Conjugate<Canonical = T>;

    pub fn factorize<ViewT>(
        matrix: SparseColMatRef<'_, I, ViewT>,
        side: Side,
        ordering: SymmetricOrdering<'_, I>,
        symbolic_params: CholeskySymbolicParams<'_>,
        regularization: LltRegularization<T::Real>,
        par: Par,
        numeric_params: Spec<LltParams, T>,
    ) -> Result<Self, SparseLltError>
    where
        ViewT: Conjugate<Canonical = T>;

    pub fn refactor<ViewT>(
        &mut self,
        matrix: SparseColMatRef<'_, I, ViewT>,
        regularization: LltRegularization<T::Real>,
        par: Par,
        numeric_params: Spec<LltParams, T>,
    ) -> Result<(), SparseLltError>
    where
        ViewT: Conjugate<Canonical = T>;

    pub fn solve_in_place(&self, rhs: &mut Col<T>) -> Result<(), SparseLltError>;
}
```

`SparseLdlt` should mirror that shape, but with:

- `LdltRegularization<'_, T::Real>`
- `Spec<LdltParams, T>`
- `SparseLdltError`

## Important Difference From LU

`faer`'s sparse Cholesky APIs do not expose a `Numeric...` owner type like
`NumericLu<I, T>`.

Instead, numeric factorization fills a caller-owned numeric value buffer and
returns a lightweight `LltRef` or `LdltRef` borrowing:

- the symbolic structure
- the numeric value slice

That means our wrapper should own the numeric value storage directly.

Recommended pattern:

- store `symbolic: SymbolicCholesky<I>`
- allocate `l_values` to `symbolic.len_val()`
- on each numeric factorization, pass `&mut l_values` into `faer`
- reconstruct a short-lived `LltRef` or `LdltRef` when solving

This is the main design difference from the existing LU wrapper.

## Direct Solve And Preconditioner Roles

Both wrappers should expose direct solve methods and implement `Precond<T>`.

The direct solve case is the primary mode when:

- the stored factors correspond to the current matrix
- we have one or a few right-hand sides

The preconditioner role matters when:

- the factors are lagged from a nearby matrix
- we want to reuse them across many iterative solves
- a block or Schur-complement preconditioner wants an exact solve on a
  symmetric sub-block

As with `SparseLu`, exact current-matrix factors are usually better used as a
direct solve than as a preconditioner. The preconditioner API is still worth
implementing because lagged and compositional uses are real.

## Pattern Compatibility Policy

Use the same strict compatibility rule as `SparseLu`:

- same `nrows`
- same `ncols`
- same stored `Side`
- identical CSC `col_ptr`
- identical CSC `row_idx`

If any of those differ, `refactor` should return `PatternMismatch`.

This avoids hidden sorting or structural normalization and keeps the symbolic
invariant simple.

## Ordering And Regularization Policy

Expose `faer`'s ordering and numeric parameters directly at the wrapper
boundary.

That means:

- `SymmetricOrdering<'_, I>` during symbolic analysis
- `LltRegularization<T::Real>` or `LdltRegularization<'_, T::Real>` during
  numeric factorization
- `Spec<LltParams, T>` or `Spec<LdltParams, T>` during numeric factorization

Why:

- we do not want to hide useful ordering choices like `Identity`, `Amd`, or a
  custom permutation
- `LDLT` regularization policy is numerically important and should stay caller
  controlled

For the first version, the wrapper should forward these parameters rather than
inventing a second project-local parameter layer.

## Suggested Internal Helpers

Factor out common private helpers inside [`src/sparse/cholesky.rs`](src/sparse/cholesky.rs):

- square-matrix validation
- strict CSC pattern matching
- shared solve scratch allocation
- short-lived `LltRef` / `LdltRef` reconstruction from stored symbolic and
  numeric buffers

The commonality is real, but the public types should stay separate because
their numeric failure modes and matrix-class assumptions are different.

## Phase 1: Core Wrappers

- add `SparseLlt`
- add `SparseLdlt`
- add public error types
- add `analyze`, `factorize`, `refactor`, `is_ready`, `nrows`, `ncols`
- add direct `solve_in_place` and `solve_rhs`

At the end of this phase, we should have direct sparse symmetric solves and
same-pattern numeric refactorization.

## Phase 2: Preconditioner Integration

- implement `Precond<T>` for `SparseLlt`
- implement `Precond<T>` for `SparseLdlt`
- add bridge tests with [`BiCGSTAB`](src/sparse/bicgstab.rs)

This gives us exact or lagged symmetric-factorization preconditioners using the
same solver hook as diagonal and LU-based preconditioners.

## Phase 3: Compensated Direct Solve Path

After the uncompensated direct solve works, consider a compensated refinement
path analogous to [`SparseLu`](src/sparse/lu.rs):

- ordinary factor solve from `faer`
- compensated residual recomputation with sparse matvec
- iterative correction solve with the stored factors
- compensated solution update

This should be a second step, not part of the initial wrapper.

## Tests

Add tests for:

- real SPD factorize and solve through `SparseLlt`
- complex Hermitian positive-definite factorize and solve through `SparseLlt`
- real symmetric indefinite or regularized solve through `SparseLdlt`
- same-pattern numeric refactorization for both wrappers
- strict pattern mismatch rejection
- `Precond<T>` application matching direct solve on a single right-hand side
- lagged-factor use inside `BiCGSTAB`

## Out Of Scope For The First Pass

- CSR-facing wrappers
- transpose or adjoint preconditioner traits
- automatic matrix-class detection
- dense fallback factorization
- block or Schur composition in the first implementation

Those should build on these wrappers, not complicate them from the start.

## Control-Systems Relevance

These wrappers are useful beyond "symmetric linear algebra for its own sake."

They are the symmetric-block ingredients needed for later work on:

- structured block preconditioners
- Schur-complement methods
- control and estimation systems with symmetric sub-blocks
- Lyapunov-adjacent workflows where symmetric positive-definite solves appear
  inside larger iterations

So this is both a direct solver feature and part of the longer control-systems
foundation.
