# Schur Complement Construction Plan

## Goal

Add reusable Schur-complement machinery for 2x2 block systems so that we can
build:

- implicit Schur operators
- explicit Schur complements later, when materialization is justified
- Schur-based block preconditioners on top of the same core

The main target block form is:

```text
[ A  B ]
[ C  D ]
```

with Schur complement:

`S = D - C A^{-1} B`

This is useful for saddle-point and KKT systems, but it is also part of the
longer path toward more structured control-systems tooling.

## Recommendation

Implement the implicit operator first.

Add a new sparse module:

- [`src/sparse/schur.rs`](src/sparse/schur.rs)

and export the public types from:

- [`src/sparse/mod.rs`](src/sparse/mod.rs)

The first public type should represent the action of the Schur complement on a
vector without explicitly materializing a new sparse matrix.

Why implicit first:

- explicit Schur complements often introduce heavy fill-in
- we already have the ingredients for operator application
- preconditioners and iterative solvers often only need `S x`, not stored CSC
  data

## Available Building Blocks Today

The current repo already has most of what an implicit Schur operator needs:

- sparse block matvec through [`SparseMatVec`](src/sparse/matvec.rs)
- exact or lagged inverse application through [`SparseLu`](src/sparse/lu.rs)
- compensated arithmetic helpers in [`compensated.rs`](src/sparse/compensated.rs)

That means the first version does not need new sparse factorization kernels. It
needs a clean operator wrapper and careful workspace management.

## Public API

Suggested first public type:

```rust
pub struct SchurComplement2<AInv, B, C, D> {
    ainv: AInv,
    b: B,
    c: C,
    d: D,
    n_a: usize,
    n_s: usize,
    tmp_b: Col<T>,
    tmp_ainv_b: Col<T>,
    tmp_d: Col<T>,
    tmp_c: Col<T>,
}
```

The type parameters represent:

- `AInv`: an object that applies `A^{-1}` to a vector
- `B`: an operator from the Schur block into the `A` block
- `C`: an operator from the `A` block into the Schur block
- `D`: the trailing block operator

The first boundary can be:

- `AInv: Precond<T>`
- `B: SparseMatVec<T>` or a small local block-operator trait
- `C: SparseMatVec<T>` or a small local block-operator trait
- `D: SparseMatVec<T>` or a small local block-operator trait

That is enough to represent `S x = D x - C (A^{-1} (B x))`.

## Why Use `Precond<T>` For `A^{-1}`

The Schur operator only needs repeated application of `A^{-1}` to dense
vectors.

That means it does not care whether `A^{-1}` comes from:

- an exact direct solve
- a lagged factorization
- a block solve
- another preconditioner-like inverse approximation

So `Precond<T>` is a reasonable first boundary for the `A^{-1}` slot, even when
the supplied object happens to be an exact solver wrapper like
[`SparseLu`](src/sparse/lu.rs).

If that name starts feeling misleading later, we can introduce a more general
inverse-operator trait and implement it for `Precond<T>` objects.

## Operator Semantics

The operator action should be:

1. compute `u = B x`
2. compute `v = A^{-1} u`
3. compute `w = C v`
4. compute `y = D x - w`

The implementation should own and reuse all temporary vectors needed for those
steps.

This is the same pattern we already use elsewhere:

- no per-apply allocation
- separate storage for logically distinct intermediate quantities
- one object owns the working buffers

## Compensation Policy

The first version should support ordinary operator application first.

After that, we should consider a second path that uses compensated operations
where we already have them:

- compensated block matvec for `B x`, `C v`, and `D x`
- compensated vector subtraction for `D x - w`

This is especially interesting because Schur complements are often used on
difficult coupled systems where cancellation can be real.

It should be a second step, not a blocker for the initial operator.

## Explicit Construction: Later, Not First

An explicit Schur complement builder is still worth planning, but it should not
be the first thing implemented.

Explicit construction would need:

- symbolic sparse matrix multiplication and subtraction
- careful control of fill and pattern growth
- decisions about when the explicit matrix is actually cheaper than the
  implicit operator

`faer` has sparse matmul primitives, so an explicit constructor is possible.
But the first practical value is the implicit operator.

## Public Explicit API To Add Later

Later, add something like:

```rust
pub struct ExplicitSchur2<I: Index, T: ComplexField> {
    matrix: SparseColMat<I, T>,
}

pub fn construct_explicit_schur2<I, T, AInv, B, C, D>(...) -> Result<ExplicitSchur2<I, T>, _>;
```

That should be delayed until we have a real use case requiring stored CSC
structure rather than operator application.

## Suggested Module Layout

The first [`src/sparse/schur.rs`](src/sparse/schur.rs) can hold:

- `SchurComplement2`
- a small local operator trait if needed
- shared scratch helpers

If explicit and implicit paths both grow substantially later, then split it
into:

- `src/sparse/schur/mod.rs`
- `src/sparse/schur/implicit.rs`
- `src/sparse/schur/explicit.rs`

## Phase 1: Implicit Operator

- add `SchurComplement2`
- implement dimension checks
- implement ordinary apply with owned temporaries
- add tests against a dense reference implementation

At the end of this phase, we should be able to treat the Schur complement as a
matrix-free operator.

## Phase 2: Solver Integration

- make sure the implicit Schur operator can be used wherever a linear operator
  is expected
- add tests with iterative methods or block preconditioners that call it

This phase is about proving that the operator is not just mathematically right,
but practically reusable.

## Phase 3: Compensated Apply Path

- add optional compensated block matvec usage
- add compensated `sum2`-style subtraction in the final combine step
- benchmark whether the extra arithmetic is justified on difficult coupled
  systems

## Phase 4: Explicit Construction

Only after the implicit path proves useful:

- add explicit symbolic construction
- add explicit numeric assembly
- expose CSC output for direct sparse factorization or reuse

This should stay use-case driven.

## Tests

Add tests for:

- exact implicit apply against a small dense hand-computed Schur complement
- dimension mismatch rejection
- complex-valued block systems
- exact Schur operator built from [`SparseLu`](src/sparse/lu.rs) as the `A^{-1}`
  slot
- optional compensated-path agreement with the ordinary path on benign systems

## Out Of Scope For The First Pass

- automatic extraction of `A`, `B`, `C`, `D` from one monolithic sparse matrix
- higher-arity block systems
- explicit sparse assembly in the first version
- Schur-specific preconditioner logic in the same module

Those should build on top of a clean implicit operator first.

## Control-Systems Relevance

Schur complements are one of the main ways structured linear systems get turned
into smaller reduced solves.

That matters for:

- constrained optimization and estimation systems
- descriptor-system linear algebra
- future block algorithms that will show up around control workflows

This is not yet a Lyapunov solver, but it is part of the infrastructure needed
to handle the structured linear algebra those workflows tend to require.
