# Composable Block Preconditioner Plan

## Goal

Add reusable block-preconditioner machinery so that we can compose larger
preconditioners from smaller exact or approximate block solves.

This is the right abstraction layer for later work on:

- saddle-point and KKT systems
- coupled multiphysics systems
- Schur-complement preconditioners
- control-systems problems with natural block structure

The immediate goal is not to support every block algorithm at once. The goal is
to define a small, composable 2x2 foundation that later preconditioners can
reuse.

## Recommendation

Start with a 2x2 block framework only.

Add a new preconditioner submodule:

- [`src/sparse/precond/block.rs`](src/sparse/precond/block.rs)

or, if it grows quickly:

- [`src/sparse/precond/block/mod.rs`](src/sparse/precond/block/mod.rs)

and re-export the public types from:

- [`src/sparse/precond/mod.rs`](src/sparse/precond/mod.rs)

The first public building blocks should be:

- a 2-way vector partition description
- a block-diagonal preconditioner built from two child `Precond`s
- a block-triangular preconditioner built from child `Precond`s and
  off-diagonal operators

That gives us an immediately useful composition layer without jumping straight
to Schur-specific logic.

## Why 2x2 First

Most of the interesting structured linear systems we care about can be phrased
first as 2x2 blocks:

- primal-dual systems
- KKT systems
- descriptor-system linearizations
- many coupled field problems

If the 2x2 API is clean, higher-arity block systems can be built later by
nesting or by adding dedicated n-block variants.

Trying to design the fully general n-block interface first would add a lot of
type and ownership complexity before we know what the actual solver users need.

## Core Design Principle

The block preconditioner should own the block structure and the child operators.
The solver should still see a single `P: Precond<T>`.

That means block splitting happens inside the preconditioner implementation, not
in the solver.

This keeps [`BiCGSTAB`](src/sparse/bicgstab.rs) and future iterative solvers
simple:

- they accept any `Precond<T>`
- a block preconditioner is just another `Precond<T>`

## Public API

Suggested supporting type:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockSplit2 {
    pub n0: usize,
    pub n1: usize,
}
```

Suggested first concrete preconditioner:

```rust
pub struct BlockDiagonalPrecond2<P0, P1> {
    split: BlockSplit2,
    p0: P0,
    p1: P1,
    scratch0: MemBuffer,
    scratch1: MemBuffer,
}
```

Suggested second concrete preconditioner:

```rust
pub struct BlockUpperPrecond2<P0, P1, B01> {
    split: BlockSplit2,
    p0: P0,
    p1: P1,
    b01: B01,
    block_tmp: Col<T>,
    scratch0: MemBuffer,
    scratch1: MemBuffer,
}
```

The exact naming of the triangular variant can wait until implementation, but
the important point is:

- diagonal block actions come from child `Precond`s
- off-diagonal block actions come from operator-like block objects

## Operator Bounds For Off-Diagonal Blocks

The off-diagonal block objects should not require a whole new abstraction if we
can avoid it.

The simplest first boundary is probably:

- use a small local helper trait for applying a block operator to a `Col<T>`
- implement that trait for sparse matrix views or wrapper types

Using [`SparseMatVec`](src/sparse/matvec.rs) directly is possible, but it is
currently shaped around full-matrix application rather than subvector partition
logic. A tiny block-operator trait keeps the block layer focused.

Suggested internal helper:

```rust
pub trait BlockOp<T: ComplexField> {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn apply(&self, out: &mut Col<T>, rhs: &Col<T>);
}
```

This can stay local to the block-preconditioner module until we know it is
reusable elsewhere.

## Vector Partition Helpers

Add internal helpers for splitting and rejoining dense vectors:

- immutable split of one `Col<T>` into two logical blocks
- mutable split of one `Col<T>` into two logical blocks
- copy between a full vector and block-local temporary vectors

The current [`src/sparse/col.rs`](src/sparse/col.rs) helpers should probably be
extended rather than bypassed.

The key goal is to avoid fresh allocations inside each preconditioner apply.

## First Concrete Variant: Block Diagonal

Start with a block-diagonal preconditioner:

`M^{-1} = diag(M_0^{-1}, M_1^{-1})`

Why first:

- easy to define
- immediately useful
- validates the block split and scratch-buffer design
- works with any existing child preconditioners, including
  [`DiagonalPrecond`](src/sparse/precond/diagonal.rs) and
  [`SparseLu`](src/sparse/lu.rs)

The application rule is just:

1. split the input vector into two blocks
2. apply `p0` to block 0
3. apply `p1` to block 1
4. assemble the output

## Second Concrete Variant: Block Triangular

After block diagonal works, add an exact block-triangular composition.

For an upper-triangular block system

```text
[ A  B ]
[ 0  D ]
```

the inverse application is:

1. solve with `D`
2. apply the off-diagonal correction through `B`
3. solve with `A`

This is still structurally simple, but it introduces the important pattern of:

- child preconditioner apply
- off-diagonal operator apply
- temporary vector reuse

That same pattern will later reappear in Schur-complement preconditioners.

## Why Not Jump Directly To Schur

Schur-based block preconditioners need more than just block partitioning. They
also need:

- an `A^{-1}` block solve or preconditioner
- a Schur operator or Schur preconditioner
- a factorization order

It is better to validate the simpler block partition and child-preconditioner
composition layer first, then build Schur logic on top.

## Public Error Handling

Add a small local error type:

```rust
pub enum BlockPrecondError {
    DimensionMismatch {
        which: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidSplit { total: usize, n0: usize, n1: usize },
}
```

These errors should be constructor-time or debug-time checks. The actual
`Precond<T>` application path should stay allocation-free and non-fallible once
the object is built.

## Phase 1: Split And Scratch Infrastructure

- add `BlockSplit2`
- add vector split/join helpers
- add local block-operator helper trait
- add constructor validation and scratch-buffer ownership

At the end of this phase, we should have the reusable substrate for block
preconditioners.

## Phase 2: Block-Diagonal Preconditioner

- implement `BlockDiagonalPrecond2<P0, P1>`
- implement `Precond<T>`
- test with combinations of identity, diagonal, and LU child preconditioners

This should be the first shipped compositional block preconditioner.

## Phase 3: Block-Triangular Preconditioner

- implement one exact triangular variant
- validate off-diagonal operator dimensions
- add tests against dense 2x2 block reference solves

This is where the block framework becomes broadly useful.

## Phase 4: Integration With Schur Machinery

After the separate Schur-construction and Schur-preconditioner work exists:

- reuse the same `BlockSplit2`
- reuse the same temporary-vector handling
- plug Schur-based solves into a block factorization preconditioner

That is the main reason to keep this block layer small and reusable.

## Tests

Add tests for:

- invalid block splits
- exact block-diagonal application on simple hand-built systems
- block-triangular application against a dense reference
- composition of child preconditioners with mixed types
- use inside [`BiCGSTAB`](src/sparse/bicgstab.rs) on a simple coupled system

## Out Of Scope For The First Pass

- arbitrary n-block composition
- nested block trees
- matrix assembly utilities for explicit block sparse matrices
- Schur-specific algorithms in the same module
- automatic block discovery from sparsity patterns

Those can all come later if the 2x2 foundation proves useful.

## Control-Systems Relevance

Control and estimation problems often lead to structured linear systems rather
than unstructured sparse ones.

A block-preconditioner layer lets us express that structure directly and sets up
later work on:

- KKT and descriptor systems
- block Schur reductions
- linear solves that appear inside Riccati, Lyapunov, and related workflows

So this is infrastructure, not just one more preconditioner type.
