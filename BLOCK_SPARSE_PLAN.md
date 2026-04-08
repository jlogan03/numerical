# Block Sparse Plan

## Goal

Add a general block-sparse operator and matrix layer that can represent
user-defined block layouts, not just fixed 2x2 systems.

This should support:

- arbitrary user-defined block partitions
- arbitrary block-sparsity patterns
- mixed block implementations, such as sparse blocks, dense blocks, or
  matrix-free block operators
- later reuse by structured solvers and preconditioners

This is distinct from the specialized 2x2 block-preconditioner plans. Those
are still the right first path for Schur-complement and factorization-style
preconditioners. The general block-sparse layer is broader infrastructure.

## Why This Is A Separate Plan

The 2x2 block-preconditioner work is about a specific block algebra:

- block diagonal
- block triangular
- Schur factorization

General block-sparse support is a different problem:

- describing user block layouts
- storing which block positions are populated
- applying the whole block operator to a full vector
- optionally attaching local solves or preconditioners to selected blocks

Trying to generalize the 2x2 preconditioner directly into "arbitrary blocks"
would mix those concerns and make the first implementation much harder to keep
coherent.

## Recommendation

Implement the general block-sparse layer as its own module:

- [`src/sparse/block.rs`](src/sparse/block.rs)

and export its public types from:

- [`src/sparse/mod.rs`](src/sparse/mod.rs)

The first milestone should be a block-sparse **operator** abstraction, not a
full explicit sparse-matrix storage format.

Why operator first:

- it matches the current solver ecosystem, which already works with operator
  application
- it allows dense, sparse, and matrix-free blocks under one interface
- it avoids premature commitment to one explicit storage format for block data

Later, if explicit block storage proves useful, it can be built on top.

## Public API

Suggested layout types:

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockLayout {
    offsets: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockIndex {
    pub row: usize,
    pub col: usize,
}
```

Suggested core operator trait:

```rust
pub trait BlockOp<T: ComplexField> {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn apply(&self, out: &mut Col<T>, rhs: &Col<T>);
}
```

Suggested general block-sparse operator:

```rust
pub struct BlockSparseOp<T, B> {
    row_layout: BlockLayout,
    col_layout: BlockLayout,
    blocks: Vec<(BlockIndex, B)>,
    row_scratch: Col<T>,
    col_scratch: Col<T>,
}
```

The exact internal representation can change, but the public model should be:

- layouts describe block boundaries
- the block list describes which block positions are populated
- each populated block owns an operator object for that submatrix

## Layout Semantics

`BlockLayout` should store a monotone offset array:

- `offsets[0] == 0`
- `offsets.last() == total_dim`
- `offsets[i] <= offsets[i + 1]`

This gives:

- number of blocks = `offsets.len() - 1`
- block size `i` = `offsets[i + 1] - offsets[i]`

This is the right first boundary because it supports:

- unequal block sizes
- arbitrary user-defined partitions
- easy slicing of full vectors into block views

## Block Pattern Representation

The first implementation should store the populated block pattern explicitly as
a list of `(BlockIndex, block)` pairs.

That is enough for:

- operator application
- simple validation
- user-defined arbitrary sparsity patterns

Later, if the number of blocks gets large, we can add a CSR-like block-pattern
index for faster traversal. That is an optimization, not a first requirement.

## Mixed Block Types

Eventually, users will want different block implementations in different slots:

- sparse matrix blocks
- dense blocks
- matrix-free blocks
- special operator wrappers

There are two obvious ways to support that:

1. generic homogeneous block storage, where every block has the same type `B`
2. an enum or boxed trait-object layer for heterogeneous blocks

Recommendation:

- start with the generic homogeneous version
- add a heterogeneous wrapper later if real use cases need it

That keeps the first implementation simple and zero-cost for the common case
where all blocks come from the same source family.

## Operator Apply Semantics

The full operator apply is:

`y_i = sum_j A_ij x_j`

over populated block positions only.

Implementation outline:

1. zero the output full vector
2. for each populated block `(i, j)`:
   - copy the relevant input block `x_j` into a block-local view or temporary
   - apply the block operator
   - accumulate the result into the output block `y_i`

The first implementation can use owned dense temporaries for clarity. Later, we
can optimize for direct slice-based block views where the memory layout allows.

## Compensation Policy

This layer should eventually support both ordinary and compensated application.

Why:

- block-sparse systems can still suffer cancellation inside a block row
- we already have compensated summation machinery elsewhere in the sparse code

The first version should implement ordinary application first. After that, add:

- compensated accumulation across multiple contributing blocks in one block row
- optional compensated block-local apply when the child block type supports it

## Relationship To `SparseMatVec`

The block-sparse operator should not replace [`SparseMatVec`](src/sparse/matvec.rs).

Instead:

- `SparseMatVec` remains the scalar sparse-matrix apply trait
- the new block layer sits above it and can wrap block objects that themselves
  use `SparseMatVec`

If the APIs end up converging later, we can revisit that. For now, the block
layer should be a sibling abstraction, not a rewrite.

## Relationship To 2x2 Block Preconditioners

The 2x2 block-preconditioner work should still go first for Schur methods.

Later, the general block-sparse layer can support those specialized types by
providing:

- shared block-layout objects
- shared block-operator helpers
- shared dimension validation

But we should not force the 2x2 preconditioner to wait on the fully general
layer if that slows down the structured algorithms we already know we want.

## Suggested Errors

Add a small local error type:

```rust
pub enum BlockSparseError {
    InvalidLayout,
    DuplicateBlock { row: usize, col: usize },
    BlockOutOfBounds { row: usize, col: usize },
    BlockDimensionMismatch {
        row: usize,
        col: usize,
        expected_nrows: usize,
        expected_ncols: usize,
        actual_nrows: usize,
        actual_ncols: usize,
    },
}
```

These checks should happen at construction time. Operator application should be
non-fallible once the structure is built.

## Phase 1: Layout And Validation

- add `BlockLayout`
- add `BlockIndex`
- validate offsets and block dimensions
- validate the block pattern and reject duplicates

At the end of this phase, we should have a sound description of arbitrary block
partitions and populated block slots.

## Phase 2: Homogeneous Block-Sparse Operator

- add `BlockOp<T>`
- add `BlockSparseOp<T, B>`
- implement full-vector application
- add unit tests against dense reference results

This is the first actually useful general block-sparse artifact.

## Phase 3: Dense, Sparse, And Matrix-Free Adapters

- add adapters for sparse matrix blocks
- add adapters for dense matrix blocks
- add simple closures or wrapper types for matrix-free blocks

This is the phase where the layer becomes broadly reusable.

## Phase 4: Compensated Apply Path

- add compensated accumulation across blocks
- integrate with existing compensated sparse kernels where possible
- benchmark whether the extra arithmetic is justified on target problems

## Phase 5: Optional Explicit Block Storage

Only if a real use case requires it:

- add a concrete explicit block-sparse matrix type
- add conversion from the operator-style representation where practical
- decide whether block CSR, block CSC, or another pattern is worth storing

This should stay use-case driven rather than speculative.

## Tests

Add tests for:

- invalid layouts
- duplicate block rejection
- mismatched block dimensions
- exact apply on a small arbitrary block pattern against a dense reference
- mixed block sizes
- block rows with multiple populated blocks contributing to one output block

## Control-Systems Relevance

General block-sparse structure matters for control work because many real
systems are sparse at the block level rather than at the scalar-entry level.

That shows up in:

- coupled dynamics and constraints
- descriptor systems
- KKT systems
- later structured linear algebra around Lyapunov- and Riccati-adjacent work

So this layer is broader than preconditioning. It is the machinery needed to
express and exploit user-defined block structure in the first place.
