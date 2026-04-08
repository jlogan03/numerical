# Schur Complement Preconditioner Plan

## Goal

Add a reusable Schur-complement-based preconditioner for 2x2 block systems.

The main target matrix form is:

```text
[ A  B ]
[ C  D ]
```

The preconditioner should exploit block factorization structure instead of
treating the whole matrix as an opaque sparse operator.

This is useful for:

- saddle-point and KKT systems
- descriptor and constrained dynamics systems
- coupled sparse problems where one block is much easier to solve than the
  whole matrix

It is also the natural consumer of the block-preconditioner and Schur-operator
machinery planned in the companion docs.

## Recommendation

Implement a 2x2 block factorization preconditioner that uses:

- an `A`-block inverse application
- a Schur-complement inverse or preconditioner
- off-diagonal block operators `B` and `C`

Add a new module:

- [`src/sparse/precond/schur.rs`](src/sparse/precond/schur.rs)

and re-export it from:

- [`src/sparse/precond/mod.rs`](src/sparse/precond/mod.rs)

The first implementation should apply a block factorization, not assemble a new
approximate matrix.

## Exact Block Factorization View

For

```text
M = [ A  B ]
    [ C  D ]
```

with Schur complement

`S = D - C A^{-1} B`

one exact inverse factorization is:

```text
M^{-1}
= [ I         -A^{-1} B ] [ A^{-1}   0 ] [ I   0 ]
  [ 0              I    ] [   0    S^{-1}] [ -C A^{-1}  I ]
```

That formula is valuable because it tells us exactly what the preconditioner
needs to do:

- apply `A^{-1}`
- apply `S^{-1}` or a Schur-block preconditioner
- apply `B` and `C`
- combine the results with dense vector updates

The first implementation does not need every variant of this factorization. It
just needs one clean, reusable one.

## Public API

Suggested first concrete type:

```rust
pub struct SchurPrecond2<AInv, SInv, B, C> {
    split: BlockSplit2,
    ainv: AInv,
    sinv: SInv,
    b: B,
    c: C,
    tmp0: Col<T>,
    tmp1: Col<T>,
    tmp_a: Col<T>,
    tmp_s: Col<T>,
}
```

Suggested bounds:

- `AInv: Precond<T>`
- `SInv: Precond<T>`
- `B`: block operator from block 1 into block 0
- `C`: block operator from block 0 into block 1

This should itself implement `Precond<T>`.

## Why This Is A Preconditioner And Not A Direct Solve

The Schur block solve can be:

- exact
- lagged
- itself preconditioned

and the `A` block solve can likewise be exact or lagged.

That means the overall object is naturally a preconditioner composition layer,
not just a disguised direct solver.

If both `A^{-1}` and `S^{-1}` are exact current-matrix solves, then the block
factorization becomes an exact direct solve for the 2x2 system. That is still
fine, but the broader value is that the same interface also covers lagged or
approximate sub-block solves.

## First Variant To Implement

The first shipped version should use one concrete factorization order and stick
to it.

Recommendation:

- implement the lower-upper style application implied by the exact factorization
- keep it right-preconditioner-friendly, since [`BiCGSTAB`](src/sparse/bicgstab.rs)
  is currently right-preconditioned

The side of preconditioning is still a solver concern. The important point here
is that the preconditioner object itself represents a single inverse-like block
operator.

## Dependency On Other Planned Work

This preconditioner depends on two other pieces of infrastructure:

- the 2x2 block split and workspace handling from
  [`BLOCK_PRECOND_PLAN.md`](BLOCK_PRECOND_PLAN.md)
- the implicit Schur operator or Schur solve path from
  [`SCHUR_COMPLEMENT_PLAN.md`](SCHUR_COMPLEMENT_PLAN.md)

So the Schur preconditioner should be implemented after, not before, those two
foundations.

## Suggested Construction Paths

There should eventually be at least two construction patterns.

Pattern 1:

- caller supplies `AInv`, `SInv`, `B`, `C`, and the split directly

Pattern 2:

- caller supplies `AInv`, `B`, `C`, `D`
- the constructor builds an implicit Schur operator
- caller also supplies or chooses how the Schur block is inverted or
  preconditioned

Pattern 1 should come first because it keeps the preconditioner focused on the
inverse application rather than on constructing the Schur machinery itself.

## Workspace And Allocation Policy

The preconditioner should own all dense temporary storage it needs.

At minimum it will need:

- one temporary in block 0 coordinates
- one temporary in block 1 coordinates
- one or more full-block temporaries for the off-diagonal corrections

As with the current solver and preconditioner bridge, the apply path should not
allocate.

## Compensation Policy

The first implementation should focus on structural correctness and reuse.

After that, consider a second path that uses compensated dense updates when
combining the block contributions, especially for:

- `x1 - C A^{-1} x0`
- `x0 - A^{-1} B y1`

If the off-diagonal block operators themselves later gain compensated apply
paths, this preconditioner can use them too.

## Phase 1: Minimal 2x2 Schur Preconditioner

- add `SchurPrecond2`
- reuse `BlockSplit2` and block temporary handling
- implement `Precond<T>` with caller-supplied `AInv`, `SInv`, `B`, and `C`
- test against a dense reference block inverse on small systems

At the end of this phase, we should have a reusable structured preconditioner
for 2x2 systems.

## Phase 2: Integration With Existing Solvers

- use [`SparseLu`](src/sparse/lu.rs) as `AInv`
- use a direct or lagged Schur-block solve as `SInv`
- exercise the result inside [`BiCGSTAB`](src/sparse/bicgstab.rs)

This phase proves the composition story with the concrete sparse wrappers we
already have.

## Phase 3: Schur-Block Builder Convenience

- add helpers that pair a Schur operator with a chosen Schur-block solver or
  preconditioner
- reduce the amount of manual wiring needed at call sites

These helpers should be thin and should reuse the explicit types built in the
earlier phases.

## Phase 4: Compensated Path And Benchmarking

- add compensated dense combination steps
- benchmark iteration count and time against diagonal and LU-based baselines
- identify which structured problems actually benefit

This should stay evidence-driven rather than assumed.

## Tests

Add tests for:

- exact small 2x2 block systems against a dense inverse reference
- complex-valued block systems
- invalid block-dimension combinations
- use with exact `SparseLu` sub-block solves
- use with lagged sub-block solves
- reduced iteration count on a structured coupled system compared with a simple
  diagonal preconditioner

## Out Of Scope For The First Pass

- automatic block extraction from a monolithic sparse matrix
- n-block Schur hierarchies
- explicit approximate Schur assembly in the first version
- AMG or multilevel Schur solvers

Those are natural future directions, but they would obscure the first useful
2x2 implementation.

## Control-Systems Relevance

Schur-based preconditioners are one of the standard ways to respect the real
structure of constrained and coupled systems instead of flattening everything
into one generic sparse solve.

That matters for later control-oriented work because many of those systems are
best handled as block operators, not as anonymous sparse matrices.

This still is not a Lyapunov solver by itself, but it is exactly the kind of
linear-solver infrastructure that makes later control-systems work tractable.
