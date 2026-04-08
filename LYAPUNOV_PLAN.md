# Lyapunov Solver Plan

## Goal

Add Lyapunov-equation solve support that is suitable for control-systems work,
with a strong emphasis on numerical reliability.

The immediate targets are the standard continuous-time Gramians for a stable
linear system

```text
x' = A x + B u
y  = C x
```

namely:

- controllability Gramian `Wc`, solving
  `A Wc + Wc A^H + B B^H = 0`
- observability Gramian `Wo`, solving
  `A^H Wo + Wo A + C^H C = 0`

This should be designed so that:

- direct solvers are reused where they are the right inner kernel
- compensated arithmetic is used where residual formation or reduction error is
  likely to dominate
- large sparse problems can be handled through low-rank factors rather than by
  explicitly forming dense Gramians too early

## Recommendation

Split the implementation into two solver families from the start.

1. Dense or modest-size direct Lyapunov solve.
2. Sparse large-scale low-rank Lyapunov solve.

The public API should make both available, but the sparse/control-oriented path
should be treated as the main long-term one.

Why:

- the full Gramian is generically dense even when `A`, `B`, and `C` are sparse
- controls workflows often care about medium-to-large sparse state matrices
- low-rank factors are usually the right object to compute and store

At the same time, a direct small-scale solver is still useful:

- as a reference implementation
- for tests and validation
- for smaller systems where explicit dense `W` is fine

## Scope Order

Recommended order:

1. continuous-time controllability solve
2. continuous-time observability solve by duality
3. continuous-time direct dense solver
4. discrete-time Stein equations
5. generalized / descriptor variants

The key principle is to get one numerically solid continuous-time path working
end-to-end before broadening the equation family.

## Public API

Suggested module:

- [`src/control/lyapunov.rs`](src/control/lyapunov.rs)

or, if we want to keep control-specific code separate:

- [`src/control/mod.rs`](src/control/mod.rs)
- [`src/control/lyapunov.rs`](src/control/lyapunov.rs)

Suggested first public types:

```rust
pub struct LyapunovParams<R> {
    pub tol: R,
    pub max_iters: usize,
}

pub struct LowRankFactor<T> {
    pub z: Mat<T>,
}

pub enum LyapunovError {
    DimensionMismatch,
    NotStable,
    InnerSolveFailed,
    IterationLimit,
}
```

Suggested first public entry points:

```rust
pub fn controllability_gramian_low_rank<T, A>(
    a: A,
    b: MatRef<'_, T>,
    params: LyapunovParams<T::Real>,
) -> Result<LowRankFactor<T>, LyapunovError>;

pub fn observability_gramian_low_rank<T, A>(
    a: A,
    c: MatRef<'_, T>,
    params: LyapunovParams<T::Real>,
) -> Result<LowRankFactor<T>, LyapunovError>;
```

and later:

```rust
pub fn controllability_gramian_dense<T>(...) -> Result<Mat<T>, LyapunovError>;
pub fn observability_gramian_dense<T>(...) -> Result<Mat<T>, LyapunovError>;
```

## Why Low-Rank First For Sparse Problems

For sparse control systems, the right-hand sides `B B^H` and `C^H C` are often
low rank or at least modest rank, while the Gramian itself is dense.

So the numerically and computationally natural target is:

- compute `Wc ≈ Zc Zc^H`
- compute `Wo ≈ Zo Zo^H`

instead of forming the full dense matrix immediately.

This matters for both memory and accuracy:

- the low-rank factors keep the intermediate algebra smaller
- residual checks can often be expressed in terms of `Z`
- downstream workflows like balanced truncation often naturally consume low-rank
  factors

## Algorithm Choice For The Sparse Path

Use low-rank ADI first.

Why low-rank ADI:

- it only requires repeated solves with shifted matrices
- it maps directly onto the sparse direct solvers we already have
- it is a standard practical method for Lyapunov equations in controls work
- it naturally produces low-rank factors

The key iteration for the controllability equation is driven by solves of

`(A + p_k I) V_k = previous_term`

for a sequence of shifts `p_k`.

Observability is the dual problem:

- solve the same algorithm on `A^H` with `C^H`
- or define the observability path as a thin dual wrapper over the same core

So once the controllability ADI core exists, observability should be a small
increment, not a separate project.

## Inner Solve Strategy

The inner shifted solves are where the existing sparse-direct work should be
used aggressively.

Preferred inner-kernel policy:

- use direct sparse solves where possible
- allow symbolic reuse when the sparsity pattern of `A + p_k I` is unchanged
- use the existing staged factorization wrappers rather than calling `faer`
  directly from Lyapunov code

That means:

- reuse [`SparseLu`](src/sparse/lu.rs) for generic shifted solves
- reuse [`SparseLlt`](src/sparse/cholesky.rs) or
  [`SparseLdlt`](src/sparse/cholesky.rs) when symmetry or definiteness makes
  them a better fit

This is one of the main reasons to build the Lyapunov solver on top of the
existing sparse wrappers instead of bypassing them.

## Compensation Policy

The Lyapunov solver should be explicitly designed to use compensated arithmetic
where it matters numerically.

Priority order:

1. compensated residual formation
2. compensated norms and stopping tests
3. compensated dense updates when combining low-rank contributions
4. direct compensated sparse matvec where residual checks use `A * X` style
   terms

The important point is that the numerically sensitive operations are not just
the inner direct solves. They are also:

- forming `B B^H`-like contributions
- accumulating low-rank basis columns
- computing residual norms for stopping

So the Lyapunov solver should use the compensated kernels already present in:

- [`src/sparse/compensated.rs`](src/sparse/compensated.rs)
- [`src/sparse/matvec.rs`](src/sparse/matvec.rs)

wherever that helps keep the outer iteration trustworthy.

## Direct Dense Solver

We should still plan a direct dense solver, but not as the main sparse path.

The cleanest role for it is:

- small-system exact solve
- reference implementation
- test oracle for low-rank methods on modest cases

Algorithmically, the likely choice is a Schur-based dense method:

- dense Schur decomposition or a Bartels-Stewart-style solve

This is not the first implementation target because it does not address the
large sparse control use case directly. But it is still valuable for
verification.

## Stability Handling

The infinite-horizon continuous-time Gramian only exists in the standard sense
when `A` is stable.

The first implementation should be conservative:

- document that the caller is expected to provide a stable `A`
- stop and return an error if the iteration clearly fails to contract
- avoid pretending we can certify stability cheaply in the sparse path

Later, we can add:

- optional small-system spectral checks
- heuristic diagnostics for likely instability

But the initial implementation should not overpromise on automatic stability
certification.

## Shift Strategy

Low-rank ADI lives or dies on shift selection.

The first implementation should separate the shift policy from the solve core.

Suggested public shape:

```rust
pub enum ShiftStrategy<R> {
    UserProvided(Vec<Complex<R>>),
    Heuristic,
}
```

Recommendation:

- support user-provided shifts immediately
- add a simple heuristic second

This keeps the solver usable on day one for informed users without forcing a
premature commitment to one baked-in heuristic.

## Low-Rank Factor Utilities

The low-rank factor type needs more than just storage.

Add helper methods for:

- rank / column count
- conversion to a dense Gramian for small validation cases
- optional recompression via QR or SVD later

Suggested direction:

```rust
impl<T: ComplexField> LowRankFactor<T> {
    pub fn rank(&self) -> usize;
    pub fn to_dense(&self) -> Mat<T>;
}
```

Do not put too much into this type initially. The important thing is to make it
a stable container for later utilities.

## Residual Evaluation

Residual evaluation is a first-class part of this design, not an afterthought.

For a low-rank controllability solution `W ≈ Z Z^H`, the residual is

`R = A Z Z^H + Z Z^H A^H + B B^H`

The first implementation should provide at least:

- an internal residual norm computation used by the stopping test
- a public residual-reporting path for diagnostics

Where practical, use compensated arithmetic in:

- dense inner products
- norm evaluations
- residual recombination

This is exactly the kind of place where controls code is often more sensitive
than generic numerical linear algebra.

## Phase 1: Low-Rank Controllability Core

- add the Lyapunov module
- add `LyapunovParams`
- add `LowRankFactor`
- add controllability low-rank ADI with user-provided shifts
- use existing sparse direct wrappers for shifted solves
- add compensated residual norm and stopping test

At the end of this phase, we should be able to compute `Wc ≈ Z Z^H` for stable
continuous-time systems.

## Phase 2: Observability By Duality

- add observability solve as a dual wrapper over the same core
- support `A^H` and `C^H`
- add tests comparing controllability and observability on dualized systems

This should be a small, deliberate extension of Phase 1, not a separate solver
implementation.

## Phase 3: Dense Direct Reference Solver

- add small-system dense solve path
- use it as a test oracle for low-rank results on modest cases
- add consistency tests between dense and low-rank outputs

This phase is mainly about correctness and diagnostics.

## Phase 4: Shift Heuristics And Recompression

- add a heuristic shift strategy
- add optional low-rank factor recompression
- benchmark convergence and rank growth on representative systems

This is the first performance-tuning phase rather than a pure capability phase.

## Phase 5: Discrete-Time Stein Equations

- add the discrete-time controllability equation
- add the discrete-time observability equation
- reuse the same low-rank factor and residual-reporting infrastructure

The continuous-time path should be solid before this starts.

## Phase 6: Generalized / Descriptor Systems

Only after the standard equation family is in place:

- generalized Lyapunov forms
- descriptor-system support
- integration with block / Schur machinery where structure helps

This is where the newer block infrastructure may start to matter more.

## Tests

Add tests for:

- small dense stable systems with known Gramians
- duality between controllability and observability
- residual decrease across ADI iterations
- complex-valued systems
- comparison against the dense direct path on modest cases
- failure behavior on obviously unstable or nonconvergent inputs

Also add cancellation-heavy tests where compensated residual evaluation is
expected to make a visible difference.

## Control-Systems Relevance

Once this exists, we can build the core controls-analysis features that depend
on Gramians, including:

- observability and controllability metrics
- balanced truncation infrastructure
- Hankel singular value workflows
- later Lyapunov- and Riccati-adjacent tooling

This is one of the main numerical-analysis capabilities still missing before
the sparse linear algebra stack starts to feel directly useful for controls.
