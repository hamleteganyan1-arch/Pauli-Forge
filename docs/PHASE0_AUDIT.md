# Phase 0 audit — 2026-10-22

The supplied archive was imported unchanged in commit `f657279`. The three
separately attached instruction documents are byte-identical to their archive
counterparts. Original tests run against an untouched extraction, independently
of implementation changes. Full command output lives in `results/baseline/`.

The initial machine lacked PyTorch and pytest: both pre-install commands failed
with `ModuleNotFoundError`. A Python 3.12 virtual environment now contains the
CPU PyTorch distribution, optional SDP/QC/equivariant/UI dependencies, and test
tooling. Installed package requirements are checked independently with pip.

After installation, the original unmodified suite completed: **7 passed in
0.79 s**. The original Gaussian smoke calculation completed with N=3 and
T_W=2.2499985658012953 in the project's `-Delta` convention. Its randomly
initialized ML energy is a smoke result, not a physical reference.

## Baseline defects and gaps

1. `assemble` reports a global certificate from interval width alone, with no
   evidence that the interval endpoints, localization or numerical errors are
   valid. Narrow intervals must be distinguished from certified intervals.
2. Density positivity checks omit NaNs/infinite values, and some shape or empty
   matrix errors occur after linear algebra has already been attempted.
3. FrozenDE silently truncates nonintegral nuclear charges, lacks a version,
   spin/parity constraints and a checkpoint-bound domain identity.
4. Independent supplied `grad_u` does not by itself define a total derivative
   with respect to density. A differentiable density representation is required.
5. The baseline encoder allocates a dense grid-by-grid distance matrix. This is
   unsuitable for the existing PySCF water export, which can have many grid nodes.
6. An SDP solver status and small residual are numerical evidence, not a rigorous
   continuum primal/dual interval. Eigenvalue projection loses target density.
7. Only RKS reference export exists; reference provenance and complete matrix,
   unit, grid-refinement and spin metadata are missing from saved examples.
8. Checkpoints omit the domain, architecture configuration, seed and dependency
   versions. No chemistry-family split or separate XC mode is implemented.
9. The UI, adaptive source-based decoding and exact disjoint-fragment interface
   are absent. Several specifically named proof sources are not provided.

## Verification policy

Numerical tests exercise executable behavior. They do not establish a research
theorem. Every implemented certificate must state its finite-dimensional or
continuum scope, independent assumptions, provenance and unclosed error budget.
The default analytic Lieb–Thirring coefficient remains zero. Positive Pauli ML
predictions never certify a feasible one-particle density matrix.
