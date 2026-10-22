# Research integration ledger — 2026-10-22

## What the new attachments change

The canonical 239-page proof archive is now available, so its previous
**source-unavailable** blocker is removed. Its bounded-domain, constrained-search
and exact-density theorems support the project's definitions and assumptions.
Their existence/continuity statements do not give a quantitative general ML
approximation rate, a polynomial decoder or a verified numerical interval.

The endpoint and sparse integration reports supply concrete additional bounds
for `W_t(mu)=T_W[G_t*mu]`. They explicitly keep complete physical M1 open. Their
atomic integrator begins with a given rational list, and a sparse covering
algorithm pays separately for obtaining mass-oracle answers. Neither condition
is supplied by an arbitrary PySCF quadrature archive.

The new `decoders/smoothing.py` contains independently testable small components:

- Explicit `13 t^a H²` fractional smoothing error arithmetic under declared
  homogeneous regularity assumptions.
- Dropped/transported-source logarithmic stability arithmetic, including zero
  perturbation and the universal `3N/(8t)` alternative.
- Exact rational `W_t(N delta_a)=3N/(8t)` in dimension three, with no claim that
  this equals Ts when N>2.
- Analytic density/gradient evaluation for a supplied finite heat mixture,
  tested against the single-atom formula, scaling and Gaussian quadrature.

All general floating bounds are marked **CONDITIONAL**, with
`certified=False` and `roundoff_certified=False`. The dense diagnostic evaluator
has O(grid*atoms) cost. It is not the source's polylogarithmic interval integrator.

## Status by logical obligation

| Obligation | Status | Remaining requirement |
| --- | --- | --- |
| Canonical bounded-domain Ts definitions and low-N sector | Source AVAILABLE; implementation conditional on valid density input | Ground-state provenance and quadrature error remain separate. |
| Generic weighted-packet theorem integration | BLOCKED | Exact October 18 primary proof still absent; explicit low-mass packet subclass is narrower. |
| Sparse-PAFF mass-only compiler | BLOCKED | Exact October 20 audit/theorem absent; cannot infer PAFF structure from arbitrary samples. |
| Quantitative continuum direct-Pauli dual | BLOCKED | Exact October 21 proof and verified regularity/Ritz constants absent. |
| Heat smoothing helper budgets | CONDITIONAL; numerically tested | Independent regularity/transport bounds and interval rounding must be supplied. |
| Atom-list polylogarithmic interval integrator | OPEN implementation | Directed transcendental arithmetic, node/tail/moment budgets and bit-cost validation. |
| Cheap short atom-list production for all physical densities | OPEN research | Sources explicitly retain this missing implication. |
| Continuum full epsilon-N Ts certificate | OPEN | Complete local primal/dual, basis, quadrature, IMS, tail and access budgets. |
| Global variational/force guarantees | OPEN | Uniform control on an explicitly defined D_trial and reliable C1/C2 information. |

## Roadmap interpretation and next concrete task

The October 22 roadmap is an engineering/research proposal. Its Pauli-factor
neural primal with a cap-I reservoir is a sound **conditional analytic route**
when exact diagonal, regularity, reservoir construction and kinetic integrals
are available. It is not permission to turn float eigenvalue clipping into a
continuum certificate. Likewise `ell_kin=sqrt(N/int rho^(5/3))` has the correct
kinetic scaling without assuming a second spatial moment, but changing a
length alone does not make a global learned head additive.

A feasible next bounded task is to implement and independently test a
**finite-matrix Pauli-factor primal plus explicit cap-I reservoir repair**:
check the exact sampled diagonal/trace, require a genuinely feasible reservoir,
verify the convex-mixture cap analytically and numerically, and compare its
energy against the existing joint SDP on tiny fixed-density problems. Label
its bounds finite-dimensional numerical unless exact rational verification
closes them. Only after this should a continuum factor/interval-quadrature
implementation be advertised.

A separate needed task is to obtain the three exact missing primary decoder
sources; a citation or secondary handoff cannot substitute for reviewing their
hypotheses.
