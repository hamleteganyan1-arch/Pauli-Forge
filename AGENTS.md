# CODEX MISSION: implement PauliForge v1 (must read first)

You are the coding agent responsible for implementing a mathematically honest **density-only OF-DFT** project, starting from this runnable PyTorch prototype. DO NOT silently substitute a molecular interatomic force field for a functional of electronic density. DO NOT assert a successful research theorem from a passing numerical test.

## Goal
Build two user-selectable scientifically distinct modes, with the same atomic-domain and certification layers:

1. **`kinetic_hybrid`**: `T_s^pred[rho] = T_W[rho] + B_P^analytic[rho] + Delta_P^ML[rho]` where `B_P` is a **proven lower barrier**, not an exact Pauli functional. Density-only nonnegative ML residual, exact `P=0` when N<=2, rigid rotation/translation/inversion/permutation and coordinate-scaling invariances exact by architecture. Optional Pauli-safe offline teachers, neural functional derivative, and interval certification layer.
2. **`analytical_pauli_xc_ml`**: `E[rho] = T_W[rho] + P_constrained[rho] + E_H[rho] + E_ext[rho;R,Z] + E_xc^ML[rho]` with constrained-search Pauli computed in a **finite basis** offline / diagnostic first, and explicit `not_continuum_certified` status until continuum error bounds implemented. Exchange-correlation model must be separate from kinetic residual, and no double-counting of KS/DFT XC terms is allowed.

All quantities atomic units, `T = -Delta` in project convention, `T_W = int |grad sqrt(rho)|^2`, `N = int rho`. PySCF uses `-Delta/2`, multiply its kinetic values by two when comparing `T_s`/Pauli project values. Spin-summed 1RDM cap `0<=Gamma<=2I`; explicit spin channels have cap I.

## Must use existing research and inspect before changes
Primary attached/project works if available (do not assume a path exists):
- `Ts_Functional_Strict_Theorem_Archive_2026-10-09.pdf` (proof corpus)
- `Ts_Adaptive_Multiscale_Physical_Atlas_Proof_RU_2026-10-15(1).md`
- `Ts_Polynomial_Catalog_Range_Entropy_and_Decoder_RU_2026-10-15.md`
- `Ts_Polynomial_Weighted_Adaptive_Packet_Decoder_RU_2026-10-18.md`
- `Ts_Direct_Decoder_LogLip_Gram_SDP_Proof_RU_2026-10-18.md`
- `Ts_M1_Fifteen_Routes_Aggregate_Decoder_Proof_RU_2026-10-19.md`
- `Ts_M1_Decoder_Audit_Exact_Mass_Gradient_9_Slabs_and_Sparse_Theorem_2026-10-20.md`
- `Ts_M1_Pauli_Direct_Dual_Quantitative_Lower_Certificates_2026-10-21.md`
- `Ts_ML_Pauli_Analytic_Research_Audit_2026-10-21_RU.md`
If source files are unavailable, mark specific theorem integration as blocked and DO NOT fabricate claims or silently replace it with a heuristic.

Existing software reference adapters, use their public API; respect license: [STRUCTURES25](https://github.com/sciai-lab/structures25) (actual ML orbital-free framework; LGPL-3.0), [e3nn](https://github.com/e3nn/e3nn), [MACE](https://github.com/ACEsuit/mace) (equivariant inspiration, NOT KEDF), [PySCF](https://github.com/pyscf/pyscf). Pin working dependency ranges after running actual integration tests; do not copy third-party source into our repository without license review.

## Exact scientific constraints
1. Build `DensityInput` with `(x_g [Bohr], w_g [Bohr^3], rho_g, grad sqrt(rho), quadrature/error certificate)`; `grad sqrt(rho)` MUST be computed from a density representation / input, never secret KS orbitals during prediction. Check nonnegativity, `int rho`, energy mass and strong H1 data.
2. Build *joint* matrix teacher: in an orthonormal basis `Gamma=Gamma*`, `0 <= Gamma <=2I`, `Tr Gamma=N`, `rho(x_g)=sum_{ij}A_gi Gamma_ij A_gj`. With nonorthonormal AOs `G_orth=S^(1/2) D S^(1/2)` and `Tr DS=N`. Avoid eigenvalue-clipping-only to claim density recovery; provide explicit equality residual and dual/primal objective bound. Include spin-restricted/unrestricted handling.
3. `T_s` and `P` are ensemble constrained-search targets. For any `N<=2`, `P=0` exactly (real-valued N included). Never replace full-diagonal local `P_a[rho|Omega]` with `P[chi^2 rho]`: sources explicitly give a counterexample.
4. `D_E`: immutable allowed Z list, fixed max nuclei, nuclear min distance, relative compact K (center geometry), allowed charge/spin sector, atomic units and finite binding/ground-state assumptions. Allowed geometry is NOT sufficient to prove supplied rho is exact ground-state density. Distinguish `D_E` (ground-state) from variational trial `D_trial`; separate benchmark assertions and minimization. If chemistry set grows, create NEW domain/model version, not implicit extrapolation.
5. Enforce scalar energy invariance to translations, proper and improper rotations, permutations of identical nuclei and grid nodes by construction; exact uniform coordinate scaling `rho_lambda(x)=lambda^3 rho(lambda x)`, `T[rho_lambda]=lambda^2 T[rho]`. Equivariance for vector gradients/forces must be tested separately. Charge and nuclear permutation must not leak arbitrary atom ordering.
6. Enforce nonnegative Pauli and analytic lower constraint, with positive `C_LT` only if the constant's exact spin/kinetic convention is certified. Default zero. Thomas-Fermi energy is a *heuristic prior* unless justified by a theorem; don't call `max(0, C_TF I-W)` a rigorous lower bound. N<=2 must override and be exact.
7. Enforce size consistency for proven separated-support test densities using composition-aware architecture or explicit separation detection with *certified* nonoverlap. Generic local neural kernel must not be declared exact at long-range Coulomb/metallic response (1/q^2 Hessian tails documented). Dissociation needs a distinct test/physics channel.
8. `P`-ML constrained output positivity is **not sufficient** for representability by a valid `Gamma`. Distinguish a scalar lower-barrier guarantee from matrix Pauli-exclusion and from accuracy. A certified interval `[L_Q,U_Q]` must come from primal/dual certificate, not a neural confidence output.
9. Weighted atlas requires `alpha_Q>=0`, `sum alpha_Q=O(epsilon N)`, full local diagonal, known IMS residual, `T_W` numerical budget. Stop only if `sum alpha_Q (U_Q-L_Q)+IMS+TW_width <= epsilon N`. The source `D1-C` qualitative termination does not prove polynomial bit complexity. Document the unresolved decoder/finite-input bottlenecks.
10. Train on energies AND functional derivatives whenever derivative data are scientifically reliable; a smooth `C0` energy fit alone does not certify density minimization or forces. Autograd checks must include numerical directional derivatives under N-preserving perturbations, `rho>=0`, gauge constant of potential.

## Implementation tasks, prioritized; do not wait for clarification
### Phase 0 — baseline audit (first commit)
- Run `pytest -q` and smoke. Review `docs/PHYSICS_CONTRACT.md` and find every TODO gap.
- Set reproducible format/lint, seed and explicit environment manifest. Add CI CPU tests and optional expensive QC matrix tests.
- Fix any inconsistent unit conversions and tests before refactors.

### Phase 1 — real data and exact domain gating
- Add PySCF RHF/RKS/UHF/UKS offline data adapters; pull canonical AO 1RDM, overlap, kinetic `-Delta/2`, quadrature rho and gradients, and log provenance, charge, spin and molecular composition.
- Build molecular/family split (exclude geometry leakage), domain versioned JSON and out-of-domain refusals. Validate integrated electrons plus quadrature convergence across grid refinements.
- Establish small labels set H, H2, He (N<=2 sector), Li, H2O, CH4, H2 dissociation and atomic/molecular rotations. Default allowed Z are illustrative [1,6,7,8]; expand only by explicit config.

### Phase 2 — defensible Pauli teacher
- Use CVXPY SDP for tiny finite-basis examples with exact equality constraints. Add explicit primal feasible Gamma and a *valid* dual lower objective where available. Check numerical tolerances and the finite-basis/coarse quadrature caveat. Do not label CVXPY `optimal` alone as a certified interval.
- For larger examples implement source `Direct_Decoder_LogLip_Gram_SDP` only after checking assumptions of positive rho, log-Lip and window a; if not met, record failure instead of extrapolating.
- Add source direct Pauli-dual `L2(rho dx)` negative spectrum lower module with Ritz truncation error and separate source provenance. Its output requires inequalities and a checked prior bound, not point estimate.

### Phase 3 — architecture choices and efficiency
- Introduce pluggable `DensityEncoder` with `InvariantMomentBaseline` (existing), `E3DensityGraph` (`e3nn` or STRUCTURES25 adapter), `SparsePointKernel`, and an optional long-range response channel. Abandon dense O(G^2) for large grids.
- Provide explicit `PauliFloorHead`, `KineticResidualHead`, `ExchangeCorrelationHead`, `CertificateAggregator`. Share scalar descriptors where valid but keep distinct physical meanings and losses.
- Implement exact group action tests on both density/geometry; fixed-support dissociation tests, electron-count exactness, `N<=2`, scalar and derivative scaling. No data augmentation in place of invariance proofs.
- User can select `kinetic_hybrid` or `analytical_pauli_xc_ml` config; never mix XC error with Ts error metrics.

### Phase 4 — adaptive source-based decoding
- Implement full-diagonal `PauliBlock`, dyadic 2:1 adaptive tree, robust active super-set, weights and IMS budget as stated in atlas, not a guessed substitute.
- Use the proven polynomial weighted-packet decoder as **golden baseline on its exact packet-subclass**. Include analytical value/mass interpolation tests and assert the precise scope.
- Implement generic log-Lip Gram/SDP and mass-to-moments analytic-certified branch. Implement sparse PAFF mass-only compiler solely when u is demonstrably piecewise affine.
- Implement aggregate bounded decoder theorem, where `sum alpha||v-w||_2^2` is certified; do not require individual block cutoffs when aggregate bound suffices.
- Build configurable ML router to choose decoder from provided proof certificate. If none matches, fallback to qualitative expensive certified estimator OR return `uncertified`; never claim polynomial runtime broadly.

### Phase 5 — training, eval, UI
- Train with energy and selected functional derivatives on physical reference data; prevent duplicate/mirror/reordered molecule leakage and same-family leakage between train and test. Compare existing STRUCTURES25 architecture as an independent benchmark, not as certified functional.
- Provide uncertainty calibration as empirical (not rigorous intervals). Track energy MAE per electron, Pauli constraint violations, inversion/scaling residuals, electron-number errors, tails, density minimization failures, matrix primal/dual gaps and wall-time/bits.
- Implement optional Streamlit frontend with one of designs in `docs/DESIGN_OPTIONS.md`: Research Console (default) or Atlas Studio. Add 3D nuclei and rho visual, energy decomposition, local block interactive map, certificate explorer, proof/open statuses and benchmark gallery.
- Add CLI train/evaluate/export/sweep, SLURM CPU/H100 scripts with no hardcoded private filesystem assumptions, reproducible checkpoint metadata, and exact command lines.

## Acceptance / rejection tests (all required for claiming V1)
- CPU `pytest` passes and pin CI. Group-action numerical relative errors <1e-8 (double precision consistent quadrature) and exact analytic relation for scaling.
- `N=1` and `N=2` => ML Pauli ==0. For `N>2`, predicted Pauli>=0 and >= certified floor if configured.
- Matrix constraints: PSD/upper cap/trace as above; nonorthogonal AO overlap transformation; demonstrate raw eigenvalue projection changes rho and joint SDP preserves target sample density within tolerance.
- FrozenDE rejects outsider element, excess nuclei, charge outside range, collisions and unbounded relative geometry; no train/inference bypass. Contains allowed list / model domain version in checkpoint.
- Demonstrate orbital-free prediction uses no PySCF import, orbitals or gamma; PySCF only in offline `teachers` path.
- Weighted certificate: honest `[L,U]` and aggregate global condition; label source and theorem assumptions explicit. Never return `certified=True` absent all necessary input certificates.
- Exact supported disjoint compact-support fragments must satisfy Pauli additivity; if model fails, document unsupported until architecture repaired.
- Run density-grid refinement convergence, geometry-vs-density consistency, and N-preserving density functional derivative tests.
- Benchmark on fixed-domain molecules, held-out geometries, unseen compositions *within domain*, dissociation and model transfer. Record where performance is empirical.
- Check root source docs, no fabricated theorem names, no claim that polynomial label VALUE bank implies polynomial decoder DESCRIPTION.

## Deliverables
A clean installable GitHub-ready repo with README, `pyproject.toml`, `configs/`, `src/`, offline data teacher adapters, verified matrix and density-only code, neural model, plugin encoders, invariant unit tests, optional UI, CI, docs mathematical assumption tables, reproducible scripts, notebook, and licenses. Ship working intermediate commits with passing tests. Create `STATUS.md` with fields `PROVED / CONDITIONAL / EMPIRICAL / OPEN / BLOCKED` for every advertised functionality. Prioritize correctness over performance claims. 
