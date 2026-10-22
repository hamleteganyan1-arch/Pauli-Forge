# Scientific and implementation status — 2026-10-22

Status words describe the evidence and exact scope below. PROVED refers to
explicit algebra/analytic subclass identities under stated premises; tests are
NUMERICAL evidence and never establish a new general research theorem.

| Requirement | Status | Implemented scope / remaining limitation |
| --- | --- | --- |
| Units and PySCF factor 2 | NUMERICAL | Project −Δ vs ordinary Hartree documented and tested, including Mode B totals. |
| FrozenDE geometry/charge/spin/element gate | NUMERICAL | Immutable full-config hash, explicit HCNO-v1 and separate He/Li domain. |
| Compact physical ground-state D_E | CONDITIONAL | Restored theorem assumptions include binding/stratum conditions; geometry checks cannot prove density provenance. |
| Trial vs ground-state densities | PROVED API distinction | Public results never claim physical_ground_state_verified from a geometry gate. |
| Joint Gamma PSD/cap/trace/density | NUMERICAL | Real/complex finite-basis SDP, nonorthogonal AO transforms, spin channels, independent residuals and dual diagnostics. |
| Exact finite-matrix certificates | PROVED finite models | Fraction arithmetic independently checks rational primal, dual, PSD, equality and stationarity; not continuum Ts. |
| Scalar Pauli nonnegativity and low-N branch | PROVED architecture / NUMERICAL evaluation | Real arithmetic spin-summed ensemble sector; floating summation allowance only. Positivity is not matrix representability. |
| Positive analytical Pauli floor | CONDITIONAL | Default zero; explicit convention/source/hypothesis evidence required, no self-verifying theorem registry. |
| E(3), permutations and kinetic scaling | PROVED architecture / NUMERICAL tests | Consistent transformed quadrature, including optional e3nn density graph. Separate regridding errors are not symmetry errors. |
| Density-only inference | NUMERICAL | No hidden PySCF/orbital/Gamma input to neural prediction. QC/SDP are offline diagnostic paths. |
| Total Mode A density derivative | NUMERICAL | Discrete density-derived gradient operator, fixed-N directional checks, gauge, derivative scaling and backpropagation. |
| Total Mode B density derivative | OPEN | Offline kinetic response absent; attempted total derivative is explicitly refused. XC-only derivative works. |
| RHF/RKS/UHF/UKS data | NUMERICAL | Actual tiny PySCF calculations, matrices, grid refinement, raw error/provenance and checked Newton fallback. |
| Training, evaluation and checkpoint export | EMPIRICAL | Mode-separated, family-isolated, deterministic CPU examples, reliable derivative-label guards. Tiny data do not establish chemical accuracy. |
| Disjoint fragment composition API | PROVED composition / NUMERICAL tests | Only explicitly supported nonoverlapping compact fragments. |
| Bare-model size consistency, overlapping dissociation | OPEN | Not guaranteed by global invariant moments or the separate composition API. |
| Exact low-mass polynomial packets | PROVED explicit subclass | Disjoint H1_0 packets each mass <=2; exact rational TW and P=0, scoped continuum certificate. |
| General weighted-packet decoder | BLOCKED | Exact October 18 primary proof still unavailable, not replaced by the narrow zero-P subclass. |
| Regular-class Gram–SDP | NUMERICAL / CONDITIONAL | Period-two Fourier/full-diagonal constraints; exact polynomial moment enclosures for a narrow regular class. General truncation/solver/rounding budget not closed. |
| Negative-spectrum Ts dual | NUMERICAL | Available D1-C finite-dimensional formula and explicit operator uncertainty, no unproved continuum negative-tail substitution. |
| Quantitative continuum Pauli-dual decoder | BLOCKED | Exact October 21 primary source and verified Ritz/tail assumptions still absent. |
| Sparse-PAFF mass-only compiler | BLOCKED | Exact audit/theorem source and demonstrable PAFF input missing. |
| Adaptive atlas | NUMERICAL / CONDITIONAL | Finite-region strong 2:1 tree, robust active superset, full diagonal, IMS primitive. Global cover/tails/access costs unclosed. |
| Energy aggregator | CONDITIONAL; PROVED explicit packet branch | Complete error obligations mandatory; small numerical width alone never certifies. |
| Heat smoothing W_t helpers | CONDITIONAL; PROVED single atom | Source-scoped smoothing/transport budget arithmetic and exact rational single-atom formula; no full atomic polylog interval integrator. |
| Universal epsilon*N Ts/ML guarantee | OPEN | Global primal/dual, basis, tail, quadrature, IMS, rounding and access budgets not jointly closed. |
| General polynomial decoding / near-linear bit cost | OPEN | No such claim; default descriptors O(G), sparse edges can be quadratic. |
| Continuum functional potential/forces/minimizers | OPEN | Discrete autograd and energy fits do not prove uniform continuum derivative or variational bounds. |
| STRUCTURES25 benchmark | OPEN | License/API reviewed; independent historical dependency stack required. e3nn and PySCF are actually tested. |
| Research Console | NUMERICAL | All pages and actual HTTP/interactive checks; Atlas UI is only an octant overview. |
| CUDA and SLURM execution | OPEN / UNRUN | Device paths and portable CPU/GPU templates supplied; no CUDA device or cluster was available. |

The next bounded mathematical engineering task is a finite-matrix Pauli-factor
primal and cap-I reservoir repair, independently checked against joint SDP.
See `docs/RESEARCH_INTEGRATION.md`; it is a next task, not a completed result.
