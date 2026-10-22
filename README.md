# PauliForge v1 — density-functional research implementation

Executable density-only PyTorch functionals, joint finite-basis Pauli teachers,
source-scoped decoders and an optional Physics Research Console. This is a
research implementation with explicit open requirements, **not** a certified
universal orbital-free functional or a solution to general polynomial decoding.

All coordinates are Bohr. Project kinetic energies use **−Δ**, twice PySCF's
ordinary Hartree kinetic convention. Coulomb/XC/full-total Hartree terms retain
their physical units; Mode B converts kinetic energy by `/2` before adding them.

## Install and verify

The tested frozen environment is Linux x86_64, Python 3.12, CPU PyTorch 2.14.1.
The lock contains resolved versions and distribution hashes, including optional
QC, SDP, e3nn, interface and developer dependencies.

```bash
cd /workspace/PauliForge  # or your extracted project directory
bash scripts/setup_environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests examples scripts
.venv/bin/ruff format --check src tests examples scripts
.venv/bin/python examples/smoke.py
```

For another supported Python/platform, install the appropriate PyTorch build
and `pip install -e '.[qc,cert,equivariant,ui,test,dev]'` in a separate environment;
that path is not the frozen CPU reproduction. GPU code accepts `--device cuda`
or `auto`; use the official [PyTorch installer](https://pytorch.org/get-started/locally/)
for your CUDA runtime. CUDA execution and SLURM submission are untested here.
CPU/GPU submission templates are under `scripts/`; H100 sites can supply their
GPU type/partition when invoking `sbatch scripts/train_slurm.sbatch`.

## Two scientific modes

| Mode | Learned quantity | Kinetic treatment |
| --- | --- | --- |
| `kinetic_hybrid` | Nonnegative Pauli residual | `Ts = TW + B_P + DeltaP`; safe default floor zero; structured external theorem evidence required for a positive coefficient. |
| `analytical_pauli_xc_ml` | Separate Hartree-unit XC model | Offline joint finite-basis constrained-search Ts, with explicit non-continuum status. No hidden KS call in density-only ML inference. |

Scalar Pauli positivity **does not** establish a feasible density matrix.
The matrix teacher imposes `0 <= Gamma <= 2I`, `Tr Gamma=N` and the complete
supplied sampled density jointly; explicit spin channels use cap `I`.
The exact `P=0` branch for real `N<=2` refers to the spin-summed unconstrained
ensemble functional, not a fixed-spin two-electron triplet functional.

The default density moment encoder uses O(G) storage/time. Sparse/e3nn density
encoders use actual cutoff edges; dense clouds can still have O(G²) edges.
Rigid motions, reflections, grid permutations and kinetic coordinate scaling
are built into scalar descriptors/heads. A finite grid and a passed invariance
test do not certify continuum quadrature, completeness or learned accuracy.

## Real reference data and training

Generate the kinetic and XC banks separately. HF samples deliberately do not
invent an XC target. Converged RKS/UKS targets remain approximate PBE references.
A failed DIIS iteration on stretched H2 triggers the supported Newton SCF
fallback, which is separately checked and recorded in provenance.

```bash
.venv/bin/python examples/reference_demo.py --output data/processed/reference_demo --extended-atoms
.venv/bin/python examples/reference_demo.py --xc-labels --output data/processed/xc_demo
mkdir -p results/demo
.venv/bin/pauliforge train data/processed/reference_demo --config configs/experiment.kinetic.json --seed 2 --domain configs/frozen_de.example.json --output results/demo/kinetic-heldout.pt
.venv/bin/pauliforge train data/processed/xc_demo --config configs/experiment.xc.json --seed 2 --domain configs/frozen_de.example.json --output results/demo/xc-heldout.pt
.venv/bin/python examples/run_research_demo.py
.venv/bin/pauliforge export results/demo/kinetic-heldout.pt --output results/demo/kinetic-portable.json
```

The demo split holds out CH4 as an entire composition family. Seven tiny
STO-3G examples cannot support an accuracy/generalization claim. Training
checkpoints contain architecture, mode, FrozenDE configuration/hash, seed,
versions, actual optimization history and separated training/validation families.
Evaluation refuses training-family overlap unless `--allow-training-families`
explicitly requests **fit metrics**. `pauliforge sweep --help` exposes seed/LR runs.

Raw AO archives preserve the density used by the offline teacher. Training
explicitly mass-normalizes its numerical grid copy, rescales `grad_u` coherently,
and records the factor/raw mass. Labels are not magically recomputed for this
slightly changed density. Off-mass samples with reliable derivative labels are
refused rather than assigned invented derivatives.

## Public domain and derivative APIs

Use `ValidatedInference(model, domain)` with `Geometry` and `DensityInput` for
chemistry-gated prediction. `FrozenDE` is an immutable, versioned HCNO domain
with atom-count, geometry, charge, spin/parity and electron limits. The example
JSON equals the default configuration. A changed hash requires a distinct
checkpoint/domain; He/Li benchmarks use an explicit separate domain version.

Low-level model calls operate on trial densities and carry no chemistry-domain
or physical-ground-state membership guarantee. Geometry validation does not
prove that a supplied density belongs to the Coulomb ground-state set D_E.

For a **total discrete density derivative**, build
`DensityInput.from_density_values(..., gradient_operator=...)` and use
`functional_derivative`; it differentiates both TW and the learned term and
supports the fixed-N gauge. Holding an unrelated supplied `grad_u` fixed is
refused. The dense derivative operator is a small-grid validation tool.
Mode B's offline constrained-search kinetic response is unavailable, so an
attempted total density derivative raises; its separate XC network remains
fully differentiable. No continuum C1/C2, force or variational-minimizer claim.

## Decoders, certificates and limits

- Exact rational primal/dual verification certifies specified tiny finite SDPs.
- Explicit disjoint compact polynomial packets with each mass <=2 have exact
  analytic `P=0`, cap-safe rank-one construction and a checked scoped interval.
- General regular Gram/Fourier SDPs and negative-spectrum Ts duals are numerical;
  regularity, moment, basis, truncation and rounding obligations remain explicit.
- The adaptive tree implements finite-region 2:1 geometry, full-density local
  restrictions, active-block supersets and IMS budget primitives. It does not
  infer unprovided continuum tails or an effective strong-H1 input from samples.
- Arbitrary narrow numeric intervals set `budget_satisfied`, never an automatic
  global certificate. ML confidence is not an energy enclosure.
- `forward_fragments` is a separate additive composed-surrogate API for checked
  explicit disjoint supports. The bare model on a union density has **no general
  size-consistency/dissociation guarantee**, especially for overlapping tails.

The restored 239-page theorem archive and added smoothing sources are retained
in `references/`. Heat-smoothed `W_t` helpers are not a full Ts/Pauli decoder.
Three later exact primary decoder sources are still missing. See
[STATUS.md](STATUS.md), [source audit](docs/SOURCE_MAP.md),
[research integration](docs/RESEARCH_INTEGRATION.md),
[validation](docs/VALIDATION.md), and [upstream licenses/APIs](docs/THIRD_PARTY.md).

## Physics Research Console

```bash
.venv/bin/pauliforge console
```

The optional Streamlit UI includes a 3D density/nuclei view, energy decomposition,
joint matrix diagnostics, training curves, domain and evidence/source pages.
Enter a checkpoint and matching `.npz` path to inspect actual trained output.
The Atlas view currently shows numerical octants, **not** a complete interactive
certified adaptive atlas. `scripts/validate_console.py` checks a real local
server and stops only the process it starts. No public preview is published.

`examples/research_demo.ipynb` is a notebook entry point for the same tested demo
script; prepare the banks/checkpoints above and select this Python environment.
