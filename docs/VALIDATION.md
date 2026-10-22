# Executed validation — 2026-10-22

The final CPU suite completed **170 passed in 18.97 s**, with no skipped or
failed tests. [JUnit](validation/pytest.xml), [runner log](validation/pytest.log),
[machine summary](validation/summary.json), and the
[actual demonstration report](validation/research-demo.json) are retained.
The summary includes installed runtime versions and a digest of Python source.

## Commands actually executed

```bash
bash scripts/setup_environment.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q --junitxml=results/validation.xml
.venv/bin/ruff check src tests examples scripts
.venv/bin/ruff format --check src tests examples scripts
.venv/bin/python -m pip check
.venv/bin/python -m build --no-isolation
.venv/bin/python scripts/validate_console.py --output results/console-validation.json
```

The hash-locked setup completed and was exercised repeatedly in this instance.
It reports CPU PyTorch 2.14.1, NumPy 2.5.4, PySCF 2.14.0, CVXPY 1.9.3 and e3nn
0.6.0. Ruff and pip checks passed. Wheel/sdist builds succeeded. Shell templates
passed `bash -n`. The optional CI workflow was added but not run on GitHub.

## What the tests exercise

- PSD, occupation caps, trace and sampled-density equalities jointly; infeasible
  densities; spectral projection's failure to preserve density; complex bases,
  nonorthogonal AO transformation, spin-channel caps, independent duals.
- Exact rational primal/dual feasibility and refusal of unchecked certificates.
- FrozenDE element/count/collision/relative-radius/charge/spin rejection, stable
  version/hash, archive/domain agreement and trial/ground-state distinction.
- Real-valued exact low-N branch, rotations/translations/reflections, atom and
  grid permutations, coordinate and functional-derivative scaling.
- Nonnegative/finite/real density data, explicit mass normalization, preservation
  of raw archives, reliable potential-label requirements and fixed-N derivatives.
- RHF/RKS/UHF/UKS PySCF calls, factor 2, Gamma/AO density reconstruction,
  grid refinement, negative quadrature-weight rejection, actual joint SDP,
  and checked second-order SCF fallback for stretched RKS H2.
- Source-scoped packet/Fourier/Gram calculations, exact rational moments,
  smoothing budgets, 2:1 atlas/full diagonal/IMS, honest global proof gating.
- Separate kinetic/XC losses, deterministic checkpoints and portable JSON reload,
  family leakage refusal, CLI train/evaluate/export/sweep in separate processes.
- Every console page, actual matrix solve, loaded kinetic and XC checkpoints,
  and real Streamlit frontend/health HTTP 200. Atlas UI is numerical octants.

## Numerical demonstration

The two banks are generated separately and contain seven tiny molecular
references each. The kinetic bank uses HF and PBE references; the XC bank uses
RKS/UKS PBE throughout, because HF archives do not provide an XC density target.
Both models were trained for 30 CPU epochs with seed 2; CH4 is excluded from
training as an entire composition/charge/spin family.

| Quantity | Actual result | Evidence scope |
| --- | --- | --- |
| H2 quadrature electron error, grids 0/1/2 | 2.514e-3 / 2.192e-6 / 2.905e-9 | Empirical fixed-density refinement, not a certified continuum error. |
| Water Ts, project −Delta | 149.2009027862694 | Numerical finite-basis reference. |
| Water TW, raw grid | 115.02259113568896 | Numerical quadrature. |
| Water Pauli reference | 34.178311650580454 | Finite-basis kinetic minus grid TW. |
| Water joint SDP density/cap/trace residual | 1.7053e-13 | Full 10,120-point supplied density; numerical, not rigorous interval. |
| Water SDP primal/dual gap | 5.6843e-14 | Floating objective diagnostics; continuum_certified=False. |
| Water learned Pauli | 34.63012119505746 | Water is in training; fit example, not independent validation. |
| Held-out CH4 kinetic MAE/electron | 0.3516413074299621 | One composition, project kinetic units, empirical only. |
| Held-out CH4 XC MAE/electron | 0.07691320766602612 | One composition, Hartree, empirical only. |
| Exact tiny rational SDP bracket | [3,3] | Independently verified exact finite rational matrices. |
| Explicit disjoint packet example | N=5/2, TW=75/4, P=0 | Only declared disjoint compact packets, each mass <=2. |

A small training loss and a single held-out composition do not establish
transferability. In particular, the low-N exact branch is not a learned
accuracy benchmark. Water's ML grid was explicitly rescaled by about
1.000006425; its original AO density and teacher remain unchanged.

The notebook's Python code cell was executed using this venv and the actual
banks/checkpoints. It invokes the same tested research-demo script; no separate
Jupyter-server or browser-kernel integration is claimed.

## Limits and next step

No CUDA device or SLURM cluster was available; their execution is **UNRUN**.
Fresh-task filesystem restoration and environment publication are unverified.
No broad continuum certificate, universal polynomial decoder, general bare-model
size consistency, reliable forces or Mode B total kinetic response is claimed.
The radial baseline requires a finite positive second moment; a finite grid
cannot establish that continuum hypothesis for an arbitrary H1 density.

The next bounded task is a **finite-matrix Pauli-factor primal with an explicit
cap-I reservoir repair**, checked against joint SDP and the rational verifier.
The mathematical roadmap and exact remaining missing source obligations are
in `RESEARCH_INTEGRATION.md` and `SOURCE_MAP.md`.
