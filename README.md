# PauliForge — research prototype

**Not a certified universal KEDF.** A working, test-covered starting implementation plus an actionable Codex implementation contract.

## Core idea
`T_s[rho] = T_W[rho] + P[rho]`. A physical baseline enforces `P>=0`, exact `P=0` for `N<=2`, and optional justified Lieb–Thirring Pauli lower bound. Density-only ML predicts a remaining positive contribution. Training targets may come from finite-basis Pauli-safe constrained optimization or (approximate) KS references. Exact Pauli cap is enforced at the **teacher matrix level**, never by asserting that scalar positivity proves `gamma<=2I`.

**Longer-term alternative:** compute constrained Pauli explicitly and train only `E_xc^ML`; requires implementing an `xc` branch and a certified continuum kinetic solver. This version is a kinetics-focused **proof of concept**; the proposed full solver is specified in `AGENTS.md`.

## Tested functionality
- Frozen finite element set and bounded, noncolliding nuclear geometry input validator.
- PSD, max 2 occupation, electron trace checks and Frobenius spectral projection, including AO overlap transformation.
- Joint density/matrix semidefinite teacher (optional CVXPY, **numerical**, not mathematically certified).
- `T_W` evaluated from density-derived `grad(sqrt(rho))`, `C_LT` lower barrier default safe zero.
- Density-only neural head with exactly constructed translation/rotation/reflection, grid-node permutation, and coordinate-scaling invariances for consistent quadrature samples.
- Basic PyTorch training from precomputed `.npz` with geometry/domain validation.
- Weighted interval aggregation with IMS error, independently of ML scores.
- Basic unit and metamorphic invariance tests.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
python examples/smoke.py
```
Optional: `pip install -e '.[qc,cert]'`. PySCF/HF/KS teacher is meant for OFFLINE data. Example:
```bash
python examples/export_water_pyscf.py
pauliforge train . --epochs 20 --output model.pt --domain configs/frozen_de.example.json
```
Training .npz requires `coords,weights,rho,grad_u,pauli_target,atomic_numbers,positions_bohr,charge` and atomic units as documented. Save one sample per file and use independent chemistry splits for validation. The demo 3-electron Gaussian is synthetic and its arbitrary untrained network output is **not** ground truth.

## Technical choices
- Research baseline uses PyTorch with **O(G^2)** pairwise distance features, suitable only for small grids. Replace with sparse E(3)-equivariant, locality-controlled kernels for production and test size consistency.
- `e3nn` or selected `STRUCTURES25` components can be optional implementations with proper license review, not mandatory dependencies for v0.
- PySCF generates numerical reference densities and kinetic labels. Kinetic labels need factor **2** relative to PySCF's default operator.
- Input nuclear coordinates/element labels are used for domain gating, *not* as a shortcut to the ML density-only KEDF.

## Folders
- `src/pauliforge/domain.py`: frozen chemical space K and electronic range
- `src/pauliforge/matrix.py`: actual matrix Pauli cap, trace and AO orthogonalization
- `src/pauliforge/model.py`: group-invariant density-only proof-of-concept
- `src/pauliforge/teachers/`: numerical SDP and PySCF offline reference exporters
- `src/pauliforge/certificates.py`: global interval aggregation
- `docs/PHYSICS_CONTRACT.md`: mathematical assumptions and explicit open gaps
- `AGENTS.md`: complete Codex implementation assignment and acceptance criteria
- `docs/DESIGN_OPTIONS.md`: proposed visual/research UI design concepts

## Reference implementations and their caveats
- [STRUCTURES25](https://github.com/sciai-lab/structures25) — actual ML orbital-free DFT project, LGPL-3.0. Prefer using it as a separate adapter and reviewing license compatibility before copying code.
- [e3nn](https://github.com/e3nn/e3nn) — E(3) equivariant layers.
- [MACE](https://github.com/ACEsuit/mace) — E(3) equivariant design principles; *interatomic* force-field model does not equal kinetic density functional.
- [PySCF](https://pyscf.org) — numerical QC references, AO density/gradient and kinetic integrals.

See `AGENTS.md` for the full code and verification plan. The Code author must avoid advertising mathematical proof of wide-class polynomial decoding until assumptions have been discharged and certificates run end-to-end.
