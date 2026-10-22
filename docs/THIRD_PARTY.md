# Upstream review — 2026-10-22

Licenses and public interfaces were inspected from official upstream revisions.
No upstream implementation source is copied into PauliForge. Dependency wheel
license notices remain part of their installed distributions; this review does
not grant a license to the user's original research or determine its license.

| Project | License verified | Public API / decision |
| --- | --- | --- |
| [e3nn](https://github.com/e3nn/e3nn/tree/2aa7f58440a06b15352a2cbce01fa4c26f824969) | MIT, official LICENSE | Optional tested density-grid `o3.spherical_harmonics` encoder. Installed/tested 0.6.0. |
| [PySCF](https://github.com/pyscf/pyscf/tree/143946e356b5a469dddb7c5a71220d35434407cf) | Apache-2.0, official LICENSE | Offline `gto`, RHF/UHF/RKS/UKS, `dft.numint.eval_ao/eval_rho`, AO integrals and density matrices. Installed/tested 2.14.0. |
| [STRUCTURES25](https://github.com/sciai-lab/structures25/tree/a6c4c0b0e73cad5d3b89dbf8d6194b274c6cc4b6) | LGPL-3.0, official LICENSE | Actual density-functional framework; package name is `mldft`, not `structures25`. Upstream pins torch 2.4.1/PySCF 2.4.0 and graph extensions. Keep in a separate environment for an independent benchmark; no untested wrapper pretends to support its incompatible dependency set. |
| [MACE](https://github.com/ACEsuit/mace/tree/7913afb6fa729878d20886b84a2079523727d958) | MIT, official LICENSE.md | Reviewed public ASE `mace.calculators`/training interfaces and equivariant message-passing design. Nuclear interatomic energies do not provide a density-only kinetic functional, so MACE is not substituted for one. |

The STRUCTURES25 comparison is **OPEN**, not a completed benchmark. Its modern
implementation and basis choice may be scientifically useful, but numerical
comparison requires a shared density representation, unit convention, split,
and independently prepared environment. Equivariant features alone do not
certify representability or variational accuracy.

The Linux/Python 3.12 CPU dependency lock pins complete resolved versions and
distribution hashes. GPU installation is a separate documented PyTorch backend
choice; CUDA execution is not claimed on a machine without a CUDA device.
