# Scientific contract and no-go boundaries

## Immutable units and definitions
- All spatial coordinates Bohr, density electrons/Bohr^3, integrals atomic units.
- The project's kinetic operator is **-Delta**, not **-Delta/2**. PySCF is -Delta/2: multiply its kinetic labels by **two** on import.
- `u=sqrt(rho)`, `T_W=int |grad u|^2`, `T_s=inf_{0<=gamma<=2I,diag gamma=rho} Tr(-Delta gamma)`, `P=T_s-T_W`.
- Occupations 0..2 apply to spin-summed ensemble one-particle matrices in an **orthonormal basis**. In spin-resolved channels the maximum is 1 each.
- For nonorthogonal AO DM `D`, occupation matrix is `S^(1/2) D S^(1/2)`; electron number is `Tr(SD)`.
- Projecting eigenvalues separately does **not** keep the exact diagonal density. The Pauli constraint and density equality must be enforced **simultaneously** in the constrained teacher.
- When `N<=2`, exact continuum `P=0`. For `N>2`, `P>=0`. A positive value of `C_LT` can be used only with a certified theorem constant in these conventions.

## Built-in symmetries of demo inference model
- Translations, SO(3) rotations, reflections: invariance of scalar energy from norms/distances.
- Permutation of grid nodes and permutation of equivalent atomic labels: invariance.
- Uniform coordinate scaling: `rho_lambda(x)=lambda^3 rho(lambda x)`, `weights->weights/lambda^3`, `grad_u->lambda^(5/2)grad_u`, `T_lambda=lambda^2 T`.
- Node count N preserved by scaling; density-only inference does not read atomic coordinates as ML descriptors.
- Important **NOT YET PROVEN/ENFORCED**: exact size consistency/dissociation for overlapping fragments, accurate Pauli potential `delta T/delta rho`, globally correct forces, uniform approximation, certified epsilon*N, near-linear bit cost.

## Matrix teacher / interval certification
Finite-basis `Gamma >=0`, `Gamma <=2I`, `Tr Gamma=N`, and `diag(A Gamma A.T)=rho_sample` should be solved jointly. `matrix_sdp.py` creates **numerical** finite-basis labels only. Its solver status alone is NOT a rigorous interval; use exact/rational primal and dual certificates in the next phase and propagate basis/truncation errors.

A proper atlas prediction requires honest local intervals `L_Q<=P_Q<=U_Q`, weights alpha>=0, bounded IMS remainder, `T_W` interval, and `sum_Q alpha_Q(U_Q-L_Q)+IMS_error+TW_error <= epsilon*N`. The ML point prediction by itself is NOT a certificate.

## Frozen D_E
`allowed_atomic_numbers` is a chosen finite alphabet, NOT automatically an abstract compact set of *all* densities of those elements. To invoke the compactness theorem, impose finite atom count, bounded nuclear charges, noncollision, bounded relative configurations, binding, allowed ground-state stratum/canonical ensemble and the exact physical Hamiltonian assumptions of the theorem. Treat membership in this physical ground-state density set as a provenance condition. `FrozenDE.validate()` only checks the explicit, decidable geometry/electron-number part. Distinguish `D_E` ground-state manifold from variational trial density domain; uniform minimizer guarantees require the latter.

## No cheating
Do not call KS, constrained search, exact Ts, true gamma, or quantum orbitals during density-only inference. Ground truth from KS-DFT used offline as labels must be tagged **approximate**; reference KS kinetic energy is not a generic exact Ts[rho] proof when basis/SCF accuracy is limited.
