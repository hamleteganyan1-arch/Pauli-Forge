# Scientific and engineering status, v0.1.0 (2026-10-21)

| Property | Status | Qualification |
| --- | --- | --- |
| Matrix `0<=Gamma<=2I`, electron trace | IMPLEMENTED / TESTED | Eigen/trace check in orthonormal basis; finite precision |
| Joint finite-basis matrix + diagonal density | IMPLEMENTED / OPTIONAL | CVXPY numerical teacher, NOT rigorous continuum certification |
| `N<=2` exact Pauli sector | IMPLEMENTED / TESTED | Density samples mass N; within quadrature tolerance |
| Scalar Pauli nonnegativity | IMPLEMENTED / TESTED | By construction; does not prove gamma representability |
| Lower bound from `C_LT` | CONDITIONAL | Default safe C_LT=0; positive coefficient requires verification |
| Rotations, translations, inversion, grid permutation | IMPLEMENTED / TESTED | Scalar values on transformed same quadrature; not separate regridding |
| Coordinate uniform scaling | IMPLEMENTED / TESTED | `-Delta` convention, consistent coordinate and weight scaling |
| Fixed chemical alphabet, geometry constraints | IMPLEMENTED / TESTED | Does not certify ground-state nature of rho |
| Frozen `D_E` mathematical compactness | CONDITIONAL | Need binding/spectral stratum hypotheses from archive |
| `T_W` from supplied `grad sqrt rho` | IMPLEMENTED | Quadrature accuracy not automatically certified |
| XC-only ML, full energy | OPEN | Specified in AGENTS.md |
| Exact Pauli representability of scalar neural prediction | OPEN | Mere positive ML Pauli does not ensure a valid gamma exists |
| Exact dissociation/size consistency | OPEN | Current global invariant head not generally additive |
| Certified `epsilon N` model prediction | OPEN | Certificate aggregation implemented, *real* local certificates missing |
| Generic polynomial Coulomb decoder | OPEN | Special subclasses proved in source corpus |
| Near-linear total bits | OPEN | Prototype uses O(G^2) dense point features |
| Force accuracy / variational density accuracy | OPEN | Energy C0 training not sufficient |
