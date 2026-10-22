"""Finite-basis density-constrained teacher (requires cvxpy).

Approximate numerical solver, NOT a rigorous continuum D1-C certificate.
The AO values must be in a quadrature-orthonormal basis and kinetic matrix
correspond to -Delta. Both matrix occupancy and sampled target density are
constrained jointly. Check solver residuals independently.
"""
import numpy as np


def solve_finite_basis(target_density, ao_values, weights, kinetic,
                       electrons, solver="CLARABEL", tol=2e-4):
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError("pip install 'pauliforge[cert]' for finite SDP teacher") from exc
    A = np.asarray(ao_values, dtype=float)
    w = np.asarray(weights, dtype=float)
    r = np.asarray(target_density, dtype=float)
    K = np.asarray(kinetic, dtype=float)
    n_g, d = A.shape
    if w.shape != (n_g,) or r.shape != (n_g,) or K.shape != (d, d):
        raise ValueError("inconsistent AO/quadrature/kinetic sizes")
    Sgrid = A.T @ (w[:, None] * A)
    if not np.allclose(Sgrid, np.eye(d), atol=1e-4):
        raise ValueError("AO basis not quadrature-orthonormal; orthonormalize first")
    if not np.allclose(K, K.T, atol=1e-9):
        raise ValueError("kinetic must be symmetric")
    if not 0 <= electrons <= 2*d:
        raise ValueError("insufficient dimension for electrons")
    G = cp.Variable((d, d), symmetric=True)
    local_rho = cp.sum(cp.multiply(A @ G, A), axis=1)
    problem = cp.Problem(cp.Minimize(cp.trace(K @ G)), [G >> 0, 2*np.eye(d)-G >> 0,
                                                         cp.trace(G) == electrons,
                                                         local_rho == r])
    problem.solve(solver=solver)
    if problem.status not in ("optimal", "optimal_inaccurate") or G.value is None:
        raise RuntimeError("finite SDP infeasible or numerical solver failed")
    gm = (G.value + G.value.T) / 2
    eig = np.linalg.eigvalsh(gm)
    residual = max(float(np.max(np.abs(np.sum((A @ gm)*A, axis=1)-r))),
                   abs(float(np.trace(gm))-electrons), max(0.,-float(eig.min())),
                   max(0.,float(eig.max())-2.))
    if residual > tol:
        raise RuntimeError(f"teacher constraints not met: residual={residual:.3g}")
    return {"gamma": gm, "ts_minus_delta": float(np.trace(K @ gm)),
            "constraint_residual": residual, "label_quality": "numerical_finite_basis_only"}
