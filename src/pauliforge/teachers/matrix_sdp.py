"""Joint finite-basis constrained search with independent primal/dual diagnostics.

All spin occupations, density equalities and electron traces are hard SDP
constraints. Solver success is only NUMERICAL evidence. Returned floating
objectives are not certified continuum intervals. ``basis_metric='analytic'``
keeps analytic AO orthogonality distinct from approximate quadrature mass.
"""

import math

import numpy as np

from .dual import negative_spectrum_dual


def _prepare(target_density, ao_values, weights, kinetic, overlap, basis_metric):
    a = np.asarray(ao_values)
    k = np.asarray(kinetic)
    w = np.asarray(weights, dtype=float)
    r = np.asarray(target_density, dtype=float)
    if a.ndim != 2 or min(a.shape) == 0:
        raise ValueError("AO values must be a nonempty [grid,basis] matrix")
    ng, d = a.shape
    if w.shape != (ng,) or r.shape != (ng,) or k.shape != (d, d):
        raise ValueError("inconsistent AO/quadrature/kinetic sizes")
    if not all(np.isfinite(x).all() for x in (a, k, w, r)):
        raise ValueError("teacher data must be finite")
    if np.any(w <= 0) or np.any(r < 0):
        raise ValueError("weights must be positive and density nonnegative")
    if not np.allclose(k, k.conj().T, atol=1e-10, rtol=1e-10):
        raise ValueError("kinetic must be Hermitian")
    if np.linalg.eigvalsh((k + k.conj().T) / 2).min() < -1e-9:
        raise ValueError("kinetic -Delta must be positive semidefinite")
    if basis_metric not in ("quadrature", "analytic"):
        raise ValueError("basis_metric must be 'quadrature' or 'analytic'")
    inverse_root = None
    if overlap is not None:
        s = np.asarray(overlap)
        if s.shape != (d, d) or not np.isfinite(s).all():
            raise ValueError("overlap must be finite with the AO matrix shape")
        if not np.allclose(s, s.conj().T, atol=1e-10, rtol=1e-10):
            raise ValueError("overlap must be Hermitian")
        values, vectors = np.linalg.eigh((s + s.conj().T) / 2)
        if np.any(values <= 0):
            raise ValueError("overlap must be strictly positive definite")
        inverse_root = (vectors * values**-0.5) @ vectors.conj().T
        a = a @ inverse_root
        k = inverse_root @ k @ inverse_root
        basis_metric = "analytic"
    grid_overlap = a.conj().T @ (w[:, None] * a)
    grid_error = float(np.max(np.abs(grid_overlap - np.eye(d))))
    if basis_metric == "quadrature" and not np.allclose(
        grid_overlap, np.eye(d), atol=1e-4, rtol=1e-5
    ):
        raise ValueError(
            "AO basis not quadrature-orthonormal; use analytic metric only for independently orthonormal AOs"
        )
    return a, w, r, (k + k.conj().T) / 2, inverse_root, grid_error


def _residuals(gamma, a, target, electrons, cap):
    eig = np.linalg.eigvalsh(gamma)
    reconstructed = np.real(np.sum((a @ gamma) * a.conj(), axis=1))
    return {
        "sample_density": float(np.max(np.abs(reconstructed - target))),
        "trace": abs(float(np.trace(gamma).real) - electrons),
        "hermiticity": float(np.max(np.abs(gamma - gamma.conj().T))),
        "psd": max(0.0, -float(eig.min())),
        "cap": max(0.0, float(eig.max()) - cap),
    }, reconstructed


def _solve(
    target_density,
    ao_values,
    weights,
    kinetic,
    spin_electrons,
    max_occupation,
    solver,
    tol,
    overlap,
    basis_metric,
    solver_options,
):
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError("install 'pauliforge[cert]' for the finite SDP teacher") from exc
    if not math.isfinite(float(tol)) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if not math.isfinite(float(max_occupation)) or max_occupation <= 0:
        raise ValueError("occupation cap must be finite and positive")
    a, w, r, k, inverse_root, grid_error = _prepare(
        target_density, ao_values, weights, kinetic, overlap, basis_metric
    )
    d = k.shape[0]
    ns = [float(x) for x in spin_electrons]
    if any(not math.isfinite(n) or not 0 <= n <= max_occupation * d for n in ns):
        raise ValueError("insufficient dimension or invalid electron number")
    complex_problem = np.iscomplexobj(a) or np.iscomplexobj(k)
    variables = [
        cp.Variable((d, d), hermitian=True)
        if complex_problem
        else cp.Variable((d, d), symmetric=True)
        for _ in ns
    ]
    identity = np.eye(d)

    def real_expression(expression):
        # CVXPY 1.9 does not canonicalize redundant real(real_expression)
        # on its real-only cone path; complex reductions do need this atom.
        return cp.real(expression) if expression.is_complex() else expression

    constraints = []
    trace_constraints = []
    for g, n in zip(variables, ns):
        constraints.extend([g >> 0, max_occupation * identity - g >> 0])
        trace_constraint = real_expression(cp.trace(g)) == n
        constraints.append(trace_constraint)
        trace_constraints.append(trace_constraint)
    total = sum(variables)
    local_rho = real_expression(cp.sum(cp.multiply(a @ total, a.conj()), axis=1))
    density_constraint = local_rho == r
    constraints.append(density_constraint)
    objective = sum(real_expression(cp.trace(k @ g)) for g in variables)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    options = dict(solver_options or {})
    if solver.upper() == "CLARABEL":
        options = {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9, **options}
    elif solver.upper() == "SCS":
        options = {"eps": 1e-8, "max_iters": 30000, **options}
    problem.solve(solver=solver, **options)
    if problem.status not in ("optimal", "optimal_inaccurate") or any(
        g.value is None for g in variables
    ):
        raise RuntimeError(f"finite SDP infeasible or numerical solver failed: {problem.status}")
    gammas = [(g.value + g.value.conj().T) / 2 for g in variables]
    gamma = sum(gammas)
    residuals, reconstructed = _residuals(gamma, a, r, sum(ns), len(ns) * max_occupation)
    for index, (g, n) in enumerate(zip(gammas, ns)):
        spin_residuals, _ = _residuals(
            g, a, np.real(np.sum((a @ g) * a.conj(), axis=1)), n, max_occupation
        )
        for name in ("trace", "hermiticity", "psd", "cap"):
            residuals[f"channel_{index}_{name}"] = spin_residuals[name]
    worst = max(residuals.values())
    if worst > tol:
        raise RuntimeError(f"teacher constraints not met: residual={worst:.3g}")
    # CVXPY uses +y.(rho(G)-rho_target)+nu.(TrG-N); negate for the
    # cap-box negative-spectrum infimum. No dual PSD/eigenvalue clipping claim.
    v = -np.asarray(density_constraint.dual_value, dtype=float)
    trace_multipliers = [-float(np.real(c.dual_value)) for c in trace_constraints]
    lower = float(v @ r)
    dual_results = []
    for n, nu in zip(ns, trace_multipliers):
        bound = negative_spectrum_dual(k, a, np.zeros_like(r), n, v, nu, max_occupation)
        lower += bound["lower_numerical"]
        dual_results.append(bound)
    primal = float(sum(np.trace(k @ g).real for g in gammas))
    result = {
        "gamma": gamma,
        "ts_minus_delta": primal,
        "constraint_residual": worst,
        "residuals": residuals,
        "status": problem.status,
        "numerical_admissible": True,
        "density_target": r.copy(),
        "density_reconstructed": reconstructed,
        "ao_values": a.copy(),
        "weights": w.copy(),
        "kinetic": k.copy(),
        "electrons": sum(ns),
        "max_occupation": len(ns) * max_occupation,
        "grid_overlap_residual": grid_error,
        "quadrature_mass": float(w @ r),
        "quadrature_mass_error": float(w @ r - sum(ns)),
        "primal_upper_numerical": primal,
        "dual_lower_numerical": lower,
        "gap_numerical": primal - lower,
        "dual_density_multiplier": v,
        "dual_trace_multipliers": trace_multipliers,
        "dual_details": dual_results,
        "label_quality": "numerical_finite_basis_only",
        "scope": "finite_basis_sampled_density",
        "certified": False,
        "continuum_certified": False,
        "not_continuum_certified": True,
    }
    if inverse_root is not None:
        result["gamma_ao"] = inverse_root @ gamma @ inverse_root
    if len(gammas) == 2:
        result.update(gamma_alpha=gammas[0], gamma_beta=gammas[1], spin_electrons=tuple(ns))
    return result


def solve_finite_basis(
    target_density,
    ao_values,
    weights,
    kinetic,
    electrons,
    solver="CLARABEL",
    tol=2e-6,
    *,
    max_occupation=2.0,
    overlap=None,
    basis_metric="quadrature",
    solver_options=None,
):
    """Solve joint density/cap/trace search; optional raw AO overlap transform.

    With overlap=None, A/K must already be orthonormal. ``analytic`` records
    that this orthogonality came from analytic integrals, so approximate grid
    orthogonality and mass are diagnostics, not silently imposed identities.
    """
    return _solve(
        target_density,
        ao_values,
        weights,
        kinetic,
        (electrons,),
        max_occupation,
        solver,
        tol,
        overlap,
        basis_metric,
        solver_options,
    )


def solve_spin_resolved(
    target_density,
    ao_values,
    weights,
    kinetic,
    spin_electrons,
    solver="CLARABEL",
    tol=2e-6,
    *,
    overlap=None,
    basis_metric="quadrature",
    solver_options=None,
):
    """Joint alpha/beta search, caps I each, spin traces and total density fixed."""
    if len(spin_electrons) != 2:
        raise ValueError("spin_electrons must contain alpha and beta electron counts")
    return _solve(
        target_density,
        ao_values,
        weights,
        kinetic,
        spin_electrons,
        1.0,
        solver,
        tol,
        overlap,
        basis_metric,
        solver_options,
    )
