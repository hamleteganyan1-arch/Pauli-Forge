"""Period-two Fourier Gram SDP, source 2026-10-18 §§6--7.

This is an executable finite trial-space diagnostic. It preserves the Fourier
identity for the FULL density factor, rather than merely matching selected
rho moments. Calling a floating SDP solver does not implement the source's
rational bit-complexity proof, moment oracle, or analytic truncation bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np

SOURCE = "references/Ts_Direct_Decoder_LogLip_Gram_SDP_Proof_RU_2026-10-18.md"


@dataclass(frozen=True)
class RegularityAssumptions:
    mass_upper: float
    log_lipschitz: float
    window_lower: float = 1.0
    window_upper: float = 1.0
    window_lipschitz: float = 0.0
    source: str = "unverified_regular_subclass_promise"
    analytically_verified: bool = False

    def __post_init__(self):
        values = (
            self.mass_upper,
            self.log_lipschitz,
            self.window_lower,
            self.window_upper,
            self.window_lipschitz,
        )
        if (
            not all(math.isfinite(v) for v in values)
            or self.mass_upper < 1
            or self.log_lipschitz < 0
        ):
            raise ValueError("regular class requires M0>=1 and finite L>=0")
        if not 0 < self.window_lower <= self.window_upper or self.window_lipschitz < 0:
            raise ValueError("positive window and explicit Lipschitz bounds required")

    @property
    def density_upper(self):
        return 32 * self.mass_upper * (1 + self.log_lipschitz) ** 3

    @property
    def log2_density_lower(self):
        return -math.ceil(3 * self.log_lipschitz + 3)

    def check_samples(self, coords, rho, window, weights):
        if np.any(rho <= 0):
            raise ValueError("positive rho is required; do not insert a density floor")
        if np.any(rho > self.density_upper * (1 + 1e-12)):
            raise ValueError("density exceeds the declared regularity bound")
        if np.any((coords < 0) | (coords > 1)):
            raise ValueError("Gram subclass is defined on the unit cube")
        if np.any(window < self.window_lower - 1e-12) or np.any(window > self.window_upper + 1e-12):
            raise ValueError("window violates its declared positive bounds")
        mass = float(np.dot(weights, rho))
        if mass < 1 - 1e-9 or mass > self.mass_upper + 1e-9:
            raise ValueError("sample mass outside declared [1,M0] regular class")
        # Values at finitely many samples cannot verify a global log-Lip bound.


@dataclass(frozen=True)
class GramSDPResult:
    coefficients: np.ndarray
    pauli_numerical: float
    fourier_identity_residual: float
    minimum_coefficient_eigenvalue: float
    cap_minimum_eigenvalue: float
    solver_status: str
    moment_operator_error: float | None
    objective_error: float | None
    unmet_assumptions: tuple[str, ...]
    status: str = "NUMERICAL_REGULAR_SUBCLASS"
    certified: bool = False
    continuum_certified: bool = False
    source: str = SOURCE + " §§6--7"

    def to_dict(self):
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "coefficients"
        }


def _frequencies(frequencies):
    raw = np.asarray(frequencies)
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.shape[0] < 1:
        raise ValueError("frequencies must be a nonempty [d,3] integer array")
    if not np.isfinite(raw).all() or not np.equal(raw, np.round(raw)).all():
        raise ValueError("Fourier frequencies must be integers")
    result = raw.astype(int)
    if len({tuple(row) for row in result}) != len(result):
        raise ValueError("duplicate frequencies are not allowed")
    return result


def fourier_groups(frequencies):
    frequencies = _frequencies(frequencies)
    groups = {}
    for i, left in enumerate(frequencies):
        for j, right in enumerate(frequencies):
            groups.setdefault(tuple(left - right), []).append((i, j))
    return groups


def solve_gram_sdp(
    frequencies,
    density_moment,
    weighted_density_moment,
    assumptions,
    *,
    solver="CLARABEL",
    tolerance=2e-6,
    moment_operator_error=None,
    objective_error=None,
):
    """Solve min Tr(E A), A>=0, full Fourier identity, G A G<=2G.

    Moment callbacks take integer triples and return int rho exp(i*pi*m*x).
    Period one is deliberately not an option (source §3 is a counterexample).
    Error parameters must come from certified integrations to be meaningful;
    accepting them here never changes the result into a continuum certificate.
    """
    import cvxpy as cp

    if not isinstance(assumptions, RegularityAssumptions):
        raise ValueError("explicit regularity assumptions are required")
    frequencies = _frequencies(frequencies)
    d = len(frequencies)
    G, E = np.empty((d, d), dtype=complex), np.empty((d, d), dtype=complex)
    for i, left in enumerate(frequencies):
        for j, right in enumerate(frequencies):
            difference = tuple(right - left)
            G[i, j] = density_moment(difference)
            E[i, j] = math.pi**2 * float(np.dot(left, right)) * weighted_density_moment(difference)
    if not np.isfinite(G).all() or not np.isfinite(E).all():
        raise ValueError("finite density moments are required")
    if not np.allclose(G, G.conj().T, atol=1e-9) or not np.allclose(E, E.conj().T, atol=1e-9):
        raise ValueError("moments must satisfy Hermitian conjugation symmetry")
    G, E = (G + G.conj().T) / 2, (E + E.conj().T) / 2
    if np.linalg.eigvalsh(G).min() <= 0:
        raise ValueError("Gram matrix must be strictly positive definite")
    mass = float(np.real(density_moment((0, 0, 0))))
    if not 1 - 1e-9 <= mass <= assumptions.mass_upper + 1e-9:
        raise ValueError("moment mass violates regularity assumptions")
    for bound in (moment_operator_error, objective_error):
        if bound is not None and (not math.isfinite(bound) or bound < 0):
            raise ValueError("moment/objective errors must be finite and nonnegative")
    # Source (7.5): compensate perturbation of both Gram factors and 2G.
    error = moment_operator_error or 0.0
    if error > 1:
        raise ValueError("source robust moment analysis assumes error<=1")
    U = 8 * assumptions.density_upper
    beta = (2 + 2 * U + error) * error
    A = cp.Variable((d, d), hermitian=True)
    constraints = [A >> 0, 2 * G - G @ A @ G >> beta * np.eye(d)]
    groups = fourier_groups(frequencies)
    for difference, entries in groups.items():
        coefficient = sum(A[i, j] for i, j in entries)
        expected = float(difference == (0, 0, 0))
        constraints.append(cp.real(coefficient) == expected)
        if difference != (0, 0, 0):
            constraints.append(cp.imag(coefficient) == 0)
    problem = cp.Problem(cp.Minimize(cp.real(cp.trace(E @ A))), constraints)
    problem.solve(solver=solver)
    if problem.status not in {"optimal", "optimal_inaccurate"} or A.value is None:
        raise RuntimeError("regular Gram SDP is infeasible or numerical solution failed")
    coefficients = (A.value + A.value.conj().T) / 2
    residual = max(
        abs(sum(coefficients[i, j] for i, j in entries) - float(difference == (0, 0, 0)))
        for difference, entries in groups.items()
    )
    minimum = float(np.linalg.eigvalsh(coefficients).min())
    cap = float(np.linalg.eigvalsh(2 * G - G @ coefficients @ G).min())
    if residual > tolerance or minimum < -tolerance or cap < beta - tolerance:
        raise RuntimeError("independent Gram feasibility checks failed")
    missing = ["rational_primal_dual_verification", "analytic_fourier_truncation_budget"]
    if not assumptions.analytically_verified:
        missing.append("global_log_lipschitz_and_window_certificate")
    if moment_operator_error is None or objective_error is None:
        missing.append("certified_moment_oracle")
    return GramSDPResult(
        coefficients,
        float(np.real(np.trace(E @ coefficients))),
        float(residual),
        minimum,
        cap,
        str(problem.status),
        moment_operator_error,
        objective_error,
        tuple(missing),
    )


def solve_quadrature_gram(coords, weights, rho, window, frequencies, assumptions, **kwargs):
    coords, weights, rho, window = map(np.asarray, (coords, weights, rho, window))
    n = len(rho)
    if coords.shape != (n, 3) or weights.shape != (n,) or window.shape != (n,):
        raise ValueError("expected coords[G,3], weights[G], rho[G], window[G]")
    if not all(np.isfinite(v).all() for v in (coords, weights, rho, window)) or np.any(
        weights <= 0
    ):
        raise ValueError("finite data and positive quadrature weights required")
    assumptions.check_samples(coords, rho, window, weights)

    def moment(m, weighted=False):
        factors = rho * weights * (window if weighted else 1.0)
        return np.dot(factors, np.exp(1j * math.pi * (coords @ np.asarray(m))))

    return solve_gram_sdp(frequencies, moment, lambda m: moment(m, True), assumptions, **kwargs)


def toeplitz_gram_completion(coefficients, K, dimension=3):
    """Explicit source (5.5) completion, with independent numerical DD checks.

    ``coefficients[m]`` represents q_m. Returned B has sum_{k-l=m}B_kl=q_m.
    Positivity is accepted only if Hermitian and diagonally dominant; arbitrary
    positive trig polynomials are NOT presumed to have this particular factor.
    """
    if not isinstance(K, int) or K < 1 or dimension not in (1, 2, 3):
        raise ValueError("K>=1 and dimension in 1..3 required")
    index = tuple(product(range(4 * K + 1), repeat=dimension))
    coefficient_map = {tuple(m): complex(value) for m, value in coefficients.items()}
    if any(len(m) != dimension or max(map(abs, m)) > 2 * K for m in coefficient_map):
        raise ValueError("q must have frequencies |m|_infinity<=2K")
    B = np.zeros((len(index), len(index)), dtype=complex)
    for i, left in enumerate(index):
        for j, right in enumerate(index):
            difference = tuple(a - b for a, b in zip(left, right))
            multiplicity = math.prod(4 * K + 1 - abs(m) for m in difference)
            B[i, j] = coefficient_map.get(difference, 0) / multiplicity
    if not np.allclose(B, B.conj().T, atol=1e-12):
        raise ValueError("completion coefficients must define a real polynomial")
    margins = B.diagonal().real - (np.abs(B).sum(1) - np.abs(B.diagonal()))
    if margins.min() < -1e-12:
        raise ValueError("source diagonal-dominance premise is not satisfied")
    return index, B
