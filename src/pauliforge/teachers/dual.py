"""Finite-model negative-spectrum duals and exact rational verification.

The spectral formula is the finite-dimensional counterpart of D1-C, section 5,
Eq. (11), in the provided weighted-labels source. It is a Ts lower bound;
subtracting an uncertified sampled T_W does not make a continuum Pauli bound.
The dedicated quantitative direct-Pauli L2(rho dx) source is not in this archive.
Its continuum theorem integration remains BLOCKED.
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral

import numpy as np

D1C_SOURCE = "references/Ts_D1C_Certified_Weighted_Labels_Full_H1_Proof_RU_2026-10-15.md#5"
DIRECT_PAULI_SOURCE_STATUS = "BLOCKED: dedicated quantitative direct-Pauli source unavailable"
_RATIONAL_VERIFICATION_TOKEN = object()


def negative_spectrum_dual(
    kinetic,
    ao_values,
    target_density,
    electrons,
    density_multiplier,
    trace_multiplier=0.0,
    max_occupation=2.0,
    operator_error_bound=0.0,
):
    """Return a *numerical* finite-basis dual value for arbitrary multipliers.

    With H=K-A*diag(v)A-tI, weak duality gives
    v.rho+tN-c Tr(H)_-. An operator uncertainty delta is paid as c*d*delta.
    Floating eigensolvers do not certify rounding; ``certified`` stays false.
    A Ritz compression without a verified omitted-tail bound is not accepted as
    a continuum negative spectral trace by this API.
    """
    k = np.asarray(kinetic)
    a = np.asarray(ao_values)
    r = np.asarray(target_density, dtype=float)
    v = np.asarray(density_multiplier, dtype=float)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] == 0:
        raise ValueError("kinetic must be a nonempty square matrix")
    if a.ndim != 2 or a.shape[1] != k.shape[0] or r.shape != (len(a),) or v.shape != r.shape:
        raise ValueError("inconsistent dual data shapes")
    if not all(np.isfinite(x).all() for x in (k, a, r, v)):
        raise ValueError("dual input must be finite")
    if not np.allclose(k, k.conj().T, atol=1e-10, rtol=1e-10):
        raise ValueError("kinetic must be Hermitian")
    if not all(
        math.isfinite(float(x))
        for x in (electrons, trace_multiplier, max_occupation, operator_error_bound)
    ):
        raise ValueError("dual scalar inputs must be finite")
    if (
        max_occupation <= 0
        or not 0 <= electrons <= max_occupation * len(k)
        or operator_error_bound < 0
    ):
        raise ValueError("invalid dual cap, electron number or error bound")
    h = k - a.conj().T @ (v[:, None] * a) - trace_multiplier * np.eye(len(k))
    h = (h + h.conj().T) / 2
    eigenvalues = np.linalg.eigvalsh(h)
    negative_trace = float(np.maximum(-eigenvalues, 0).sum())
    uncertainty = max_occupation * len(k) * operator_error_bound
    value = float(
        v @ r + trace_multiplier * electrons - max_occupation * negative_trace - uncertainty
    )
    return {
        "lower_numerical": value,
        "negative_trace_numerical": negative_trace,
        "operator_uncertainty_paid": float(uncertainty),
        "source": D1C_SOURCE,
        "scope": "finite_basis_sampled_density",
        "status": "NUMERICAL",
        "certified": False,
        "continuum_certified": False,
        "roundoff_certified": False,
    }


def _fraction(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, str):
        return Fraction(value)
    raise ValueError("exact verifier accepts integers, Fraction or rational strings; not floats")


def _matrix(values, name, shape=None):
    rows = [[_fraction(value) for value in row] for row in values]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{name} must be a nonempty rectangular rational matrix")
    if shape is not None and (len(rows), len(rows[0])) != shape:
        raise ValueError(f"{name} has incorrect shape")
    return rows


def _psd(matrix):
    """Exact Schur elimination; a zero PSD pivot forces its entire row to zero."""
    n = len(matrix)
    if any(matrix[i][j] != matrix[j][i] for i in range(n) for j in range(n)):
        return False
    a = [row[:] for row in matrix]
    for k in range(n):
        pivot = a[k][k]
        if pivot < 0:
            return False
        if pivot == 0:
            if any(a[k][j] != 0 for j in range(k + 1, n)):
                return False
            continue
        for i in range(k + 1, n):
            for j in range(i, n):
                a[i][j] -= a[i][k] * a[k][j] / pivot
                a[j][i] = a[i][j]
    return True


@dataclass(frozen=True)
class RationalFiniteCertificate:
    """Exactly verified bracket for the specified real rational finite SDP."""

    lower: Fraction
    upper: Fraction
    electrons: Fraction
    basis_dimension: int
    sampled_points: int
    source: str = D1C_SOURCE
    certified_finite_model: bool = True
    continuum_certified: bool = False
    status: str = "PROVED_FINITE_RATIONAL"
    _verification_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._verification_token is not _RATIONAL_VERIFICATION_TOKEN:
            raise ValueError("obtain a certificate through verify_rational_primal_dual")

    @property
    def width(self):
        return self.upper - self.lower

    def to_dict(self):
        return {
            "lower_exact": str(self.lower),
            "upper_exact": str(self.upper),
            "width_exact": str(self.width),
            "electrons_exact": str(self.electrons),
            "basis_dimension": self.basis_dimension,
            "sampled_points": self.sampled_points,
            "source": self.source,
            "status": self.status,
            "scope": "real_rational_finite_basis_sampled_density",
            "certified_finite_model": self.certified_finite_model,
            "continuum_certified": self.continuum_certified,
        }


def verify_rational_primal_dual(
    gamma,
    kinetic,
    ao_values,
    target_density,
    electrons,
    dual_density,
    dual_trace=0,
    lower_psd=None,
    upper_psd=None,
    max_occupation=2,
):
    """Verify exact rational primal and dual feasibility, without a solver.

    Multipliers use the convention K-S+T+nu I+A.T diag(y) A=0, with S,T PSD.
    The dual lower objective is -c Tr(T)-nu*N-y.rho. Every matrix inequality,
    trace and sampled-density equality is checked with Fraction arithmetic.
    No quadrature/basis/continuum approximation error is implicitly set to zero.
    """
    g = _matrix(gamma, "gamma")
    d = len(g)
    if len(g[0]) != d:
        raise ValueError("gamma must be square")
    k = _matrix(kinetic, "kinetic", (d, d))
    a = _matrix(ao_values, "ao_values")
    if len(a[0]) != d:
        raise ValueError("AO dimension does not match gamma")
    r = [_fraction(x) for x in target_density]
    y = [_fraction(x) for x in dual_density]
    if len(r) != len(a) or len(y) != len(a):
        raise ValueError("density and multiplier lengths must match samples")
    n, nu, cap = map(_fraction, (electrons, dual_trace, max_occupation))
    if cap <= 0 or not 0 <= n <= cap * d or any(x < 0 for x in r):
        raise ValueError("invalid exact cap, electron number or density")
    zero = [[Fraction(0) for _ in range(d)] for _ in range(d)]
    s = _matrix(lower_psd, "lower_psd", (d, d)) if lower_psd is not None else zero
    t = _matrix(upper_psd, "upper_psd", (d, d)) if upper_psd is not None else zero
    # Exact PSD tests include symmetry, unlike silently symmetrizing input.
    cap_minus_g = [[cap * (i == j) - g[i][j] for j in range(d)] for i in range(d)]
    if not _psd(g) or not _psd(cap_minus_g):
        raise ValueError("exact primal matrix violates PSD or occupation cap")
    if sum(g[i][i] for i in range(d)) != n:
        raise ValueError("exact primal trace mismatch")
    for ag, target in zip(a, r):
        if sum(ag[i] * g[i][j] * ag[j] for i in range(d) for j in range(d)) != target:
            raise ValueError("exact sampled density mismatch")
    if not _psd(s) or not _psd(t):
        raise ValueError("exact dual PSD multiplier violation")
    if any(k[i][j] != k[j][i] for i in range(d) for j in range(d)):
        raise ValueError("exact kinetic matrix must be symmetric")
    if not _psd(k):
        raise ValueError("exact kinetic -Delta matrix must be PSD")
    for i in range(d):
        for j in range(d):
            stationarity = k[i][j] - s[i][j] + t[i][j] + nu * (i == j)
            stationarity += sum(y[q] * a[q][i] * a[q][j] for q in range(len(a)))
            if stationarity != 0:
                raise ValueError("exact dual stationarity violation")
    lower = (
        -cap * sum(t[i][i] for i in range(d))
        - nu * n
        - sum(v * density for v, density in zip(y, r))
    )
    upper = sum(k[i][j] * g[j][i] for i in range(d) for j in range(d))
    if lower > upper:
        raise ValueError("exact certificate violates weak duality")
    return RationalFiniteCertificate(
        lower, upper, n, d, len(a), _verification_token=_RATIONAL_VERIFICATION_TOKEN
    )
