"""Energy interval arithmetic, with evidence separate from small numerical gaps.

The atlas inequality is CONDITIONAL on full-diagonal local searches, a complete
cover, IMS control, and an independently enclosed Weizsaecker energy. Merely
putting floats into this module does not establish any of those premises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Protocol, Sequence


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


@dataclass(frozen=True)
class LocalInterval:
    lower: float
    upper: float
    alpha: float
    source: str = "unspecified_numerical_interval"
    scope: str = "numerical"

    def __post_init__(self):
        if not _finite(self.lower, self.upper, self.alpha):
            raise ValueError("local interval endpoints and weight must be finite")
        if self.alpha < 0 or self.lower < 0 or self.upper < self.lower:
            raise ValueError("invalid local Pauli interval")
        if self.scope not in {"numerical", "finite_model", "continuum"}:
            raise ValueError("unknown interval scope")


class GlobalProof(Protocol):
    """A verifier must check actual supplied data, not return a stored flag."""

    scope: str

    def verify_global(self, intervals, tw_lower, tw_upper, ims_upper, n) -> bool: ...


@dataclass(frozen=True)
class ErrorBudget:
    """Named absolute energy errors; every entry is included in the width."""

    quadrature: float = 0.0
    basis: float = 0.0
    truncation: float = 0.0
    tail: float = 0.0
    rounding: float = 0.0
    moments: float = 0.0
    missing: tuple[str, ...] = ()

    def __post_init__(self):
        if not _finite(*self.values) or any(value < 0 for value in self.values):
            raise ValueError("error budgets must be finite and nonnegative")

    @property
    def values(self):
        return (
            self.quadrature,
            self.basis,
            self.truncation,
            self.tail,
            self.rounding,
            self.moments,
        )

    @property
    def total(self):
        return math.fsum(self.values)


@dataclass(frozen=True)
class GlobalInterval:
    lower: float
    upper: float
    width: float
    certified_at_epsilon_n: bool
    budget_satisfied: bool = False
    certified_finite_model: bool = False
    status: str = "CONDITIONAL"
    missing_assumptions: tuple[str, ...] = field(default_factory=tuple)
    scope: str = "conditional_atlas_bound"


def assemble(
    intervals: Sequence[LocalInterval],
    tw_lower: float,
    tw_upper: float,
    ims_upper: float,
    n: float,
    epsilon: float,
    *,
    errors: ErrorBudget | None = None,
    proof: GlobalProof | None = None,
) -> GlobalInterval:
    """Aggregate full-diagonal Pauli intervals and all absolute error budgets.

    ``budget_satisfied`` only evaluates a scalar inequality. The stronger
    ``certified_at_epsilon_n`` additionally needs a recognized, independently
    checked continuum proof. The implemented positive case is explicit
    disjoint low-mass polynomial packets; numerical SDP labels cannot certify
    a continuum functional. Numerical defaults are deliberately safe.
    """
    if not _finite(tw_lower, tw_upper, ims_upper, n, epsilon):
        raise ValueError("global interval data must be finite")
    if tw_lower < 0 or tw_upper < tw_lower or ims_upper < 0 or n <= 0 or epsilon <= 0:
        raise ValueError("invalid global input interval")
    errors = errors or ErrorBudget()
    lower = tw_lower + math.fsum(a.alpha * a.lower for a in intervals) - errors.total
    lower = max(0.0, lower)
    upper = tw_upper + math.fsum(a.alpha * a.upper for a in intervals) + ims_upper + errors.total
    lower = max(0.0, math.nextafter(float(lower), -math.inf))
    upper = math.nextafter(float(upper), math.inf)
    width = upper - lower
    budget_satisfied = Fraction(upper) - Fraction(lower) <= Fraction(epsilon) * Fraction(n)
    verified = False
    scope = "conditional_atlas_bound"
    missing = list(errors.missing)
    if proof is None:
        missing.extend(
            (
                "independent_local_bounds",
                "complete_full_diagonal_cover",
                "certified_ims_and_tw",
                "continuum_basis_and_tail_control",
            )
        )
    else:
        # A user object whose verify_global merely returns True is not accepted.
        from .decoders.packets import DisjointPacketCertificate

        if isinstance(proof, DisjointPacketCertificate):
            verified = proof.verify_global(intervals, tw_lower, tw_upper, ims_upper, n)
            scope = proof.scope
        if not verified:
            missing.append("proof_verification_failed_or_unsupported")
    continuum_ok = verified and scope == "continuum_explicit_packet_subclass" and not missing
    finite_ok = verified and scope == "finite_model" and not missing
    return GlobalInterval(
        lower,
        upper,
        width,
        continuum_ok and budget_satisfied,
        budget_satisfied,
        finite_ok and budget_satisfied,
        "PROVED" if continuum_ok else "CONDITIONAL",
        tuple(dict.fromkeys(missing)),
        scope,
    )


def bounded_decoder_error(
    *,
    delta: float,
    weight_sum_bound: float,
    local_energy_upper: float,
    tau: float,
    aggregate_l2_squared: float,
) -> float:
    """Source 2026-10-19 §3 (3), without classifying individual blocks.

    All inputs are bounds supplied by independent premises. This verifies their
    arithmetic domain, not the underlying analytic certificates.
    """
    values = (delta, weight_sum_bound, local_energy_upper, tau, aggregate_l2_squared)
    if not _finite(*values) or min(values) < 0 or tau == 0:
        raise ValueError("invalid bounded-decoder premises")
    return delta * weight_sum_bound + local_energy_upper * aggregate_l2_squared / tau**2


def histogram_l2_budget(
    *,
    n_electrons: float,
    subdivisions: int,
    overlap: float,
    kinetic_per_electron: float,
    beta: float,
    blocks_per_electron: float,
) -> float:
    """Source 2026-10-19 §4 (10); masses do NOT recover TW."""
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or subdivisions < 1:
        raise ValueError("subdivisions must be a positive integer")
    values = (n_electrons, overlap, kinetic_per_electron, beta, blocks_per_electron)
    if not _finite(*values) or min(values) < 0 or n_electrons <= 0:
        raise ValueError("invalid histogram budget premises")
    return (
        18 * overlap * kinetic_per_electron / subdivisions**2 + 6 * beta * blocks_per_electron
    ) * n_electrons


class CertificateAggregator:
    """Convenience façade; contains no learned confidence-derived intervals."""

    def __call__(self, intervals, **kwargs):
        return assemble(intervals, **kwargs)


def outward_fraction(value: Fraction) -> tuple[float, float]:
    """Adjacent binary floats enclosing an exact rational."""
    value = Fraction(value)
    approximation = float(value)
    exact_float = Fraction.from_float(approximation)
    lower = approximation if exact_float <= value else math.nextafter(approximation, -math.inf)
    upper = approximation if exact_float >= value else math.nextafter(approximation, math.inf)
    return lower, upper
