"""Heat-smoothed Weizsaecker helpers from the supplied October 20 reports.

These functions concern W_t(mu), NOT Ts, the Pauli constrained search, or a
universal polynomial decoder. General error bounds remain conditional on the
supplied analytic premises and do not certify floating-point rounding.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

import torch

ENDPOINT_SOURCE = (
    "references/1-Ts_M1_Endpoint_Smoothing_and_Polylog_Bandwidth_Atomic_Algorithm_RU_2026-10-20.md"
)
SPARSE_SOURCE = (
    "references/2-Ts_M1_Five_Agent_Sparse_Integration_Search_and_Stability_RU_2026-10-20.md"
)


@dataclass(frozen=True)
class ConditionalSmoothingBound:
    """Numerical evaluation of a theorem bound, not a verified certificate."""

    upper: float
    source: str
    assumptions: tuple[str, ...]
    scope: str = "weizsaecker_heat_smoothed_measure"
    status: str = "CONDITIONAL"
    certified: bool = False
    roundoff_certified: bool = False

    def __post_init__(self):
        if not math.isfinite(self.upper) or self.upper < 0:
            raise ValueError("smoothing bound must be finite and nonnegative")


def _finite_positive(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def fractional_smoothing_error(time, exponent, seminorm_squared):
    """Source endpoint report §3 (3.1): 13 t^a ||u||_Hdot^(2).

    Requires u>=0 in H^(1+a), 0<a<1, and an independently valid bound for
    the homogeneous seminorm squared. Sampled gradients alone do not supply it.
    """
    time = _finite_positive("time", time)
    exponent = float(exponent)
    seminorm_squared = float(seminorm_squared)
    if not math.isfinite(exponent) or not 0 < exponent < 1:
        raise ValueError("fractional exponent must lie strictly between zero and one")
    if not math.isfinite(seminorm_squared) or seminorm_squared < 0:
        raise ValueError("homogeneous seminorm bound must be finite and nonnegative")
    return ConditionalSmoothingBound(
        13 * time**exponent * seminorm_squared,
        ENDPOINT_SOURCE + " §3 (3.1)",
        ("u>=0 in H^(1+a)", "independent homogeneous seminorm bound"),
    )


def source_transport_stability(total_mass, time, dropped_mass=0.0, transport_cost=0.0):
    """Sparse report §5 (5.2), including its zero and q>N cases.

    A transport plan must join the retained source and replacement measure.
    ``transport_cost`` bounds int |x-y| dplan, rather than squared distance.
    The scalar inputs do not prove the existence or validity of that plan.
    """
    total_mass = _finite_positive("total mass upper bound", total_mass)
    time = _finite_positive("time", time)
    dropped_mass, transport_cost = float(dropped_mass), float(transport_cost)
    if not all(math.isfinite(v) and v >= 0 for v in (dropped_mass, transport_cost)):
        raise ValueError("dropped mass and transport cost must be finite and nonnegative")
    if dropped_mass > total_mass:
        raise ValueError("dropped mass exceeds source mass upper bound")
    q = dropped_mass + transport_cost / math.sqrt(time)
    universal = 3 * total_mass / (8 * time)
    if q == 0:
        upper = 0.0
    elif q > total_mass:
        upper = universal
    else:
        upper = min(universal, 6 * q / time * (1 + math.log(total_mass) - math.log(q)))
    return ConditionalSmoothingBound(
        upper,
        SPARSE_SOURCE + " §5 (5.2)",
        (
            "nonnegative finite measures of mass <=N",
            "valid dropped-source decomposition",
            "verified transport-plan first-moment bound",
        ),
    )


def single_atom_weizsaecker_exact(mass, time):
    """Exact rational W_t(mass*delta_a)=3*mass/(8*time) in three dimensions.

    The identity follows by integrating |x-a|² G_t/(16t²). It does not assert
    Ts=W_t for mass>2. Only exact integer/string/Fraction parameters are accepted.
    """
    if any(isinstance(value, (float, bool)) for value in (mass, time)):
        raise ValueError("exact heat energy requires rational inputs, not floats or booleans")
    mass, time = Fraction(mass), Fraction(time)
    if mass < 0 or time <= 0:
        raise ValueError("nonnegative mass and positive heat time required")
    return 3 * mass / (8 * time)


def gaussian_atom_samples(coords, centers, masses, time):
    """Density and analytic grad sqrt(rho) of a finite heat-kernel mixture.

    Dense O(grid*atoms) diagnostic evaluator; this is not an implementation of
    the report's interval/polylogarithmic atomic integration algorithm.
    """
    time = _finite_positive("time", time)
    coords = torch.as_tensor(coords)
    centers = torch.as_tensor(centers, dtype=coords.dtype, device=coords.device)
    masses = torch.as_tensor(masses, dtype=coords.dtype, device=coords.device)
    if not coords.is_floating_point() or coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("real floating coords[grid,3] required")
    if centers.ndim != 2 or centers.shape[1] != 3 or masses.shape != (len(centers),):
        raise ValueError("centers[atoms,3] and masses[atoms] required")
    if not len(centers) or not all(torch.isfinite(x).all() for x in (coords, centers, masses)):
        raise ValueError("nonempty finite atom list and finite coordinates required")
    if torch.any(masses <= 0):
        raise ValueError("atom masses must be positive")
    displacement = coords[:, None, :] - centers[None, :, :]
    logits = masses.log()[None, :] - displacement.square().sum(-1) / (4 * time)
    log_density = torch.logsumexp(logits, dim=1) - 1.5 * math.log(4 * math.pi * time)
    density = log_density.exp()
    score = -(logits.softmax(1)[:, :, None] * displacement).sum(1) / (2 * time)
    grad_u = 0.5 * (0.5 * log_density).exp()[:, None] * score
    return density, grad_u
