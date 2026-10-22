"""Atomic-unit density integrals with kinetic operator ``-Delta``.

Positive Lieb--Thirring floors are conditional on external theorem evidence;
the unconditional default coefficient is zero.
"""

import math
from dataclasses import dataclass

import torch

C_TF_MINUS_DELTA = (3.0 / 5.0) * (3.0 * math.pi**2) ** (2.0 / 3.0)


@dataclass(frozen=True)
class LTConstantEvidence:
    """External hypotheses, not a self-verifying theorem certificate.

    ``assumptions_checked`` records a caller assertion. The theorem source and
    its application still require mathematical review. A Thomas--Fermi
    coefficient is not automatically a valid Lieb--Thirring constant.
    """

    coefficient: float
    source: str
    kinetic_operator: str = "-Delta"
    spin_cap: float = 2.0
    assumptions_checked: bool = False

    def __post_init__(self):
        if not math.isfinite(self.coefficient) or self.coefficient <= 0:
            raise ValueError("theorem coefficient must be finite and positive")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("a specific theorem source is required")
        if self.kinetic_operator != "-Delta" or self.spin_cap != 2.0:
            raise ValueError("theorem conventions must be -Delta and spin-summed cap 2")
        if self.assumptions_checked is not True:
            raise ValueError("the theorem hypotheses must be explicitly checked")


def _quadrature(values: torch.Tensor, weights: torch.Tensor) -> None:
    if values.ndim != 1 or weights.shape != values.shape or values.numel() == 0:
        raise ValueError("nonempty density and weights must both have shape [grid]")
    if values.is_complex() or weights.is_complex():
        raise ValueError("density and quadrature weights must be real")
    if not torch.isfinite(values).all() or not torch.isfinite(weights).all():
        raise ValueError("density and quadrature weights must be finite")
    if torch.any(values < 0) or torch.any(weights <= 0):
        raise ValueError("density must be nonnegative and weights strictly positive")


def electron_count(rho: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    _quadrature(rho, weights)
    return torch.sum(rho * weights)


def von_weizsaecker(grad_sqrt_rho: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if weights.ndim != 1 or not weights.numel():
        raise ValueError("nonempty quadrature weights must have shape [grid]")
    if grad_sqrt_rho.shape != (weights.numel(), 3):
        raise ValueError("grad_sqrt_rho must be [grid,3]")
    if weights.is_complex() or grad_sqrt_rho.is_complex():
        raise ValueError("density gradients and weights must be real")
    if not torch.isfinite(grad_sqrt_rho).all() or not torch.isfinite(weights).all():
        raise ValueError("density gradients and weights must be finite")
    if torch.any(weights <= 0):
        raise ValueError("quadrature weights must be strictly positive")
    return torch.sum(weights * grad_sqrt_rho.square().sum(-1))


def critical_integral(rho: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    _quadrature(rho, weights)
    return torch.sum(weights * rho.pow(5.0 / 3.0))


def analytic_pauli_lower_bound(
    rho: torch.Tensor,
    weights: torch.Tensor,
    tw: torch.Tensor,
    c_lt: float = 0.0,
    certified_constant: bool | LTConstantEvidence = False,
) -> torch.Tensor:
    """Conditional ``max(0, C_LT int rho^(5/3) - T_W)``.

    Positive coefficients require structured theorem evidence and remain
    CONDITIONAL. A boolean toggle does not verify a theorem. The exact N<=2
    override belongs to the functional head, not this inequality.
    """
    if not math.isfinite(c_lt) or c_lt < 0:
        raise ValueError("C_LT must be finite and nonnegative")
    if tw.numel() != 1 or not torch.isfinite(tw).all() or tw.detach().item() < 0:
        raise ValueError("T_W must be a finite nonnegative scalar")
    if c_lt > 0:
        if not isinstance(certified_constant, LTConstantEvidence):
            raise ValueError("positive C_LT requires explicit structured theorem evidence")
        if certified_constant.coefficient != c_lt:
            raise ValueError("coefficient differs from theorem evidence")
    return torch.clamp(c_lt * critical_integral(rho, weights) - tw, min=0.0)
