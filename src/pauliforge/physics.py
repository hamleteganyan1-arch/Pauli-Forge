"""Natural atomic units, kinetic operator -Delta (NOT PySCF -Delta/2)."""
import math
import torch

C_TF_MINUS_DELTA = (3.0 / 5.0) * (3.0 * math.pi ** 2) ** (2.0 / 3.0)


def electron_count(rho: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(rho * weights)


def von_weizsaecker(grad_sqrt_rho: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if grad_sqrt_rho.shape != (weights.numel(), 3):
        raise ValueError("grad_sqrt_rho must be [grid,3]")
    return torch.sum(weights * grad_sqrt_rho.square().sum(-1))


def critical_integral(rho: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(weights * torch.clamp(rho, min=0.0).pow(5.0 / 3.0))


def analytic_pauli_lower_bound(rho: torch.Tensor, weights: torch.Tensor,
                               tw: torch.Tensor, c_lt: float = 0.0,
                               certified_constant: bool = False) -> torch.Tensor:
    """B=max(0,C_LT int rho^(5/3)-T_W).

    Positive c_lt MUST be backed by a verified Lieb-Thirring constant in the
    same kinetic and spin convention. Default 0 is universally safe.
    """
    if c_lt < 0 or (c_lt > 0 and not certified_constant):
        raise ValueError("positive C_LT requires explicit certification")
    return torch.clamp(c_lt * critical_integral(rho, weights) - tw, min=0.0)
