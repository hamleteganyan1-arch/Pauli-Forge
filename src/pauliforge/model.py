"""Density-only differentiable KEDF prototype with built-in symmetries.

Inputs are density rho at lab-frame points and volume quadrature weights;
grad_sqrt_rho is supplied/computed from a density representation, never from
an orbital hidden state. Pairwise point costs O(G^2): a research baseline,
not the promised near-linear algorithm or certified physical decoder.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .physics import electron_count, von_weizsaecker, analytic_pauli_lower_bound, critical_integral

@dataclass
class KEDFOutput:
    ts: torch.Tensor
    tw: torch.Tensor
    pauli: torch.Tensor
    analytical_pauli_floor: torch.Tensor
    learned_pauli: torch.Tensor
    n: torch.Tensor

class PauliKEDF(nn.Module):
    def __init__(self, hidden: int = 48, scales=(0.5, 1.0, 2.0, 4.0),
                 c_lt: float = 0.0, certified_lt_constant: bool = False):
        super().__init__()
        self.scales = tuple(float(x) for x in scales)
        self.c_lt = c_lt
        self.certified_lt_constant = certified_lt_constant
        # dimensionless scalar invariants only; size scales lambda^-2 explicitly
        self.net = nn.Sequential(nn.Linear(len(self.scales) + 2, hidden),
                                 nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, 1))

    def descriptors(self, coords, weights, rho):
        if coords.ndim != 2 or coords.shape[1] != 3 or rho.ndim != 1 or weights.shape != rho.shape or coords.shape[0] != rho.shape[0]:
            raise ValueError("coords[G,3], weights[G], rho[G] required")
        if torch.any(rho < 0) or torch.any(weights <= 0) or not torch.isfinite(coords).all():
            raise ValueError("density nonnegative, quadrature weights positive, finite coords")
        q = weights * rho
        n = q.sum()
        if n.item() <= 1e-12:
            raise ValueError("zero density mass")
        center = (q[:, None] * coords).sum(0) / n
        centered = coords - center
        ell2 = (q * centered.square().sum(-1)).sum() / n
        ell = torch.sqrt(torch.clamp(ell2, min=1e-25))
        r2 = torch.cdist(centered, centered).square() / ell.square()
        normalized_q = q / n
        pair = normalized_q[:, None] * normalized_q[None, :]
        kernels = [(pair * torch.exp(-r2 / (2.0 * s * s))).sum() for s in self.scales]
        # Int rho^(5/3) has kinetic dimension length^-2; make dimensionless.
        i53 = critical_integral(rho, weights) * ell2 / n
        features = torch.stack([torch.log1p(n), torch.log1p(torch.clamp(i53, min=0.0)), *kernels])
        return features, ell2, n

    def forward(self, coords: torch.Tensor, weights: torch.Tensor, rho: torch.Tensor,
                grad_sqrt_rho: torch.Tensor) -> KEDFOutput:
        feats, ell2, n = self.descriptors(coords, weights, rho)
        tw = von_weizsaecker(grad_sqrt_rho, weights)
        floor = analytic_pauli_lower_bound(rho, weights, tw, self.c_lt,
                                           self.certified_lt_constant)
        # Explicit uniform coordinate scaling covariance: ell^-2.
        # This positive head satisfies a lower bound, NOT the matrix cap by itself.
        learnt = F.softplus(self.net(feats)).squeeze(-1) / ell2 * torch.relu(n - 2.0)
        # N<=2 exact sector (continuous at threshold for the learned component).
        pauli = torch.where(n <= 2.0 + 1e-10, torch.zeros_like(learnt), floor + learnt)
        return KEDFOutput(ts=tw + pauli, tw=tw, pauli=pauli,
                          analytical_pauli_floor=floor, learned_pauli=learnt, n=n)
