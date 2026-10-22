"""Separate physical output heads; positivity does not imply representability."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .physics import analytic_pauli_lower_bound


def low_electron_sector(n):
    # Only a floating summation allowance, not an empirical electron threshold.
    roundoff = 8 * torch.finfo(n.dtype).eps * max(2.0, abs(float(n.detach())))
    return n <= 2.0 + roundoff


class PauliFloorHead(nn.Module):
    def __init__(self, c_lt=0.0, certified_constant=False):
        super().__init__()
        self.c_lt = float(c_lt)
        self.certified_constant = certified_constant

    def forward(self, rho, weights, tw, n):
        floor = analytic_pauli_lower_bound(rho, weights, tw, self.c_lt, self.certified_constant)
        return torch.where(low_electron_sector(n), torch.zeros_like(floor), floor)


class KineticResidualHead(nn.Module):
    def __init__(self, features, hidden=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features, ell2, n):
        result = F.softplus(self.net(features)).squeeze(-1) * torch.relu(n - 2) / ell2
        return torch.where(low_electron_sector(n), torch.zeros_like(result), result)


@dataclass
class XCHeadOutput:
    exchange: torch.Tensor
    correlation: torch.Tensor

    @property
    def exc(self):
        return self.exchange + self.correlation


class ExchangeCorrelationHead(nn.Module):
    """A distinct Hartree-unit XC surrogate, not a kinetic residual.

    Exchange is negative and scales with coordinate dilation as lambda. The
    correlation channel sees log(ell / 1 Bohr); no false homogeneous scaling
    law is imposed on the exact exchange-correlation functional.
    """

    def __init__(self, features, hidden=48):
        super().__init__()
        self.exchange_net = nn.Sequential(
            nn.Linear(features, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.correlation_net = nn.Sequential(
            nn.Linear(features + 1, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, features, ell2, n):
        ell = torch.sqrt(ell2)
        exchange = -F.softplus(self.exchange_net(features)).squeeze(-1) * n / ell
        correlation_features = torch.cat((features, torch.log(ell).reshape(1)))
        correlation = self.correlation_net(correlation_features).squeeze(-1) * n
        return XCHeadOutput(exchange, correlation)
