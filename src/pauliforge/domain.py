"""Finite chemical alphabet + compact nuclear geometry *parameter* set.

Passing this validator does NOT prove that a supplied density is a Coulomb
Hamiltonian ground-state density. Compactness of D_E requires the stated
spectral/binding hypotheses; this code checks only input-domain conditions.
"""
from dataclasses import dataclass
from typing import Tuple
import torch

@dataclass(frozen=True)
class Geometry:
    nuclear_charges: Tuple[int, ...]
    positions_bohr: torch.Tensor
    charge: int = 0

@dataclass(frozen=True)
class FrozenDE:
    allowed_atomic_numbers: Tuple[int, ...] = (1, 6, 7, 8)  # demonstration; configurable
    max_atoms: int = 8
    min_separation_bohr: float = 0.6
    max_centered_radius_bohr: float = 12.0
    allowed_charges: Tuple[int, ...] = (0,)
    min_electrons: int = 1
    max_electrons: int = 100

    def validate(self, geometry: Geometry) -> int:
        z = geometry.nuclear_charges
        x = torch.as_tensor(geometry.positions_bohr, dtype=torch.float64)
        if not (1 <= len(z) <= self.max_atoms):
            raise ValueError("atom count outside frozen domain")
        if x.shape != (len(z), 3) or not torch.isfinite(x).all():
            raise ValueError("invalid atomic positions (bohr)")
        if any(int(zi) not in self.allowed_atomic_numbers for zi in z):
            raise ValueError("element outside allowed chemical alphabet")
        if geometry.charge not in self.allowed_charges:
            raise ValueError("molecular charge not allowed")
        N = sum(int(zi) for zi in z) - geometry.charge
        if not self.min_electrons <= N <= self.max_electrons:
            raise ValueError("electron number outside domain")
        xc = x - x.mean(dim=0, keepdim=True)
        if torch.linalg.vector_norm(xc, dim=-1).max().item() > self.max_centered_radius_bohr + 1e-10:
            raise ValueError("geometry outside bounded relative-position domain")
        if len(z) > 1:
            d = torch.pdist(x)
            if d.min().item() < self.min_separation_bohr - 1e-10:
                raise ValueError("nuclear collision / separation violation")
        return N
