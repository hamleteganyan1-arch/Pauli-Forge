"""Versioned chemical parameter domains, not ground-state-density oracles.

Membership in the physical ground-state manifold D_E additionally requires
binding/Hamiltonian and ground-state provenance hypotheses. Passing this gate
never proves them. Spin validates the reference sector, not a different
spin-summed ensemble kinetic functional.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from numbers import Integral
from typing import Mapping, Tuple

import torch


def _integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class Geometry:
    nuclear_charges: Tuple[int, ...]
    positions_bohr: torch.Tensor
    charge: int = 0
    spin: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "nuclear_charges", tuple(self.nuclear_charges))


@dataclass(frozen=True)
class FrozenDE:
    """Immutable HCNO-v1 configuration; all settings contribute to its hash.

    Changing settings changes domain_hash and requires retraining/revalidation.
    theorem_status is metadata, never a certification-enabling switch.
    """

    allowed_atomic_numbers: Tuple[int, ...] = (1, 6, 7, 8)
    max_atoms: int = 8
    min_separation_bohr: float = 0.6
    max_centered_radius_bohr: float = 12.0
    allowed_charges: Tuple[int, ...] = (0,)
    min_electrons: int = 1
    max_electrons: int = 100
    allowed_spins: Tuple[int, ...] = (0, 1, 2)
    version: str = "hcno-v1"
    units: str = "Bohr; kinetic energy = -Delta"
    theorem_status: str = "conditional_on_binding_and_stratum_assumptions"

    def __post_init__(self):
        for name in ("allowed_atomic_numbers", "allowed_charges", "allowed_spins"):
            values = tuple(_integer(v, name) for v in getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be nonempty without duplicates")
            object.__setattr__(self, name, tuple(sorted(values)))
        if min(self.allowed_atomic_numbers) <= 0:
            raise ValueError("nuclear charges must be positive")
        for name in ("max_atoms", "min_electrons", "max_electrons"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.max_atoms < 1 or not 1 <= self.min_electrons <= self.max_electrons:
            raise ValueError("invalid atom/electron limits")
        if min(self.allowed_spins) < 0:
            raise ValueError("allowed_spins uses nonnegative 2*S sectors")
        for name in ("min_separation_bohr", "max_centered_radius_bohr"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("domain version must be nonempty")
        if self.units != "Bohr; kinetic energy = -Delta":
            raise ValueError("FrozenDE requires Bohr and kinetic operator -Delta")
        if not isinstance(self.theorem_status, str):
            raise ValueError("theorem_status must be descriptive text")

    def to_dict(self) -> dict:
        result = asdict(self)
        for name in ("allowed_atomic_numbers", "allowed_charges", "allowed_spins"):
            result[name] = list(result[name])
        return result

    @classmethod
    def from_dict(cls, config: Mapping) -> "FrozenDE":
        if not isinstance(config, Mapping):
            raise ValueError("domain configuration must be a mapping")
        unknown = set(config) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown domain settings: {sorted(unknown)}")
        return cls(**dict(config))

    @property
    def domain_hash(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self.domain_hash

    def resolved_spin(self, geometry: Geometry) -> int:
        n = sum(_integer(z, "atomic number") for z in geometry.nuclear_charges) - _integer(
            geometry.charge, "charge"
        )
        return n % 2 if geometry.spin is None else _integer(geometry.spin, "spin")

    def validate(self, geometry: Geometry) -> int:
        z = tuple(_integer(value, "atomic number") for value in geometry.nuclear_charges)
        charge = _integer(geometry.charge, "charge")
        x = torch.as_tensor(geometry.positions_bohr)
        if not (1 <= len(z) <= self.max_atoms):
            raise ValueError("atom count outside frozen domain; no applicability guarantee")
        if x.is_complex() or x.shape != (len(z), 3) or not torch.isfinite(x).all():
            raise ValueError("invalid atomic positions (bohr)")
        x = x.to(dtype=torch.float64)
        if any(zi not in self.allowed_atomic_numbers for zi in z):
            raise ValueError(
                "element outside allowed chemical alphabet; no applicability guarantee"
            )
        if charge not in self.allowed_charges:
            raise ValueError("molecular charge not allowed")
        n = sum(z) - charge
        if not self.min_electrons <= n <= self.max_electrons:
            raise ValueError("electron number outside domain")
        spin = self.resolved_spin(geometry)
        if spin not in self.allowed_spins or spin > n or (n - spin) % 2:
            raise ValueError("spin sector outside domain or incompatible with electron parity")
        centered = x - x.mean(dim=0, keepdim=True)
        if (
            torch.linalg.vector_norm(centered, dim=-1).max().item()
            > self.max_centered_radius_bohr + 1e-10
        ):
            raise ValueError("geometry outside bounded relative-position domain")
        if len(z) > 1 and torch.pdist(x).min().item() < self.min_separation_bohr - 1e-10:
            raise ValueError("nuclear collision / separation violation")
        return n
