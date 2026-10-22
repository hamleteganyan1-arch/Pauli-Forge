"""An explicit exact packet subclass and a conditional mass-table decoder.

The separately named 2026-10-18 weighted-packet proof is NOT in the archive.
This module does not manufacture that theorem. Its exact subclass follows
rank-one admissibility and disjoint H1_0 direct sums, stated in the supplied
research audit. Each root is a tensor product of compact quadratic factors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from ..certificates import LocalInterval, assemble, outward_fraction

MISSING_WEIGHTED_PACKET_SOURCE = "Ts_Polynomial_Weighted_Adaptive_Packet_Decoder_RU_2026-10-18.md"


def _rational(value) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError(
            "exact packet parameters require integers, strings, or Fraction, not floats"
        )
    result = Fraction(value)
    return result


@dataclass(frozen=True)
class PolynomialPacket:
    """rho=u²; u=C product_i(1-((x_i-center_i)/half_width_i)²)_+.

    ``mass`` is exact. The zero trace at the box boundary makes the extension
    H1(R3), unlike a discontinuous constant-density top hat.
    """

    center: tuple[Fraction, Fraction, Fraction]
    half_width: tuple[Fraction, Fraction, Fraction]
    mass: Fraction

    def __post_init__(self):
        if len(self.center) != 3 or len(self.half_width) != 3:
            raise ValueError("packet geometry must be three-dimensional")
        object.__setattr__(self, "center", tuple(_rational(x) for x in self.center))
        object.__setattr__(self, "half_width", tuple(_rational(x) for x in self.half_width))
        object.__setattr__(self, "mass", _rational(self.mass))
        if self.mass <= 0 or min(self.half_width) <= 0:
            raise ValueError("packet mass and half widths must be positive")

    @property
    def bounds(self):
        return tuple((c - h, c + h) for c, h in zip(self.center, self.half_width))

    @property
    def tw_exact(self) -> Fraction:
        # ∫[-1,1](1-t²)²=16/15; ∫[-1,1](-2t)²=8/3.
        return self.mass * Fraction(5, 2) * sum((1 / h**2 for h in self.half_width), Fraction())

    def evaluate(self, coords):
        """Return rho and analytic grad sqrt(rho), using density parameters only."""
        import torch

        center = coords.new_tensor([float(v) for v in self.center])
        width = coords.new_tensor([float(v) for v in self.half_width])
        t = (coords - center) / width
        inside = (t.abs() < 1).all(-1)
        factors = torch.clamp(1 - t.square(), min=0)
        denominator = Fraction(16, 15) ** 3 * math.prod(self.half_width)
        amplitude = math.sqrt(float(self.mass / denominator))
        u = amplitude * factors.prod(-1)
        gradient_parts = []
        for axis in range(3):
            others = [j for j in range(3) if j != axis]
            value = -2 * amplitude * t[:, axis] / width[axis]
            value = value * factors[:, others].prod(-1)
            gradient_parts.append(torch.where(inside, value, torch.zeros_like(value)))
        return u.square(), torch.stack(gradient_parts, -1)


def _disjoint(packets: Sequence[PolynomialPacket]) -> bool:
    return all(
        any(a1 <= b0 or b1 <= a0 for (a0, a1), (b0, b1) in zip(left.bounds, right.bounds))
        for i, left in enumerate(packets)
        for right in packets[i + 1 :]
    )


@dataclass(frozen=True)
class DisjointPacketCertificate:
    packets: tuple[PolynomialPacket, ...]
    scope: str = "continuum_explicit_packet_subclass"
    source: str = "references/analysis_uploaded.md §D5.2; rank-one cap and exact disjoint-support construction"

    def __post_init__(self):
        object.__setattr__(self, "packets", tuple(self.packets))
        if not self.packets or not all(isinstance(p, PolynomialPacket) for p in self.packets):
            raise ValueError("explicit polynomial packets required")
        if any(p.mass > 2 for p in self.packets):
            raise ValueError("exact decoder supports only individual packet mass <=2")
        if not _disjoint(self.packets):
            raise ValueError("packet supports overlap; exact composition is not justified")
        if self.scope != "continuum_explicit_packet_subclass":
            raise ValueError("cannot relabel packet proof scope")

    @property
    def n_exact(self):
        return sum((p.mass for p in self.packets), Fraction())

    @property
    def tw_exact(self):
        return sum((p.tw_exact for p in self.packets), Fraction())

    def verify_global(self, intervals, tw_lower, tw_upper, ims_upper, n):
        # Recompute all conditions rather than trusting a status flag.
        if (
            not self.packets
            or not _disjoint(self.packets)
            or any(not 0 < p.mass <= 2 for p in self.packets)
        ):
            return False
        if len(intervals) != len(self.packets) or any(
            i.lower != 0 or i.upper != 0 for i in intervals
        ):
            return False
        # Local Pauli values are identically zero; all nonnegative weights work.
        if any(i.alpha < 0 for i in intervals) or ims_upper < 0:
            return False
        try:
            return Fraction(n) == self.n_exact and Fraction(tw_lower) <= self.tw_exact <= Fraction(
                tw_upper
            )
        except (ValueError, TypeError, OverflowError):
            return False


@dataclass(frozen=True)
class PacketDecodeResult:
    n_exact: Fraction
    tw_exact: Fraction
    pauli_exact: Fraction
    intervals: tuple[LocalInterval, ...]
    proof: DisjointPacketCertificate
    status: str = "PROVED_EXPLICIT_SUBCLASS"
    general_weighted_packet_status: str = "BLOCKED_MISSING_SOURCE"

    def global_interval(self, epsilon=1e-8):
        low, high = outward_fraction(self.tw_exact)
        return assemble(self.intervals, low, high, 0.0, self.n_exact, epsilon, proof=self.proof)

    def to_dict(self):
        interval = self.global_interval()
        return {
            "n": str(self.n_exact),
            "tw_exact": str(self.tw_exact),
            "pauli_exact": str(self.pauli_exact),
            "ts_lower": interval.lower,
            "ts_upper": interval.upper,
            "status": self.status,
            "certified_at_epsilon_n": interval.certified_at_epsilon_n,
            "scope": self.proof.scope,
            "general_weighted_packet_status": self.general_weighted_packet_status,
            "missing_source": MISSING_WEIGHTED_PACKET_SOURCE,
        }


def decode_low_mass_packets(packets: Sequence[PolynomialPacket]) -> PacketDecodeResult:
    proof = DisjointPacketCertificate(tuple(packets))
    intervals = tuple(
        LocalInterval(0.0, 0.0, 1.0, source=proof.source, scope="continuum") for _ in packets
    )
    return PacketDecodeResult(proof.n_exact, proof.tw_exact, Fraction(), intervals, proof)


@dataclass(frozen=True)
class MassTableResult:
    value: float
    error_bound_conditional: float
    status: str = "CONDITIONAL_ON_TABLE_AND_MODULUS"
    certified: bool = False


class PacketMassTable:
    """Linear interpolation with an explicit, conditional Lipschitz error.

    Given valid endpoint label errors <=e and a true Lipschitz modulus L,
    interpolation error is <= e+L*mesh_width/2. Neither those labels nor L is
    inferred from samples. This is not the unavailable weighted-packet proof.
    """

    def __init__(
        self,
        masses,
        values,
        *,
        lipschitz_constant,
        label_error=0.0,
        source="external_unverified_labels",
    ):
        self.masses, self.values = tuple(map(float, masses)), tuple(map(float, values))
        self.lipschitz_constant, self.label_error = float(lipschitz_constant), float(label_error)
        self.source = str(source)
        if len(self.masses) < 2 or len(self.masses) != len(self.values):
            raise ValueError("at least two aligned mass/value entries required")
        if not all(
            math.isfinite(x)
            for x in (*self.masses, *self.values, self.lipschitz_constant, self.label_error)
        ):
            raise ValueError("table and errors must be finite")
        if (
            min(self.masses) < 0
            or min(self.values) < 0
            or self.lipschitz_constant < 0
            or self.label_error < 0
        ):
            raise ValueError("nonnegative Pauli table and modulus required")
        if any(b <= a for a, b in zip(self.masses, self.masses[1:])):
            raise ValueError("mass nodes must increase strictly")

    def __call__(self, mass):
        import bisect

        mass = float(mass)
        if not math.isfinite(mass) or not self.masses[0] <= mass <= self.masses[-1]:
            raise ValueError("mass outside packet table; extrapolation is unsupported")
        index = min(max(bisect.bisect_right(self.masses, mass) - 1, 0), len(self.masses) - 2)
        a, b = self.masses[index : index + 2]
        theta = (mass - a) / (b - a)
        value = (1 - theta) * self.values[index] + theta * self.values[index + 1]
        error = self.label_error + 2 * self.lipschitz_constant * theta * (1 - theta) * (b - a)
        return MassTableResult(value, error)
