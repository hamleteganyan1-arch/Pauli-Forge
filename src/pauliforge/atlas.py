"""Finite-region adaptive atlas primitives from ADAPTIVE-ATLAS (2026-10-15).

Implements robust mass/energy refinement, strong 2:1 closure, the 17/16 cosine
windows, and full-diagonal blocks. A finite list of numerical grid points is
not a certified strong H1 name on R3. Consequently this class never certifies
that an unprovided spatial tail or outside background grid is covered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Callable

WINDOW_DILATION = Fraction(17, 16)
OVERLAP_BOUND = 512
# cos(pi/(2s))=sin(pi/34)>=1/17 and pi²<10. A rational upper
# bound for source (11), not an empirically chosen moderate constant.
IMS_CONSTANT_UPPER = math.ceil(Fraction(120 * OVERLAP_BOUND * 17**6) / WINDOW_DILATION**2)
ATLAS_SOURCE = "references/Ts_Adaptive_Multiscale_Physical_Atlas_Proof_RU_2026-10-15.md"


@dataclass(frozen=True, order=True)
class DyadicCube:
    level: int = 0
    index: tuple[int, int, int] = (0, 0, 0)
    root_side: Fraction = Fraction(1)

    def __post_init__(self):
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 0:
            raise ValueError("level must be a nonnegative integer")
        if len(self.index) != 3 or any(
            isinstance(i, bool) or not isinstance(i, int) for i in self.index
        ):
            raise ValueError("dyadic indices must be three integers")
        object.__setattr__(self, "index", tuple(self.index))
        object.__setattr__(self, "root_side", Fraction(self.root_side))
        if self.root_side <= 0:
            raise ValueError("root side must be positive")

    @property
    def side(self):
        return self.root_side / 2**self.level

    @property
    def center(self):
        return tuple(self.side * (Fraction(i) + Fraction(1, 2)) for i in self.index)

    def bounds(self, dilation=1):
        half = Fraction(dilation) * self.side / 2
        return tuple((c - half, c + half) for c in self.center)

    def children(self):
        return tuple(
            DyadicCube(
                self.level + 1, tuple(2 * i + j for i, j in zip(self.index, bits)), self.root_side
            )
            for bits in product((0, 1), repeat=3)
        )

    def touches(self, other):
        return all(
            a0 <= b1 and b0 <= a1 for (a0, a1), (b0, b1) in zip(self.bounds(), other.bounds())
        )

    def contains(self, other):
        return all(
            a0 <= b0 and b1 <= a1 for (a0, a1), (b0, b1) in zip(self.bounds(), other.bounds())
        )


@dataclass(frozen=True)
class IntegralEstimate:
    value: float
    absolute_error: float
    source: str = "numerical_quadrature"
    rigorous: bool = False

    def __post_init__(self):
        if (
            not math.isfinite(self.value)
            or not math.isfinite(self.absolute_error)
            or self.absolute_error < 0
        ):
            raise ValueError("finite integral estimate and nonnegative error required")
        if self.value + self.absolute_error < 0:
            raise ValueError("mass/kinetic integral cannot be strictly negative")

    @property
    def lower(self):
        return max(0.0, self.value - self.absolute_error)

    @property
    def upper(self):
        return max(0.0, self.value + self.absolute_error)


class AtlasResolutionError(RuntimeError):
    """Requested refinement exceeded an explicit computation resource budget."""


@dataclass(frozen=True)
class PauliBlock:
    cube: DyadicCube
    mass_interval: IntegralEstimate
    alpha: Fraction
    full_diagonal: bool = True
    status: str = "NUMERICAL_FINITE_REGION"
    source: str = ATLAS_SOURCE + " §§7,9,11"

    def __post_init__(self):
        if not self.full_diagonal:
            raise ValueError("a Pauli block requires full rho|Omega, never chi²rho")
        if Fraction(self.alpha) != (WINDOW_DILATION * self.cube.side) ** -2:
            raise ValueError("atlas weight must equal (17*h/16)^-2")

    @property
    def window_side(self):
        return WINDOW_DILATION * self.cube.side

    def restrict_density(self, coords, rho):
        """Return the full restricted density; window is a separate objective."""
        import torch

        bounds = self.cube.bounds(WINDOW_DILATION)
        low = coords.new_tensor([float(x[0]) for x in bounds])
        high = coords.new_tensor([float(x[1]) for x in bounds])
        mask = torch.logical_and(coords >= low, coords <= high).all(-1)
        return mask, rho[mask]


class AdaptiveAtlas:
    """Source-faithful geometry on a supplied finite collection of root cubes.

    ``oracle(cube, dilation, field)`` returns an IntegralEstimate for either
    rho (field='mass') or |grad sqrt(rho)|² (field='kinetic'). To claim a global
    atlas, callers still need the background-grid/tail construction of §3.1.
    """

    def __init__(
        self, roots, *, mass_threshold=8.0, energy_threshold=1.0, max_depth=10, max_leaves=100000
    ):
        self.roots = tuple(roots)
        if not self.roots or any(not isinstance(q, DyadicCube) or q.level != 0 for q in self.roots):
            raise ValueError("nonempty level-zero root cubes required")
        if len(set(self.roots)) != len(self.roots) or len({q.root_side for q in self.roots}) != 1:
            raise ValueError("roots must be distinct cubes on the same dyadic lattice")
        if (
            not math.isfinite(mass_threshold)
            or mass_threshold < 8
            or not math.isfinite(energy_threshold)
            or energy_threshold < 1
        ):
            raise ValueError("source thresholds require M>=8 and G>=1")
        if (
            not isinstance(max_depth, int)
            or max_depth < 0
            or not isinstance(max_leaves, int)
            or max_leaves < 1
        ):
            raise ValueError("invalid atlas resource budget")
        self.mass_threshold, self.energy_threshold = float(mass_threshold), float(energy_threshold)
        self.max_depth, self.max_leaves = max_depth, max_leaves
        self.leaves = tuple(self.roots)
        self.original_marks: dict[DyadicCube, str] = {}
        self.balance_witnesses: dict[DyadicCube, DyadicCube | None] = {}
        self.active_blocks: tuple[PauliBlock, ...] = ()
        self.complete_space = False
        self.continuum_certified = False
        self.source = ATLAS_SOURCE
        self.oracle = None

    def _checked_estimate(self, oracle, cube, dilation, field, max_error):
        estimate = oracle(cube, dilation, field)
        if not isinstance(estimate, IntegralEstimate):
            raise ValueError("integral oracle must return an explicit estimate/error pair")
        if estimate.absolute_error > max_error:
            raise ValueError("integral oracle error exceeds robust source comparison budget")
        return estimate

    def build(self, oracle: Callable):
        self.oracle = oracle
        self.original_marks.clear()
        pending, leaves = list(self.roots), []
        while pending:
            cube = pending.pop()
            mass = self._checked_estimate(oracle, cube, 32, "mass", self.mass_threshold / 16)
            if mass.value >= self.mass_threshold / 2:
                mark = "mass"
            else:
                kinetic = self._checked_estimate(
                    oracle, cube, 3, "kinetic", self.energy_threshold / (16 * float(cube.side) ** 2)
                )
                mark = (
                    "energy"
                    if float(cube.side) ** 2 * kinetic.value >= self.energy_threshold / 2
                    else None
                )
            if mark:
                if cube.level >= self.max_depth:
                    raise AtlasResolutionError("mass/energy threshold unresolved at maximum depth")
                self.original_marks[cube] = mark
                pending.extend(cube.children())
            else:
                leaves.append(cube)
            if len(pending) + len(leaves) > self.max_leaves:
                raise AtlasResolutionError(
                    "atlas exceeds max_leaves; no completeness claim returned"
                )
        self.leaves = tuple(sorted(leaves))
        self.balance()
        self.select_active(oracle)
        return self

    def balance(self):
        leaves = set(self.leaves)
        self.balance_witnesses.clear()
        while True:
            ordered = sorted(leaves)
            split = None
            for i, left in enumerate(ordered):
                for right in ordered[i + 1 :]:
                    if abs(left.level - right.level) > 1 and left.touches(right):
                        split = left if left.level < right.level else right
                        break
                if split is not None:
                    break
            if split is None:
                break
            if split.level >= self.max_depth or len(leaves) + 7 > self.max_leaves:
                raise AtlasResolutionError("strong 2:1 closure exceeds resource budget")
            candidates = [
                q
                for q in self.original_marks
                if q.side <= split.side
                and max(abs(a - b) for a, b in zip(q.center, split.center))
                <= Fraction(3, 2) * split.side
            ]
            self.balance_witnesses[split] = min(candidates) if candidates else None
            leaves.remove(split)
            leaves.update(split.children())
        self.leaves = tuple(sorted(leaves))
        return self

    def is_strongly_balanced(self):
        return all(
            abs(left.level - right.level) <= 1 or not left.touches(right)
            for i, left in enumerate(self.leaves)
            for right in self.leaves[i + 1 :]
        )

    def select_active(self, oracle=None):
        oracle = oracle or self.oracle
        if oracle is None:
            raise ValueError("mass oracle required for robust active superset")
        blocks = []
        for cube in self.leaves:
            mass = self._checked_estimate(oracle, cube, WINDOW_DILATION, "mass", 1 / 8)
            if mass.value >= 1.5:
                blocks.append(PauliBlock(cube, mass, (WINDOW_DILATION * cube.side) ** -2))
        self.active_blocks = tuple(blocks)
        return self.active_blocks

    def normalized_windows(self, coords):
        """chi, grad chi on supplied covered points, satisfying sum chi²=1.

        Uses every leaf, including inactive ones. Normalizing only active
        windows would change the localization formula and is not permitted.
        """
        import torch

        if coords.ndim != 2 or coords.shape[1] != 3 or not torch.isfinite(coords).all():
            raise ValueError("finite coords[G,3] required")
        zetas, gradients = [], []
        for cube in self.leaves:
            c = coords.new_tensor([float(v) for v in cube.center])
            side = float(WINDOW_DILATION * cube.side)
            scaled = (coords - c) / side
            inside = (scaled.abs() < 0.5).all(-1)
            cosine = torch.cos(math.pi * scaled)
            zeta = torch.where(inside, cosine.prod(-1), torch.zeros_like(scaled[:, 0]))
            gradient = []
            for axis in range(3):
                others = [j for j in range(3) if j != axis]
                derivative = (
                    -math.pi
                    / side
                    * torch.sin(math.pi * scaled[:, axis])
                    * cosine[:, others].prod(-1)
                )
                gradient.append(torch.where(inside, derivative, torch.zeros_like(derivative)))
            zetas.append(zeta)
            gradients.append(torch.stack(gradient, -1))
        zeta = torch.stack(zetas, 1)
        dzeta = torch.stack(gradients, 1)
        denominator = zeta.square().sum(1)
        if torch.any(denominator <= 0):
            raise ValueError("finite root collection does not cover supplied points")
        norm = denominator.sqrt()
        dot = (zeta[..., None] * dzeta).sum(1)
        chi = zeta / norm[:, None]
        dchi = (
            dzeta / norm[:, None, None]
            - zeta[..., None] * dot[:, None, :] / norm[:, None, None] ** 3
        )
        return chi, dchi

    def numerical_ims(self, coords, weights, rho):
        """Quadrature diagnostic, not a rigorous IMS bound without input error."""
        _, gradients = self.normalized_windows(coords)
        return (weights * rho * gradients.square().sum((1, 2))).sum()

    def to_dict(self):
        return {
            "leaf_count": len(self.leaves),
            "active_count": len(self.active_blocks),
            "strongly_balanced": self.is_strongly_balanced(),
            "complete_space": False,
            "continuum_certified": False,
            "source": self.source,
            "status": "NUMERICAL_FINITE_REGION",
            "missing": [
                "certified_global_H1_name_and_tail",
                "independent_local_intervals",
                "quadrature_error_bounds",
            ],
            "blocks": [
                {
                    "level": block.cube.level,
                    "index": list(block.cube.index),
                    "center": list(map(float, block.cube.center)),
                    "side": float(block.cube.side),
                    "alpha": float(block.alpha),
                    "mass_lower": block.mass_interval.lower,
                    "mass_upper": block.mass_interval.upper,
                    "full_diagonal": True,
                }
                for block in self.active_blocks
            ],
        }


def sampled_integral_oracle(coords, weights, rho, grad_u):
    """Discrete-grid oracle for diagnostics, explicitly NOT a continuum bound.

    Its zero error means no external error budget was supplied for the finite
    weighted sum. ``rigorous=False`` records that quadrature/tails are unknown.
    It must never be used to infer global continuum certificates.
    """
    import torch

    if coords.shape != (len(rho), 3) or weights.shape != rho.shape or grad_u.shape != coords.shape:
        raise ValueError("unaligned density quadrature")
    if not all(torch.isfinite(v).all() for v in (coords, weights, rho, grad_u)):
        raise ValueError("finite density quadrature required")
    if torch.any(weights <= 0) or torch.any(rho < 0):
        raise ValueError("positive weights and nonnegative density required")

    def oracle(cube, dilation, field):
        bounds = cube.bounds(dilation)
        lower = coords.new_tensor([float(v[0]) for v in bounds])
        upper = coords.new_tensor([float(v[1]) for v in bounds])
        mask = ((coords >= lower) & (coords < upper)).all(-1)
        if field == "mass":
            values = rho
        elif field == "kinetic":
            values = grad_u.square().sum(-1)
        else:
            raise ValueError("unknown integral field")
        return IntegralEstimate(
            float((values[mask] * weights[mask]).sum().detach()),
            0.0,
            "finite_weighted_grid_sum_no_quadrature_certificate",
            False,
        )

    return oracle
