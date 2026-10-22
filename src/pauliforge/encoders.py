"""Invariant density encoders; no nuclei, orbitals or density matrices are read.

The default encoder costs O(G) and uses no pair matrix. Sparse variants store
only radius-cutoff edges, O(G + E) memory (E may be quadratic for dense clouds).
These descriptors are inductive biases, not an accuracy/completeness theorem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .physics import critical_integral


@dataclass
class EncodedDensity:
    features: torch.Tensor
    ell2: torch.Tensor
    n: torch.Tensor


class DensityEncoder(nn.Module):
    """Interface for dimensionless O(3)-invariant density descriptions."""

    feature_dim: int

    @staticmethod
    def geometry(coords, weights, rho):
        if (
            coords.ndim != 2
            or coords.shape[1] != 3
            or rho.ndim != 1
            or weights.shape != rho.shape
            or len(coords) != len(rho)
        ):
            raise ValueError("coords[G,3], weights[G], rho[G] required")
        if len(rho) < 2 or not all(torch.isfinite(v).all() for v in (coords, weights, rho)):
            raise ValueError("density coordinates, weights and values must be finite")
        if torch.any(rho < 0) or torch.any(weights <= 0):
            raise ValueError("density nonnegative and quadrature weights positive required")
        q = weights * rho
        n = q.sum()
        if n.detach().item() <= 0:
            raise ValueError("zero density mass")
        normalized_q = q / n
        center = (normalized_q[:, None] * coords).sum(0)
        centered = coords - center
        ell2 = (normalized_q * centered.square().sum(-1)).sum()
        if ell2.detach().item() <= 0 or not torch.isfinite(ell2):
            raise ValueError("density must have a nonzero finite second moment")
        return centered, normalized_q, ell2, n


class InvariantMomentBaseline(DensityEncoder):
    """Density-centered radial moments, O(G) time and storage.

    All input features are dimensionless. ell2 transforms as lambda**-2;
    electron mass is unchanged. Scalar radial moments are exactly O(3)
    invariant in real arithmetic, including reflections.
    """

    def __init__(self, scales=(0.5, 1.0, 2.0, 4.0)):
        super().__init__()
        self.scales = tuple(float(s) for s in scales)
        if not self.scales or any(not math.isfinite(s) or s <= 0 for s in self.scales):
            raise ValueError("positive finite radial scales required")
        self.feature_dim = len(self.scales) + 2

    def forward(self, coords, weights, rho):
        centered, q, ell2, n = self.geometry(coords, weights, rho)
        r2 = centered.square().sum(-1) / ell2
        radial = [(q * torch.exp(-r2 / (2 * s * s))).sum() for s in self.scales]
        i53 = critical_integral(rho, weights) * ell2 / n
        features = torch.stack([torch.log1p(n), torch.log1p(i53), *radial])
        return EncodedDensity(features, ell2, n)


def _radius_edges(coords, ell2, cutoff):
    """Find all cutoff edges, without a dense distance matrix.

    CPU tree construction is a discrete topology operation. Gradients use the
    selected edges and are valid away from topology changes; a smooth compact
    kernel vanishing with first derivative at the cutoff is used below.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("SparsePointKernel requires scipy") from exc
    normalized = (coords / torch.sqrt(ell2)).detach().cpu().numpy()
    pairs = cKDTree(normalized).query_pairs(float(cutoff), output_type="ndarray")
    if len(pairs) == 0:
        return torch.empty((2, 0), device=coords.device, dtype=torch.long)
    pairs = torch.as_tensor(pairs, device=coords.device, dtype=torch.long)
    return torch.cat((pairs.T, pairs[:, [1, 0]].T), dim=1)


class SparsePointKernel(InvariantMomentBaseline):
    """Finite-radius density graph, with exact neighbor set and no pair matrix.

    Topology is radius based rather than a nearest-k tie-breaking rule, so grid
    permutation, O(3), and coordinate scaling preserve the selected edge set.
    This finite-range channel is not a theorem about long-range response.
    """

    def __init__(self, scales=(0.5, 1.0, 2.0, 4.0), cutoff=0.75):
        super().__init__(scales)
        self.cutoff = float(cutoff)
        if not math.isfinite(self.cutoff) or self.cutoff <= 0:
            raise ValueError("positive finite graph cutoff required")
        self.feature_dim += len(self.scales)

    def forward(self, coords, weights, rho):
        base = super().forward(coords, weights, rho)
        edges = _radius_edges(coords, base.ell2, self.cutoff)
        q = rho * weights / base.n
        if edges.numel():
            a, b = edges
            r2 = (coords[a] - coords[b]).square().sum(-1) / base.ell2
            envelope = torch.clamp(1 - r2 / self.cutoff**2, min=0).square()
            pairs = q[a] * q[b]
            sparse = [(pairs * envelope * torch.exp(-r2 / (2 * s * s))).sum() for s in self.scales]
        else:
            sparse = [base.n * 0 for _ in self.scales]
        return EncodedDensity(torch.cat((base.features, torch.stack(sparse))), base.ell2, base.n)


class E3DensityGraph(InvariantMomentBaseline):
    """Optional e3nn public spherical-harmonic density-graph adapter (MIT).

    Angular irreps are pooled per point, then contracted to scalar norms.
    No third-party source is copied. This is a density graph, not an atomic
    force field. e3nn's parity (-1)**l makes reflections safe after norms.
    """

    def __init__(self, scales=(0.5, 1.0, 2.0, 4.0), cutoff=0.75):
        super().__init__(scales)
        try:
            from e3nn import o3
        except ImportError as exc:
            raise RuntimeError("E3DensityGraph requires the optional e3nn package") from exc
        self._spherical_harmonics = o3.spherical_harmonics
        self.cutoff = float(cutoff)
        if not math.isfinite(self.cutoff) or self.cutoff <= 0:
            raise ValueError("positive finite graph cutoff required")
        self.feature_dim += 3

    def forward(self, coords, weights, rho):
        base = super().forward(coords, weights, rho)
        edges = _radius_edges(coords, base.ell2, self.cutoff)
        q = rho * weights / base.n
        invariants = []
        if edges.numel():
            a, b = edges
            displacement = (coords[b] - coords[a]) / torch.sqrt(base.ell2)
            r2 = displacement.square().sum(-1)
            radial = torch.clamp(1 - r2 / self.cutoff**2, min=0).square() * torch.exp(-r2)
            for order in (0, 1, 2):
                sh = self._spherical_harmonics(
                    order, displacement, normalize=True, normalization="component"
                )
                message = q[b, None] * radial[:, None] * sh
                pooled = torch.zeros(
                    (len(coords), 2 * order + 1), dtype=coords.dtype, device=coords.device
                )
                pooled = pooled.index_add(0, a, message)
                invariants.append((q * pooled.square().sum(-1)).sum())
        else:
            invariants = [base.n * 0] * 3
        return EncodedDensity(
            torch.cat((base.features, torch.stack(invariants))), base.ell2, base.n
        )


def build_encoder(name="moment", scales=(0.5, 1.0, 2.0, 4.0), cutoff=0.75):
    if name == "moment":
        return InvariantMomentBaseline(scales)
    if name == "sparse":
        return SparsePointKernel(scales, cutoff)
    if name == "e3nn":
        return E3DensityGraph(scales, cutoff)
    raise ValueError(f"unknown density encoder: {name}")
