"""Density-only functionals with exact architectural scalar symmetries.

Kinetic energies use -Delta. XC output uses physical Hartree units. No model
imports quantum chemistry, density matrices or nuclear features at inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass

import torch
from torch import nn

from .encoders import build_encoder
from .heads import ExchangeCorrelationHead, KineticResidualHead, PauliFloorHead
from .physics import von_weizsaecker


@dataclass
class KEDFOutput:
    ts: torch.Tensor
    tw: torch.Tensor
    pauli: torch.Tensor
    analytical_pauli_floor: torch.Tensor
    learned_pauli: torch.Tensor
    n: torch.Tensor
    matrix_representability_certified: bool = False
    accuracy_status: str = "EMPIRICAL_untrained_or_learned_surrogate"


class PauliKEDF(nn.Module):
    mode = "kinetic_hybrid"

    def __init__(
        self,
        hidden=48,
        scales=(0.5, 1.0, 2.0, 4.0),
        c_lt=0.0,
        certified_lt_constant=False,
        encoder="moment",
        cutoff=0.75,
    ):
        super().__init__()
        self.hidden = int(hidden)
        if self.hidden <= 0:
            raise ValueError("positive hidden width required")
        self.scales = tuple(float(s) for s in scales)
        self.c_lt = float(c_lt)
        if isinstance(certified_lt_constant, dict):
            from .physics import LTConstantEvidence

            certified_lt_constant = LTConstantEvidence(**certified_lt_constant)
        self.certified_lt_constant = certified_lt_constant
        self.encoder_name = str(encoder)
        self.cutoff = float(cutoff)
        self.encoder = build_encoder(self.encoder_name, self.scales, self.cutoff)
        self.floor_head = PauliFloorHead(self.c_lt, self.certified_lt_constant)
        self.kinetic_head = KineticResidualHead(self.encoder.feature_dim, self.hidden)

    @property
    def net(self):
        """Compatibility accessor for the original prototype's trainable head."""
        return self.kinetic_head.net

    def get_config(self):
        return {
            "hidden": self.hidden,
            "scales": list(self.scales),
            "c_lt": self.c_lt,
            "certified_lt_constant": (
                asdict(self.certified_lt_constant)
                if is_dataclass(self.certified_lt_constant)
                else self.certified_lt_constant
            ),
            "encoder": self.encoder_name,
            "cutoff": self.cutoff,
        }

    def descriptors(self, coords, weights, rho):
        out = self.encoder(coords, weights, rho)
        return out.features, out.ell2, out.n

    def forward(self, coords, weights, rho, grad_sqrt_rho):
        features, ell2, n = self.descriptors(coords, weights, rho)
        if grad_sqrt_rho.shape != coords.shape or not torch.isfinite(grad_sqrt_rho).all():
            raise ValueError("finite grad_sqrt_rho[G,3] required")
        tw = von_weizsaecker(grad_sqrt_rho, weights)
        floor = self.floor_head(rho, weights, tw, n)
        residual = self.kinetic_head(features, ell2, n)
        pauli = floor + residual
        return KEDFOutput(tw + pauli, tw, pauli, floor, residual, n)

    def forward_density(self, density):
        density.validate()
        return self(density.coords, density.weights, density.rho, density.grad_u)

    def forward_fragments(self, fragments):
        """Additive only for analytically supported nonoverlapping fragments.

        Each fragment must come from CompactPolynomialFragment in modes.py.
        The actual support balls and represented sample values are checked;
        arbitrary sampled zeros are not accepted as support certificates.
        This proves additivity of this composed model, not learned accuracy.
        """
        from .modes import CompactPolynomialFragment

        fragments = tuple(fragments)
        if not fragments or not all(isinstance(f, CompactPolynomialFragment) for f in fragments):
            raise ValueError("analytic compact polynomial fragments required")
        for fragment in fragments:
            fragment.validate()
        for i, left in enumerate(fragments):
            for right in fragments[i + 1 :]:
                distance = torch.linalg.vector_norm(left.center - right.center).item()
                if distance <= left.radius + right.radius:
                    raise ValueError(
                        "fragment supports overlap or touch; exact composition unsupported"
                    )
        results = [self.forward_density(fragment.density) for fragment in fragments]

        def total(name):
            return torch.stack([getattr(value, name) for value in results]).sum()

        return KEDFOutput(
            *(
                total(name)
                for name in ("ts", "tw", "pauli", "analytical_pauli_floor", "learned_pauli", "n")
            ),
            accuracy_status="CONDITIONAL_disjoint_support_additivity_EMPIRICAL_accuracy",
        )


@dataclass
class XCOutput:
    exc: torch.Tensor
    exchange: torch.Tensor
    correlation: torch.Tensor
    n: torch.Tensor
    energy_unit: str = "Hartree"
    accuracy_status: str = "EMPIRICAL_XC_surrogate"


class DensityXCModel(nn.Module):
    mode = "analytical_pauli_xc_ml"

    def __init__(self, hidden=48, scales=(0.5, 1.0, 2.0, 4.0), encoder="moment", cutoff=0.75):
        super().__init__()
        self.hidden, self.scales = int(hidden), tuple(float(s) for s in scales)
        if self.hidden <= 0:
            raise ValueError("positive hidden width required")
        self.encoder_name, self.cutoff = str(encoder), float(cutoff)
        self.encoder = build_encoder(self.encoder_name, self.scales, self.cutoff)
        self.xc_head = ExchangeCorrelationHead(self.encoder.feature_dim, self.hidden)

    def get_config(self):
        return {
            "hidden": self.hidden,
            "scales": list(self.scales),
            "encoder": self.encoder_name,
            "cutoff": self.cutoff,
        }

    def forward(self, coords, weights, rho, grad_sqrt_rho=None):
        encoded = self.encoder(coords, weights, rho)
        head = self.xc_head(encoded.features, encoded.ell2, encoded.n)
        return XCOutput(head.exc, head.exchange, head.correlation, encoded.n)

    def forward_density(self, density):
        density.validate()
        return self(density.coords, density.weights, density.rho, density.grad_u)
