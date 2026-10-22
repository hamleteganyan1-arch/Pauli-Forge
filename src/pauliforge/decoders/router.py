"""Explicit scope router: unsupported proofs become BLOCKED, not predictions."""

from dataclasses import dataclass

from .gram_sdp import solve_quadrature_gram
from .packets import MISSING_WEIGHTED_PACKET_SOURCE, decode_low_mass_packets


@dataclass(frozen=True)
class DecoderUnavailable:
    decoder: str
    reason: str
    source: str
    status: str = "BLOCKED"
    certified: bool = False


class DecoderRouter:
    def decode(self, route, **inputs):
        if route == "exact_low_mass_packets":
            return decode_low_mass_packets(inputs["packets"])
        if route == "regular_gram_sdp":
            return solve_quadrature_gram(**inputs)
        if route == "analytic_polynomial_gram":
            return inputs["oracle"].solve(inputs["frequencies"])
        if route == "weighted_packet_source":
            return DecoderUnavailable(
                route,
                "Exact source is not included; no broad weighted-packet theorem is implemented",
                MISSING_WEIGHTED_PACKET_SOURCE,
            )
        if route == "paff_mass_only":
            return DecoderUnavailable(
                route,
                "Exact PAFF compiler/audit source is absent; samples alone do not prove piecewise-affine root structure",
                "Ts_M1_Decoder_Audit_Exact_Mass_Gradient_9_Slabs_and_Sparse_Theorem_2026-10-20.md",
            )
        if route == "qualitative_h1":
            return DecoderUnavailable(
                route,
                "D1-C proves computability but a strong H1 name, exact-density enumerator and certified tail/Schur solver are not implemented",
                "Ts_D1C_Certified_Weighted_Labels_Full_H1_Proof_RU_2026-10-15.md",
                "OPEN",
            )
        raise ValueError(f"unknown decoder route: {route}")
