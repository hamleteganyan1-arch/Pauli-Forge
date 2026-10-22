import math
from fractions import Fraction

import numpy as np
import pytest
import torch

from pauliforge.decoders import PacketMassTable, PolynomialPacket, decode_low_mass_packets
from pauliforge.decoders.analytic_moments import PolynomialMomentOracle, pi_interval
from pauliforge.decoders.gram_sdp import (
    RegularityAssumptions,
    solve_quadrature_gram,
    toeplitz_gram_completion,
)
from pauliforge.decoders.router import DecoderRouter


def test_polynomial_packets_exact_mass_tw_and_dissociation():
    nodes, weights = np.polynomial.legendre.leggauss(3)
    coords = torch.tensor(
        np.array(np.meshgrid(nodes, nodes, nodes, indexing="ij")).reshape(3, -1).T,
        dtype=torch.float64,
    )
    w = torch.tensor(
        np.einsum("i,j,k->ijk", weights, weights, weights).ravel(), dtype=torch.float64
    )
    packet = PolynomialPacket((0, 0, 0), (1, 1, 1), Fraction(3, 2))
    rho, grad = packet.evaluate(coords)
    assert (rho * w).sum().item() == pytest.approx(1.5, abs=1e-14)
    assert (grad.square().sum(1) * w).sum().item() == pytest.approx(
        float(packet.tw_exact), abs=1e-13
    )
    translated = PolynomialPacket((6, 0, 0), (1, 1, 1), Fraction(3, 2))
    decoded = decode_low_mass_packets([packet, translated])
    assert decoded.tw_exact == 2 * packet.tw_exact
    assert decoded.pauli_exact == 0
    assert decoded.global_interval().certified_at_epsilon_n
    with pytest.raises(ValueError):
        decode_low_mass_packets([PolynomialPacket((0, 0, 0), (1, 1, 1), 3)])


def test_mass_table_interpolation_and_no_extrapolation():
    table = PacketMassTable([0, 1, 2], [0, 0, 0], lipschitz_constant=0)
    value = table(Fraction(3, 2))
    assert value.value == value.error_bound_conditional == 0
    assert not value.certified
    with pytest.raises(ValueError, match="outside"):
        table(2.1)


def test_missing_source_routes_are_explicit_not_heuristic_substitutes():
    router = DecoderRouter()
    assert router.decode("weighted_packet_source").status == "BLOCKED"
    assert router.decode("paff_mass_only").status == "BLOCKED"
    assert router.decode("qualitative_h1").status == "OPEN"


def test_rational_pi_and_monomial_moments():
    pi = pi_interval(100)
    assert pi.upper - pi.lower < Fraction(1, 2**100)
    assert float(pi.lower) == pytest.approx(math.pi)
    oracle = PolynomialMomentOracle({(0, 0, 0): 1, (2, 0, 0): 3})
    assert oracle.mass_exact == 2
    assert oracle.enclosure((0, 0, 0)).real.lower == 2
    assert oracle((0, 0, 0)) == 2 + 0j
    # constant density ∫_0^1 exp(i*pi*x)dx=2i/pi.
    constant = PolynomialMomentOracle({(0, 0, 0): 4})
    value = constant.enclosure((1, 0, 0))
    assert value.midpoint() == pytest.approx(8j / math.pi)
    assert value.absolute_error() < Fraction(1, 10**14)
    assert oracle.regularity().analytically_verified
    with pytest.raises(ValueError):
        PolynomialMomentOracle({(0, 0, 0): 1, (1, 0, 0): -1})


def test_exact_fourier_gram_completion_identity_and_psd():
    index, B = toeplitz_gram_completion({(0,): 0.2, (1,): 0.01, (-1,): 0.01}, 1, dimension=1)
    assert np.linalg.eigvalsh(B).min() >= -1e-14
    for x in np.linspace(0, 2, 13):
        phi = np.exp(1j * math.pi * x * np.array(index).ravel())
        value = np.einsum("i,ij,j->", phi, B, phi.conj())
        assert value.real == pytest.approx(0.2 + 0.02 * math.cos(math.pi * x), abs=1e-14)
    with pytest.raises(ValueError, match="diagonal"):
        toeplitz_gram_completion({(0,): 0.001, (1,): 1, (-1,): 1}, 1, dimension=1)


def test_regular_gram_preserves_full_factor_identity_and_cap():
    pytest.importorskip("cvxpy")
    oracle = PolynomialMomentOracle({(0, 0, 0): 4})
    frequencies = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
    result = oracle.solve(frequencies)
    assert result.fourier_identity_residual < 2e-6
    assert result.cap_minimum_eigenvalue >= -2e-6
    assert result.pauli_numerical >= -1e-6
    assert not result.certified and not result.continuum_certified
    assert "certified_moment_oracle" not in result.unmet_assumptions
    assert "analytic_fourier_truncation_budget" in result.unmet_assumptions
    x = np.linspace(0, 1, 25)
    phi = np.exp(1j * math.pi * x[:, None] * frequencies[:, 0])
    diagonal = np.einsum("gi,ij,gj->g", phi, result.coefficients, phi.conj())
    assert np.max(np.abs(diagonal - 1)) < 2e-6


def test_analytic_moments_do_not_silently_truncate_noninteger_frequencies():
    oracle = PolynomialMomentOracle({(0, 0, 0): 1})
    with pytest.raises(ValueError, match="integers"):
        oracle.solve([[0.7, 0, 0]])


def test_gram_rejects_zero_density_without_adding_floor():
    xyz = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    with pytest.raises(ValueError, match="positive rho"):
        solve_quadrature_gram(
            xyz,
            np.ones(2) / 2,
            np.array([0.0, 3.0]),
            np.ones(2),
            [[0, 0, 0]],
            RegularityAssumptions(3, 1),
        )
