import math
from fractions import Fraction

import numpy as np
import pytest
import torch

from pauliforge.decoders.smoothing import (
    fractional_smoothing_error,
    gaussian_atom_samples,
    single_atom_weizsaecker_exact,
    source_transport_stability,
)


def test_exact_single_atom_energy_and_units():
    assert single_atom_weizsaecker_exact(2, "1/4") == Fraction(3)
    assert single_atom_weizsaecker_exact(4, "1/2") == Fraction(3)
    # This is W_t for mass4, not an exact N>2 kinetic/Pauli claim.
    assert single_atom_weizsaecker_exact(0, 1) == 0
    with pytest.raises(ValueError, match="rational"):
        single_atom_weizsaecker_exact(2.0, 1)


def test_gaussian_samples_match_analytic_single_atom_and_scaling():
    dtype = torch.float64
    coords = torch.tensor([[0, 0, 0], [0.5, 1, -0.2], [2, -1, 3]], dtype=dtype)
    centers = torch.tensor([[0.2, -0.4, 0.1]], dtype=dtype)
    masses, time = torch.tensor([3.0], dtype=dtype), 0.7
    rho, grad = gaussian_atom_samples(coords, centers, masses, time)
    displacement = coords - centers[0]
    expected = (
        3 * torch.exp(-displacement.square().sum(1) / (4 * time)) / (4 * math.pi * time) ** 1.5
    )
    assert torch.allclose(rho, expected, atol=1e-14, rtol=1e-14)
    assert torch.allclose(grad, -displacement * rho.sqrt()[:, None] / (4 * time))
    factor = 1.9
    scaled_rho, scaled_grad = gaussian_atom_samples(
        coords / factor, centers / factor, masses, time / factor**2
    )
    assert torch.allclose(scaled_rho, factor**3 * rho, rtol=1e-13)
    assert torch.allclose(scaled_grad, factor**2.5 * grad, rtol=1e-13)


def test_fractional_bound_is_conditional_and_refuses_missing_regularity():
    result = fractional_smoothing_error(0.25, 0.5, 2)
    assert result.upper == 13
    assert result.status == "CONDITIONAL" and not result.certified
    assert result.scope == "weizsaecker_heat_smoothed_measure"
    with pytest.raises(ValueError):
        fractional_smoothing_error(1, 1, 2)
    with pytest.raises(ValueError):
        fractional_smoothing_error(1, 0.5, float("nan"))


def _separable_mixture_energy(masses, centers, time):
    # Actual 3D mixture has shared y,z Gaussian factors; integrate the x score
    # under its component Gaussians and add the exact y,z contribution.
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    energy_x = 0.0
    masses, centers = np.asarray(masses), np.asarray(centers)
    for mass, center in zip(masses, centers):
        points = center + 2 * math.sqrt(time) * nodes
        displacements = points[:, None] - centers[None, :]
        logits = np.log(masses)[None, :] - displacements**2 / (4 * time)
        normalized = np.exp(logits - logits.max(1)[:, None])
        posterior = normalized / normalized.sum(1)[:, None]
        score = -(posterior * displacements).sum(1) / (2 * time)
        energy_x += mass * np.dot(weights, score**2) / (4 * math.sqrt(math.pi))
    return energy_x + masses.sum() / (4 * time)


def test_transport_drop_budget_bounds_real_mixture_change():
    time, dropped = 0.8, 0.005
    original = _separable_mixture_energy([1, dropped], [0, 3], time)
    retained = float(single_atom_weizsaecker_exact(1, "4/5"))
    result = source_transport_stability(1 + dropped, time, dropped_mass=dropped)
    assert abs(original - retained) <= result.upper
    assert not result.roundoff_certified and not result.certified
    assert source_transport_stability(2, time).upper == 0
    assert source_transport_stability(2, time, transport_cost=100).upper == pytest.approx(
        3 * 2 / (8 * time)
    )
    with pytest.raises(ValueError):
        source_transport_stability(1, time, dropped_mass=2)


def test_single_atom_gaussian_quadrature_matches_exact_energy():
    # Tensor-product Gauss-Hermite integrates the Gaussian quadratic exactly.
    nodes, weights = np.polynomial.hermite.hermgauss(4)
    points = torch.tensor(
        np.array(np.meshgrid(nodes, nodes, nodes, indexing="ij")).reshape(3, -1).T,
        dtype=torch.float64,
    )
    time, mass = 0.5, 2
    coords = 2 * math.sqrt(time) * points
    rho, grad_u = gaussian_atom_samples(coords, torch.zeros((1, 3)), [mass], time)
    w3 = np.einsum("i,j,k->ijk", weights, weights, weights).reshape(-1)
    integrand = grad_u.square().sum(1) / rho
    value = mass * np.dot(w3, integrand.numpy()) / math.pi**1.5
    assert value == pytest.approx(float(single_atom_weizsaecker_exact(mass, "1/2")), abs=1e-13)
