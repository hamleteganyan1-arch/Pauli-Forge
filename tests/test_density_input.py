"""Input, quadrature, representation and N-preserving derivative checks."""

from dataclasses import replace

import pytest
import torch

from pauliforge.density import (
    DensityInput,
    GaussianDensity,
    QuadratureCertificate,
    cartesian_gradient_operator,
    functional_derivative,
)

T = torch.float64


def cloud():
    axis = torch.linspace(-1.0, 1.0, 3, dtype=T)
    coords = torch.cartesian_prod(axis, axis, axis)
    weights = torch.full((len(coords),), 0.1, dtype=T)
    rho = 0.4 + torch.exp(-coords.square().sum(-1))
    rho = rho * (3 / (rho * weights).sum())
    return DensityInput.from_density_values(
        coords, weights, rho, cartesian_gradient_operator(coords), expected_electrons=3
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("rho", lambda d: d.rho * float("nan")),
        ("rho", lambda d: -d.rho),
        ("weights", lambda d: -d.weights),
        ("weights", lambda d: d.weights * float("inf")),
        ("coords", lambda d: d.coords[:, :2]),
        ("grad_u", lambda d: d.grad_u * float("nan")),
        ("units", lambda d: "Angstrom"),
        ("expected_electrons", lambda d: 4),
    ],
)
def test_invalid_density_inputs(field, value):
    density = cloud()
    with pytest.raises(ValueError):
        replace(density, **{field: value(density)})


def test_gradient_consistency_and_scaling():
    density = cloud()
    with pytest.raises(ValueError, match="inconsistent"):
        replace(density, grad_u=density.grad_u + 1)
    scaled = density.scaled(1.8)
    assert torch.allclose(scaled.n, density.n, atol=1e-14)
    assert torch.allclose(scaled.tw, 1.8**2 * density.tw, atol=1e-12)
    assert torch.allclose(scaled.energy_mass, 1.8**2 * density.energy_mass, atol=1e-12)
    assert not density.strong_h1_certified


def test_quadrature_metadata_cannot_turn_samples_into_proof():
    error = QuadratureCertificate(
        electron_error_bound=0.01,
        tw_error_bound=0.2,
        h1_error_bound=0.1,
        evidence_kind="conditional_bound",
        source="test assumption",
    )
    assert not error.continuum_certified
    assert error.scaled(2).tw_error_bound == 0.8
    assert error.scaled(2).electron_error_bound == 0.01
    with pytest.raises(ValueError):
        QuadratureCertificate(tw_error_bound=float("nan"))
    with pytest.raises(ValueError):
        QuadratureCertificate(evidence_kind="PROVED")


def test_total_tw_derivative_under_positive_n_preserving_perturbation():
    density = cloud()
    torch.manual_seed(81)
    direction = torch.randn_like(density.rho)
    direction -= (direction * density.weights).sum() / density.weights.sum()
    potential = functional_derivative(lambda d: d.tw, density)
    assert (density.weights * density.rho * potential).sum().abs() < 1e-12
    step = 1e-6
    plus = density.with_rho(density.rho + step * direction)
    minus = density.with_rho(density.rho - step * direction)
    numerical = (plus.tw - minus.tw) / (2 * step)
    predicted = (density.weights * potential * direction).sum()
    assert torch.allclose(predicted, numerical, rtol=2e-7, atol=1e-9)
    # Independent supplied grad_u is deliberately rejected as a potential model.
    independent = replace(density, gradient_operator=None)
    with pytest.raises(ValueError, match="gradient_operator"):
        functional_derivative(lambda d: d.tw, independent)


def test_derivative_coordinate_scaling_and_parameter_backpropagation():
    density = cloud()
    potential = functional_derivative(lambda d: d.tw, density)
    scaled = density.scaled(1.7)
    scaled_potential = functional_derivative(lambda d: d.tw, scaled)
    assert torch.allclose(scaled_potential, 1.7**2 * potential, rtol=1e-10, atol=1e-10)
    coefficient = torch.tensor(1.2, dtype=T, requires_grad=True)
    v = functional_derivative(lambda d: coefficient * d.tw, density, create_graph=True)
    loss = v.square().sum()
    loss.backward()
    assert coefficient.grad is not None and coefficient.grad.abs() > 0


def test_finite_difference_polynomial_exactness_and_grid_permutation():
    density = cloud()
    x = density.coords
    u = 1 + x.square().sum(-1)
    assert torch.allclose(
        torch.einsum("gai,i->ga", density.gradient_operator, u), 2 * x, atol=1e-14
    )
    permutation = torch.randperm(len(x))
    op = cartesian_gradient_operator(x[permutation])
    assert torch.allclose(
        torch.einsum("gai,i->ga", op, u[permutation]), 2 * x[permutation], atol=1e-14
    )
    with pytest.raises(ValueError, match="Cartesian"):
        cartesian_gradient_operator(x[:-1])


def test_analytic_density_only_representation_and_grid_refinement():
    representation = GaussianDensity(electrons=3.0, exponent=0.5)
    records = []
    for size in (9, 17, 33):
        axis = torch.linspace(-5.0, 5.0, size, dtype=T)
        coords = torch.cartesian_prod(axis, axis, axis)
        weights = torch.full((len(coords),), float((axis[1] - axis[0]) ** 3), dtype=T)
        density = DensityInput.from_representation(representation, coords, weights)
        assert torch.allclose(
            density.grad_u,
            -0.5 * density.coords * torch.sqrt(density.rho)[:, None],
            rtol=1e-12,
            atol=1e-12,
        )
        records.append((abs(density.n.item() - 3), abs(density.tw.item() - 2.25)))
    assert records[-1][0] < 1e-5
    assert records[-1][1] < 1e-4
    assert records[-1][1] < records[0][1]
