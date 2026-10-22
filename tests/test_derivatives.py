"""Directional checks of the represented discrete density functional."""

import pytest
import torch

from pauliforge.density import DensityInput, cartesian_gradient_operator, functional_derivative
from pauliforge.model import PauliKEDF


def density(n=3.0):
    grid = torch.linspace(-1.8, 1.8, 4, dtype=torch.float64)
    xyz = torch.cartesian_prod(grid, grid, grid)
    weights = torch.full((len(xyz),), float((grid[1] - grid[0]) ** 3), dtype=xyz.dtype)
    values = torch.exp(-0.4 * xyz.square().sum(-1))
    values = values * (n / (weights * values).sum())
    operator = cartesian_gradient_operator(xyz)
    return DensityInput.from_density_values(xyz, weights, values, operator)


@pytest.mark.parametrize("n", [1.5, 3.0])
def test_full_density_derivative_matches_n_preserving_direction(n):
    torch.manual_seed(75)
    model = PauliKEDF().double()
    data = density(n)
    potential = functional_derivative(lambda item: model.forward_density(item).ts, data)
    eta = torch.sin(data.coords[:, 0]) + 0.2 * data.coords[:, 1]
    eta = eta - (data.weights * data.rho * eta).sum() / data.n
    direction = data.rho * eta
    assert abs((data.weights * direction).sum().item()) < 1e-13
    step = 1e-5
    plus, minus = (
        data.with_rho(data.rho + step * direction),
        data.with_rho(data.rho - step * direction),
    )
    fd = (model.forward_density(plus).ts - model.forward_density(minus).ts) / (2 * step)
    analytic = (data.weights * potential * direction).sum()
    torch.testing.assert_close(analytic, fd, rtol=2e-6, atol=2e-8)
    assert abs((data.weights * data.rho * potential).sum().item()) < 1e-11


def test_functional_derivative_has_kinetic_scaling():
    torch.manual_seed(49)
    model = PauliKEDF().double()
    data = density()
    derivative = functional_derivative(lambda item: model.forward_density(item).ts, data)
    dilation = 1.4
    scaled = functional_derivative(
        lambda item: model.forward_density(item).ts, data.scaled(dilation)
    )
    torch.testing.assert_close(scaled, derivative * dilation**2, rtol=1e-9, atol=1e-10)


def test_fixed_spatial_gradient_is_not_accepted_as_functional_derivative():
    model = PauliKEDF().double()
    data = density()
    fixed = DensityInput(data.coords, data.weights, data.rho, data.grad_u)
    with pytest.raises(ValueError, match="gradient_operator"):
        functional_derivative(lambda item: model.forward_density(item).ts, fixed)


def test_density_potential_can_train_parameters_with_second_autograd():
    torch.manual_seed(15)
    model = PauliKEDF(hidden=8).double()
    potential = functional_derivative(
        lambda item: model.forward_density(item).ts, density(), create_graph=True
    )
    loss = potential.square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )
