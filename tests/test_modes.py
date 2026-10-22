import pytest
import torch

from pauliforge.density import DensityInput
from pauliforge.model import DensityXCModel, PauliKEDF
from pauliforge.modes import (
    AnalyticalPauliXCFunctional,
    CompactPolynomialFragment,
    build_functional,
)


def compact_fragment(center, n=3.0):
    grid = torch.linspace(-1.3, 1.3, 7, dtype=torch.float64)
    xyz = torch.cartesian_prod(grid, grid, grid)
    weights = torch.full((len(xyz),), float((grid[1] - grid[0]) ** 3), dtype=xyz.dtype)
    center = torch.tensor(center, dtype=xyz.dtype)
    return CompactPolynomialFragment.from_grid(xyz + center, weights, center, 1.2, n)


def test_disjoint_compact_support_additivity_and_refusal_of_overlap():
    model = PauliKEDF().double()
    a = compact_fragment([-3, 0, 0])
    b = compact_fragment([3, 0, 0], 3.5)
    combined = model.forward_fragments([a, b])
    individual_a, individual_b = model.forward_density(a.density), model.forward_density(b.density)
    assert combined.pauli.item() == (individual_a.pauli + individual_b.pauli).item()
    assert combined.ts.item() == (individual_a.ts + individual_b.ts).item()
    assert "disjoint_support" in combined.accuracy_status
    assert not combined.matrix_representability_certified
    overlap = compact_fragment([-2, 0, 0])
    with pytest.raises(ValueError, match="overlap"):
        model.forward_fragments([a, overlap])
    with pytest.raises(ValueError, match="analytic compact"):
        model.forward_fragments([a.density, b.density])


def test_exact_two_electron_fragment_dissociation():
    model = PauliKEDF().double()
    for separation in (3.0, 5.0, 10.0):
        fragments = [
            compact_fragment([-separation, 0, 0], 2),
            compact_fragment([separation, 0, 0], 2),
        ]
        out = model.forward_fragments(fragments)
        assert out.pauli.item() == 0
        torch.testing.assert_close(out.ts, out.tw, rtol=0, atol=0)


def test_fragment_certificate_rejects_altered_density():
    model = PauliKEDF().double()
    fragment = compact_fragment([0, 0, 0])
    fragment.density.rho[0] += 0.1
    with pytest.raises(ValueError, match="differ|gradient"):
        model.forward_fragments([fragment])


def test_xc_head_has_exchange_not_kinetic_scaling():
    torch.manual_seed(12)
    model = DensityXCModel().double()
    density = compact_fragment([0, 0, 0]).density
    out = model.forward_density(density)
    dilation = 1.7
    scaled = model.forward_density(density.scaled(dilation))
    torch.testing.assert_close(scaled.exchange, out.exchange * dilation, rtol=1e-11, atol=1e-11)
    assert out.exchange < 0
    assert not torch.isclose(scaled.exc, out.exc * dilation**2)


def test_two_modes_remain_separate_and_use_consistent_units():
    density = DensityInput(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        torch.full((2,), 1.5, dtype=torch.float64),
        torch.zeros(2, 3, dtype=torch.float64),
    )
    xc = DensityXCModel().double()
    mode_b = AnalyticalPauliXCFunctional(xc)
    teacher = {
        "numerical_admissible": True,
        "density_target": density.rho.detach().numpy(),
        "electrons": 3.0,
        "ts_minus_delta": 1.2,
        "gamma": torch.eye(2, dtype=torch.float64) * 1.5,
        "ao_values": torch.eye(2, dtype=torch.float64),
        "kinetic": torch.eye(2, dtype=torch.float64) * 0.4,
        "weights": density.weights,
    }
    out = mode_b(density, teacher, 3.0, -7.0, 1.0)
    torch.testing.assert_close(
        out.total_hartree, out.ts_finite_basis / 2 + 3 - 7 + 1 + out.exc_hartree
    )
    torch.testing.assert_close(out.total_project_units, 2 * out.total_hartree)
    assert out.pauli_finite_basis.item() == pytest.approx(1.2)
    assert not out.continuum_certified and "not_continuum" in out.status
    assert not hasattr(xc, "kinetic_head")
    assert isinstance(build_functional("kinetic_hybrid"), PauliKEDF)
    assert isinstance(build_functional("analytical_pauli_xc_ml"), AnalyticalPauliXCFunctional)
    with pytest.raises(ValueError, match="unknown scientific mode"):
        build_functional("mixed")
    teacher["density_target"] = teacher["density_target"] * 1.1
    with pytest.raises(ValueError, match="does not match"):
        mode_b(density, teacher, 3.0, -7.0)


def test_negative_radius_direct_constructor_cannot_fake_disjoint_support():
    from dataclasses import replace

    model = PauliKEDF().double()
    fragment = compact_fragment([0, 0, 0])
    invalid = replace(fragment, radius=-fragment.radius)
    with pytest.raises(ValueError, match="compact support parameters"):
        model.forward_fragments([invalid, fragment])


def test_mode_b_does_not_trust_teacher_flag_or_offer_partial_total_gradient():
    from dataclasses import replace

    density = DensityInput(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        torch.full((2,), 1.5, dtype=torch.float64),
        torch.zeros(2, 3, dtype=torch.float64),
    )
    mode = AnalyticalPauliXCFunctional(DensityXCModel().double())
    teacher = {
        "numerical_admissible": True,
        "density_target": density.rho.detach().numpy(),
        "electrons": 3.0,
        "ts_minus_delta": 1.2,
    }
    with pytest.raises(ValueError, match="gamma/A/K"):
        mode(density, teacher, 1.0, -2.0)
    teacher.update(
        gamma=torch.eye(2, dtype=torch.float64) * 1.5,
        ao_values=torch.eye(2, dtype=torch.float64),
        kinetic=torch.eye(2, dtype=torch.float64) * 0.4,
        weights=density.weights,
    )
    differentiable = replace(density, rho=density.rho.clone().requires_grad_())
    total = mode(differentiable, teacher, 1.0, -2.0)
    assert not total.kinetic_derivative_available
    with pytest.raises(RuntimeError, match="total density derivative unavailable"):
        torch.autograd.grad(total.total_hartree, differentiable.rho)
    teacher["gamma"] = torch.diag(torch.tensor([2.5, 0.5], dtype=torch.float64))
    with pytest.raises(ValueError, match="PSD/cap"):
        mode(density, teacher, 1.0, -2.0)
