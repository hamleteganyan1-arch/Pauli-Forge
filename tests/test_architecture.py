"""Metamorphic scalar and gradient tests; numerical evidence, not accuracy proofs."""

import pytest
import torch

from pauliforge.model import DensityXCModel, PauliKEDF

DTYPE = torch.float64


def cloud(n=3.0, points=43):
    generator = torch.Generator().manual_seed(131)
    coords = torch.randn(points, 3, dtype=DTYPE, generator=generator)
    coords = coords * torch.tensor([0.7, 1.3, 1.9], dtype=DTYPE)
    weights = torch.rand(points, dtype=DTYPE, generator=generator) + 0.5
    rho = torch.exp(-(coords.square() * torch.tensor([0.6, 0.3, 0.2])).sum(-1))
    rho = rho * (n / torch.sum(rho * weights))
    grad_u = -coords * torch.tensor([0.6, 0.3, 0.2]) * torch.sqrt(rho)[:, None]
    return coords, weights, rho, grad_u


def orthogonal():
    q, _ = torch.linalg.qr(
        torch.tensor([[0.7, 0.3, -0.4], [-0.1, 0.8, 0.6], [0.5, -0.2, 0.9]], dtype=DTYPE)
    )
    q[:, 0] = -q[:, 0]  # includes an improper rotation when determinant < 0
    return q


@pytest.mark.parametrize("encoder", ["moment", "sparse", "e3nn"])
def test_scalar_group_action_permutation_and_scaling(encoder):
    if encoder == "e3nn":
        pytest.importorskip("e3nn")
    torch.manual_seed(73)
    model = PauliKEDF(encoder=encoder, cutoff=1.3).double()
    x, w, rho, g = cloud()
    original = model(x, w, rho, g)
    rotation = orthogonal()
    moved = model(x @ rotation.T + x.new_tensor([7.0, -11.0, 3.0]), w, rho, g @ rotation.T)
    torch.testing.assert_close(moved.ts, original.ts, rtol=1e-11, atol=1e-11)
    permutation = torch.randperm(len(x))
    shuffled = model(x[permutation], w[permutation], rho[permutation], g[permutation])
    torch.testing.assert_close(shuffled.ts, original.ts, rtol=1e-11, atol=1e-11)
    for dilation in (0.37, 2.1):
        scaled = model(x / dilation, w / dilation**3, rho * dilation**3, g * dilation**2.5)
        torch.testing.assert_close(scaled.ts, original.ts * dilation**2, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("n", [0.25, 1.0, 1.5, 2.0])
def test_real_ensemble_low_sector_is_exact(n):
    model = PauliKEDF().double()
    out = model(*cloud(n))
    assert out.pauli.item() == out.learned_pauli.item() == out.analytical_pauli_floor.item() == 0
    assert out.ts.item() == out.tw.item()
    assert not out.matrix_representability_certified


def test_n_above_two_and_scalar_positivity_not_matrix_certificate():
    model = PauliKEDF().double()
    out = model(*cloud(2.0 + 1e-8))
    assert out.pauli.item() > 0
    assert not out.matrix_representability_certified


def test_default_encoder_has_no_dense_pair_matrix(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("O(G^2) distance matrix is forbidden in default encoder")

    monkeypatch.setattr(torch, "cdist", forbidden)
    out = PauliKEDF().double()(*cloud(points=4000))
    assert torch.isfinite(out.ts)


@pytest.mark.parametrize("which", ["rho", "weights", "grad"])
def test_nonfinite_density_inputs_rejected(which):
    values = list(cloud())
    index = {"rho": 2, "weights": 1, "grad": 3}[which]
    values[index] = values[index].clone()
    values[index].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        PauliKEDF().double()(*values)


def test_vector_derivative_equivariance():
    torch.manual_seed(99)
    model = PauliKEDF().double()
    x, w, rho, g = cloud()
    x = x.requires_grad_()
    g = g.requires_grad_()
    energy = model(x, w, rho, g).ts
    dx, dg = torch.autograd.grad(energy, (x, g))
    rotation = orthogonal()
    moved_x = (x.detach() @ rotation.T + 4).requires_grad_()
    moved_g = (g.detach() @ rotation.T).requires_grad_()
    moved_energy = model(moved_x, w, rho, moved_g).ts
    moved_dx, moved_dg = torch.autograd.grad(moved_energy, (moved_x, moved_g))
    torch.testing.assert_close(moved_dx, dx @ rotation.T, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(moved_dg, dg @ rotation.T, rtol=1e-10, atol=1e-11)


def test_configuration_and_state_roundtrip():
    model = PauliKEDF(hidden=11, encoder="sparse", cutoff=1.1).double()
    copy = PauliKEDF(**model.get_config()).double()
    copy.load_state_dict(model.state_dict())
    torch.testing.assert_close(copy(*cloud()).ts, model(*cloud()).ts)


def test_orbital_free_inference_does_not_import_quantum_teachers(monkeypatch):
    import builtins

    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"pyscf", "cvxpy"}:
            raise AssertionError("quantum teacher called during density-only inference")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    assert torch.isfinite(PauliKEDF().double()(*cloud()).ts)
    assert torch.isfinite(DensityXCModel().double()(*cloud()).exc)
