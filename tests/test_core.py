import math

import pytest
import torch

from pauliforge import FrozenDE, Geometry, PauliKEDF
from pauliforge.certificates import LocalInterval, assemble
from pauliforge.matrix import check_pauli_matrix, orthonormal_density_matrix, project_occupations

T = torch.float64


def make_cloud(n=3.0):
    grid = torch.linspace(-3, 3, 5, dtype=T)
    x = torch.cartesian_prod(grid, grid, grid)
    w = torch.full((len(x),), float((grid[1] - grid[0]) ** 3), dtype=T)
    a = 0.7
    base = torch.exp(-a * x.square().sum(-1))
    rho = base * (n / (base * w).sum())
    gu = -a * x * torch.sqrt(rho)[:, None]
    return x, w, rho, gu


def test_pauli_matrix_frobenius_projection():
    torch.manual_seed(4)
    a = torch.randn(6, 6, dtype=T)
    m = project_occupations((a + a.T) / 2, 7.0)
    assert check_pauli_matrix(m, 7.0, tol=1e-8)
    bad = torch.diag(torch.tensor([3.0, -1.0, 1.0], dtype=T))
    assert not check_pauli_matrix(bad, 3.0)
    s = torch.diag(torch.tensor([2.0, 3.0], dtype=T))
    d = torch.diag(torch.tensor([0.5, 1.0 / 3], dtype=T))
    assert check_pauli_matrix(orthonormal_density_matrix(d, s), 2.0)


def test_geometry_group_actions():
    dom = FrozenDE()
    x = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=T)
    geo = Geometry((1, 8), x)
    assert dom.validate(geo) == 9
    rot = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=T)
    assert dom.validate(Geometry((8, 1), x.flip(0) @ rot.T + torch.tensor([3.0, 2.0, -5.0]))) == 9
    with pytest.raises(ValueError):
        dom.validate(Geometry((1, 8), torch.zeros((2, 3))))
    with pytest.raises(ValueError):
        dom.validate(Geometry((1, 11), x))


def test_invariance_scaling_and_exact_low_n():
    torch.manual_seed(18)
    model = PauliKEDF().double().eval()
    xyz, w, rho, gu = make_cloud(3.0)
    orig = model(xyz, w, rho, gu)
    ang = 0.637
    rot = torch.tensor(
        [[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1]], dtype=T
    )
    moved = model(xyz @ rot.T + torch.tensor([7.0, -12.0, 3.0], dtype=T), w, rho, gu @ rot.T)
    assert torch.allclose(orig.ts, moved.ts, atol=1e-9, rtol=1e-10)
    ix = torch.randperm(len(xyz))
    perm = model(xyz[ix], w[ix], rho[ix], gu[ix])
    assert torch.allclose(orig.pauli, perm.pauli, atol=1e-9, rtol=1e-10)
    lam = 1.8
    scaled = model(xyz / lam, w / lam**3, rho * lam**3, gu * lam ** (5.0 / 2.0))
    assert torch.allclose(scaled.ts, orig.ts * lam**2, rtol=2e-10, atol=1e-8)
    x1, w1, r1, g1 = make_cloud(1.0)
    out = model(x1, w1, r1, g1)
    assert torch.abs(out.pauli).item() < 1e-10
    assert torch.allclose(out.ts, out.tw)
    assert orig.pauli.item() >= 0


def test_global_bounds():
    res = assemble(
        [LocalInterval(1, 1.1, 0.5), LocalInterval(0.2, 0.3, 0.7)], 2, 2.01, 0.08, 3, 0.1
    )
    assert res.lower <= res.upper
    assert res.budget_satisfied
    assert not res.certified_at_epsilon_n  # endpoint validity has not been established


def test_reflection_and_two_electron_threshold():
    torch.manual_seed(9)
    model = PauliKEDF().double()
    x, w, r, g = make_cloud(2.0)
    r0 = model(x, w, r, g)
    assert r0.pauli.item() == 0.0
    reflect = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=T))
    r1 = model(x @ reflect, w, r, g @ reflect)
    assert torch.allclose(r0.ts, r1.ts, atol=1e-9)


def test_spectral_projection_not_density_preserving():
    # The finite spatial density is diag(A Gamma A.T); A=I in this toy example.
    gamma = torch.tensor([[3.0, 0.2], [0.2, 0.0]], dtype=T)
    projected = project_occupations(gamma, 2.0)
    assert check_pauli_matrix(projected, 2.0)
    assert not torch.allclose(torch.diag(gamma), torch.diag(projected))
