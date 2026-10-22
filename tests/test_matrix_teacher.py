"""Hard joint constraints, independent duals and exact finite certificates."""

from fractions import Fraction

import numpy as np
import pytest
import torch

from pauliforge.matrix import (
    ao_density_matrix,
    check_pauli_matrix,
    orthonormal_density_matrix,
    project_occupations,
    sampled_density,
)
from pauliforge.teachers.dual import (
    RationalFiniteCertificate,
    negative_spectrum_dual,
    verify_rational_primal_dual,
)
from pauliforge.teachers.matrix_sdp import solve_finite_basis, solve_spin_resolved

T = torch.float64


def test_joint_density_pauli_trace_and_independent_dual():
    result = solve_finite_basis([1.5, 1, 0.5], np.eye(3), np.ones(3), np.diag([1, 2, 4]), 3)
    assert result["numerical_admissible"]
    assert np.max(np.abs(result["density_reconstructed"] - [1.5, 1, 0.5])) < 1e-7
    assert check_pauli_matrix(torch.tensor(result["gamma"]), 3, tol=1e-7)
    assert result["ts_minus_delta"] == pytest.approx(5.5, abs=1e-7)
    assert result["dual_lower_numerical"] <= result["primal_upper_numerical"] + 1e-7
    assert abs(result["gap_numerical"]) < 1e-6
    assert not result["certified"] and not result["continuum_certified"]


def test_infeasible_density_not_fixed_by_projection():
    with pytest.raises(RuntimeError, match="infeasible"):
        solve_finite_basis([2.5, 0.5], np.eye(2), np.ones(2), np.eye(2), 3)
    raw = torch.diag(torch.tensor([2.5, 0.5], dtype=T))
    projected = project_occupations(raw, 3)
    assert check_pauli_matrix(projected, 3)
    assert not torch.allclose(
        sampled_density(projected, torch.eye(2, dtype=T)), torch.tensor([2.5, 0.5], dtype=T)
    )


def test_nonorthogonal_basis_and_analytic_quadrature_distinction():
    s = np.diag([2.0, 3.0])
    a = np.diag(np.sqrt([2.0, 3.0]))
    result = solve_finite_basis([1.5, 0.5], a, [0.9, 1.1], np.diag([2.0, 6.0]), 2, overlap=s)
    assert result["quadrature_mass"] == pytest.approx(1.9)
    assert result["quadrature_mass_error"] == pytest.approx(-0.1)
    assert result["ts_minus_delta"] == pytest.approx(2.5, abs=1e-7)
    gamma = orthonormal_density_matrix(torch.tensor(result["gamma_ao"]), torch.tensor(s))
    assert check_pauli_matrix(gamma, 2)
    assert torch.allclose(
        ao_density_matrix(gamma, torch.tensor(s)), torch.tensor(result["gamma_ao"]), atol=1e-10
    )
    with pytest.raises(ValueError, match="quadrature-orthonormal"):
        solve_finite_basis([1.5, 0.5], np.eye(2), [0.9, 1.1], np.diag([1.0, 2.0]), 2)
    already_orth = solve_finite_basis(
        [1.5, 0.5], np.eye(2), [0.9, 1.1], np.diag([1.0, 2.0]), 2, basis_metric="analytic"
    )
    assert already_orth["quadrature_mass_error"] == pytest.approx(-0.1)


def test_complex_basis_joint_constraints():
    phase = np.exp(2j * np.pi / 3)
    a = np.array([[phase ** (i * j) for j in range(3)] for i in range(3)]) / np.sqrt(3)
    target = np.real(np.sum((a @ np.diag([1.5, 1, 0.5])) * a.conj(), axis=1))
    k = np.array([[2, 0.2j, 0], [-0.2j, 3, 0], [0, 0, 4]], dtype=complex)
    result = solve_finite_basis(target, a, np.ones(3), k, 3)
    assert check_pauli_matrix(torch.tensor(result["gamma"]), 3, tol=1e-6)
    assert np.max(np.abs(result["density_reconstructed"] - target)) < 1e-7
    assert result["gap_numerical"] > -1e-6


def test_spin_resolved_caps_and_joint_density():
    result = solve_spin_resolved([1.5, 0.5], np.eye(2), np.ones(2), np.diag([1, 3]), (1, 1))
    assert check_pauli_matrix(torch.tensor(result["gamma_alpha"]), 1, max_occupation=1)
    assert check_pauli_matrix(torch.tensor(result["gamma_beta"]), 1, max_occupation=1)
    assert result["ts_minus_delta"] == pytest.approx(3, abs=1e-7)
    assert np.allclose(result["density_reconstructed"], [1.5, 0.5], atol=1e-7)
    assert abs(result["gap_numerical"]) < 1e-6


def test_arbitrary_potential_weak_dual_and_operator_uncertainty():
    k = np.diag([1.0, 3.0])
    rho = np.array([1.5, 0.5])
    for v in ([0, 0], [2, -3], [30, 40]):
        result = negative_spectrum_dual(k, np.eye(2), rho, 2, v, trace_multiplier=0.7)
        assert result["lower_numerical"] <= 3 + 1e-12
        assert not result["certified"]
        uncertain = negative_spectrum_dual(
            k, np.eye(2), rho, 2, v, trace_multiplier=0.7, operator_error_bound=0.02
        )
        assert uncertain["lower_numerical"] == pytest.approx(result["lower_numerical"] - 0.08)


def exact_example(**overrides):
    kwargs = dict(
        gamma=[["3/2", 0], [0, "1/2"]],
        kinetic=[[1, 0], [0, 3]],
        ao_values=[[1, 0], [0, 1]],
        target_density=["3/2", "1/2"],
        electrons=2,
        dual_density=[-1, -3],
    )
    kwargs.update(overrides)
    return verify_rational_primal_dual(**kwargs)


def test_exact_rational_finite_certificate_not_continuum():
    certificate = exact_example()
    assert certificate.lower == certificate.upper == Fraction(3)
    assert certificate.width == 0
    assert certificate.certified_finite_model
    assert not certificate.continuum_certified
    assert certificate.to_dict()["lower_exact"] == "3"
    with pytest.raises(ValueError, match="through verify"):
        RationalFiniteCertificate(Fraction(0), Fraction(1), Fraction(2), 2, 2)


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"gamma": [["5/2", 0], [0, "-1/2"]]}, "PSD or occupation"),
        ({"electrons": 1}, "trace"),
        ({"target_density": [1, 1]}, "density"),
        ({"dual_density": [-1, -2]}, "stationarity"),
        ({"lower_psd": [[0, 1], [1, 0]]}, "dual PSD"),
        ({"electrons": 2.0}, "not floats"),
    ],
)
def test_exact_certificate_rejects_missing_premises(overrides, reason):
    with pytest.raises(ValueError, match=reason):
        exact_example(**overrides)


@pytest.mark.parametrize(
    "bad", [torch.zeros(3), torch.zeros(0, 0), torch.full((2, 2), float("nan"))]
)
def test_projection_rejects_invalid_matrices(bad):
    with pytest.raises(ValueError):
        project_occupations(bad, 1)
    assert not check_pauli_matrix(bad, 1)


def test_matrix_parameter_validation():
    with pytest.raises(ValueError, match="occupation"):
        project_occupations(torch.eye(2), 1, max_occupation=-1)
    with pytest.raises(ValueError, match="electron"):
        project_occupations(torch.eye(2), float("nan"))
    with pytest.raises(ValueError, match="Hermitian"):
        orthonormal_density_matrix(torch.tensor([[1.0, 2.0], [0.0, 1.0]]), torch.eye(2))


def test_real_gamma_complex_basis_and_overlap_preserve_imaginary_components():
    a = torch.tensor([[1j, 0]], dtype=torch.complex128)
    assert sampled_density(torch.eye(2, dtype=T), a).item() == pytest.approx(1)
    overlap = torch.tensor([[2, 0.3j], [-0.3j, 2]], dtype=torch.complex128)
    dm = torch.eye(2, dtype=T) / 2
    gamma = orthonormal_density_matrix(dm, overlap)
    assert gamma.is_complex()
    assert check_pauli_matrix(gamma, 2)
    assert torch.allclose(ao_density_matrix(gamma, overlap), dm.to(torch.complex128), atol=1e-12)
