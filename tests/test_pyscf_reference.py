import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("pyscf")
from pauliforge.domain import FrozenDE
from pauliforge.teachers.pyscf_reference import density_grid_convergence, pyscf_reference

pytestmark = pytest.mark.qc


@pytest.mark.parametrize("method", ["RHF", "RKS", "UHF", "UKS"])
def test_actual_pyscf_spin_methods_and_kinetic_conversion(method):
    is_unrestricted = method.startswith("U")
    atoms = [("H", (0, 0, 0))] if is_unrestricted else [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    result = pyscf_reference(atoms, method=method, spin=1 if is_unrestricted else 0, grid_level=1)
    expected = 1 if is_unrestricted else 2
    assert np.trace(result["gamma"]) == pytest.approx(expected, abs=1e-8)
    assert result["ts_target"] == pytest.approx(
        2 * np.einsum("ij,ji->", result["dm_ao"], result["kinetic_ao_minus_delta"] / 2), abs=1e-10
    )
    assert np.min(np.linalg.eigvalsh(result["gamma"])) > -1e-8
    assert np.max(np.linalg.eigvalsh(result["gamma"])) <= 2 + 1e-8
    assert result["pauli_target"] == 0
    assert abs(result["grid_electrons"] - expected) < 1e-4
    recovered = np.einsum(
        "gi,ij,gj->g", result["ao_values_orth"], result["gamma"], result["ao_values_orth"]
    )
    assert np.max(abs(recovered - result["rho"])) < 1e-9
    assert result["metadata"]["quality"] == "NUMERICAL_REFERENCE"
    assert result["metadata"]["physical_ground_state_verified"] is False
    if is_unrestricted:
        assert np.linalg.eigvalsh(result["gamma_alpha"]).max() <= 1 + 1e-8
        assert np.linalg.eigvalsh(result["gamma_beta"]).max() <= 1 + 1e-8
    if method.endswith("KS"):
        assert np.isfinite(result["xc_target"])
        ordinary = (
            result["ts_target"] / 2
            + result["hartree_target"]
            + result["external_energy_hartree"]
            + result["xc_target"]
            + result["nuclear_repulsion_hartree"]
        )
        assert ordinary == pytest.approx(result["scf_total_energy_hartree"], abs=1e-7)
    else:
        assert "xc_target" not in result


def test_actual_fixed_density_grid_refinement():
    report = density_grid_convergence([("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))], levels=(0, 1, 2))
    assert report["same_scf_density"] and not report["certified"]
    assert report["rows"][-1]["electron_mass_error"] < 1e-5
    assert report["rows"][-1]["tw_change"] < report["rows"][1]["tw_change"]


def test_default_domain_refuses_helium_but_explicit_version_allows_it():
    with pytest.raises(ValueError, match="element"):
        pyscf_reference([("He", (0, 0, 0))], method="RHF")
    expanded = FrozenDE(allowed_atomic_numbers=(1, 2), version="hydrogen-helium-v1")
    result = pyscf_reference([("He", (0, 0, 0))], method="RHF", domain=expanded, grid_level=1)
    assert result["pauli_target"] == 0
    assert result["metadata"]["domain_version"] == "hydrogen-helium-v1"


def test_importing_reference_module_does_not_import_pyscf():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pauliforge.teachers.pyscf_reference; assert 'pyscf' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_negative_coarse_grid_weights_are_not_silently_dropped():
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    with pytest.raises(ValueError, match="negative quadrature weights"):
        pyscf_reference(atoms, method="RHF", grid_level=0)
    report = density_grid_convergence(atoms, levels=(0, 1))
    coarse = report["rows"][0]
    assert coarse["negative_quadrature_weights"] > 0
    assert coarse["electron_mass_error"] < 0.01


def test_actual_pyscf_reference_joint_matrix_teacher():
    pytest.importorskip("cvxpy")
    from pauliforge.teachers.matrix_sdp import solve_finite_basis

    reference = pyscf_reference(
        [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))], method="RHF", grid_level=1
    )
    active = np.flatnonzero(reference["rho"] > 0.1 * reference["rho"].max())
    selected = active[np.linspace(0, len(active) - 1, 12).astype(int)]
    result = solve_finite_basis(
        reference["rho"][selected],
        reference["ao_values_orth"][selected],
        reference["weights"][selected],
        reference["kinetic_orth_minus_delta"],
        reference["electrons"],
        basis_metric="analytic",
        tol=1e-6,
    )
    assert result["constraint_residual"] < 1e-6
    assert result["numerical_admissible"]
    assert not result["continuum_certified"]


def test_real_h2_normalized_representation_preserves_exact_pauli_sector(tmp_path):
    from pauliforge.data import load_density_npz, save_reference_npz
    from pauliforge.model import PauliKEDF

    reference = pyscf_reference(
        [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))], method="RHF", grid_level=1
    )
    path = save_reference_npz(tmp_path / "H2.npz", reference)
    sample = load_density_npz(path, normalize_electrons=True)
    output = PauliKEDF(hidden=8).double()(
        sample["coords"], sample["weights"], sample["rho"], sample["grad_u"]
    )
    assert float(output.n) == pytest.approx(2, abs=1e-15)
    assert float(output.pauli.detach()) == 0
    assert sample["metadata"]["raw_grid_electrons"] > 2
