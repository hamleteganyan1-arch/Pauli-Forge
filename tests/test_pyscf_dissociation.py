import pytest

pytest.importorskip("pyscf")

from pauliforge.teachers.pyscf_reference import pyscf_reference

pytestmark = pytest.mark.qc


def test_stretched_rks_h2_has_checked_numerical_scf_and_recorded_strategy():
    reference = pyscf_reference(
        [("H", (0, 0, -5)), ("H", (0, 0, 5))], method="RKS", spin=0, grid_level=1
    )
    assert reference["metadata"]["scf_converged"]
    assert reference["metadata"]["scf_solver_strategy"] in {
        "diis",
        "newton_fallback_after_unconverged_diis",
    }
    assert reference["pauli_target"] == 0.0
    assert not reference["metadata"]["physical_ground_state_verified"]
