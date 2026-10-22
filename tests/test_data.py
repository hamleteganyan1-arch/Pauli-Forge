import numpy as np
import pytest

from pauliforge.data import (
    canonical_geometry_key,
    chemical_family,
    load_density_npz,
    save_reference_npz,
    split_by_family,
)
from pauliforge.domain import FrozenDE


def reference(z=(1, 1), positions=None):
    positions = np.array([[0, 0, 0], [0, 0, 1.4]]) if positions is None else np.asarray(positions)
    n = sum(z)
    return {
        "coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "weights": np.ones(3),
        "rho": np.ones(3) * n / 3,
        "grad_u": np.zeros((3, 3)),
        "pauli_target": np.asarray(0.0),
        "atomic_numbers": np.asarray(z),
        "positions_bohr": positions,
        "charge": 0,
        "spin": n % 2,
        "metadata": {"quality": "SYNTHETIC_TEST", "functional_derivative_reliable": False},
    }


def test_archive_roundtrip_and_domain_binding(tmp_path):
    path = save_reference_npz(tmp_path / "h2.npz", reference())
    result = load_density_npz(path)
    assert result["expected_electrons"] == 2
    assert result["metadata"]["domain_hash"] == FrozenDE().domain_hash
    assert result["metadata"]["physical_ground_state_verified"] is False
    with pytest.raises(FileExistsError):
        save_reference_npz(path, reference())
    with pytest.raises(ValueError, match="different FrozenDE"):
        load_density_npz(path, domain=FrozenDE(version="new-hcno-version"))


def test_archive_rejects_bad_mass_and_unreliable_potential(tmp_path):
    value = reference()
    value["rho"] *= 1.2
    with pytest.raises(ValueError, match="integral"):
        save_reference_npz(tmp_path / "bad.npz", value)
    value = reference()
    value.update(gradient_operator=np.zeros((3, 3, 3)), functional_derivative_target=np.ones(3))
    with pytest.raises(ValueError, match="reliable provenance"):
        save_reference_npz(tmp_path / "potential.npz", value)


def test_archive_rejects_float_charge_and_elements(tmp_path):
    value = reference()
    value["charge"] = 0.5
    with pytest.raises(ValueError, match="integer"):
        save_reference_npz(tmp_path / "float_charge.npz", value)
    value = reference()
    value["atomic_numbers"] = np.array([1.2, 1.2])
    # Loading raw archives must never silently truncate nuclear charges.
    np.savez(tmp_path / "float_z.npz", **{k: v for k, v in value.items() if k != "metadata"})
    with pytest.raises(ValueError, match="integer vector"):
        load_density_npz(tmp_path / "float_z.npz")


def test_geometry_fingerprint_rigid_reflection_and_permutation():
    z = np.array([8, 1, 1])
    positions = np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 1.0], [-1.3, 0.0, 1.0]])
    perm = [2, 0, 1]
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    changed = positions @ rotation.T + np.array([3.0, -2.0, 1.0])
    assert canonical_geometry_key(z, positions) == canonical_geometry_key(z[perm], changed[perm])
    assert chemical_family(z) == chemical_family(z[perm])


def test_family_split_keeps_all_geometries_and_reorderings_together(tmp_path):
    paths = []
    for name, data in [
        ("h2_a", reference()),
        ("h2_b", reference(positions=[[0, 0, 0], [0, 0, 2.0]])),
        ("h", reference(z=(1,), positions=[[0, 0, 0]])),
    ]:
        paths.append(save_reference_npz(tmp_path / f"{name}.npz", data))
    train, test = split_by_family(paths, test_fraction=0.5)
    train_families = {load_density_npz(p)["family_id"] for p in train}
    test_families = {load_density_npz(p)["family_id"] for p in test}
    assert train and test and train_families.isdisjoint(test_families)
    assert set(paths) == set(train + test)
    assert split_by_family(paths, test_fraction=0.5) == (train, test)


def test_explicit_mass_normalization_records_changed_density_and_preserves_archive(tmp_path):
    value = reference()
    value["rho"] *= 1 + 1e-5
    value["grad_u"] = np.ones((3, 3))
    path = save_reference_npz(tmp_path / "h2-offmass.npz", value)
    before = path.read_bytes()
    raw = load_density_npz(path)
    normalized = load_density_npz(path, normalize_electrons=True)
    factor = normalized["metadata"]["electron_normalization_factor"]
    assert float((normalized["weights"] * normalized["rho"]).sum()) == pytest.approx(2, abs=1e-15)
    assert np.isclose(factor, 1 / (1 + 1e-5))
    assert np.allclose(normalized["grad_u"].numpy(), raw["grad_u"].numpy() * factor**0.5)
    assert normalized["metadata"]["raw_grid_electrons"] > 2
    assert normalized["metadata"]["labels_recomputed_at_normalized_density"] is False
    assert path.read_bytes() == before


def test_offmass_derivative_archive_cannot_be_silently_normalized(tmp_path):
    value = reference()
    value["rho"] *= 1 + 1e-5
    value.update(gradient_operator=np.zeros((3, 3, 3)), functional_derivative_target=np.zeros(3))
    value["metadata"]["functional_derivative_reliable"] = True
    path = save_reference_npz(tmp_path / "offmass-potential.npz", value)
    with pytest.raises(ValueError, match="different density"):
        load_density_npz(path, normalize_electrons=True)


@pytest.mark.parametrize("field", ["rho", "grad_u", "coords", "positions_bohr", "pauli_target"])
def test_complex_density_inputs_cannot_silently_discard_imaginary_parts(tmp_path, field):
    value = reference()
    value[field] = np.asarray(value[field], dtype=complex) + 1j
    path = tmp_path / "complex.npz"
    np.savez(path, **{key: item for key, item in value.items() if key != "metadata"})
    with pytest.raises(ValueError, match="must be real"):
        load_density_npz(path)
