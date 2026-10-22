"""Safe density archives and conservative molecular-family partitions.

No PySCF import or orbital computation occurs in this module. A geometry gate
is always applied; accepting an archive does not prove ground-state density
membership. The family splitter intentionally groups *all* geometries with
one composition/charge/spin to prevent geometry and isomer leakage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .domain import FrozenDE, Geometry

SCHEMA_VERSION = "1.0"


def chemical_family(atomic_numbers, charge=0, spin=0) -> str:
    """Conservative composition/charge/spin family, independent of geometry."""
    composition = sorted(Counter(int(z) for z in atomic_numbers).items())
    return json.dumps(
        {"composition": composition, "charge": int(charge), "spin": int(spin)}, sort_keys=True
    )


def canonical_geometry_key(atomic_numbers, positions_bohr, charge=0, spin=0, *, decimals=6):
    """O(3), translation and atom-order invariant duplicate fingerprint.

    The colored distance multiset may conflate homometric structures. Such
    overgrouping is conservative for leakage prevention; it is not a complete
    reconstruction or molecular-identity theorem.
    """
    z = np.asarray(atomic_numbers)
    x = np.asarray(positions_bohr, dtype=float)
    if z.ndim != 1 or x.shape != (len(z), 3) or not np.isfinite(x).all():
        raise ValueError("invalid geometry for duplicate fingerprint")
    distances = []
    for i in range(len(z)):
        for j in range(i):
            colors = sorted((int(z[i]), int(z[j])))
            distance = round(float(np.linalg.norm(x[i] - x[j])), decimals)
            distances.append((*colors, distance))
    payload = {
        "family": chemical_family(z, charge, spin),
        "colored_pair_distances": sorted(distances),
        "rounding_decimals": decimals,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _metadata(archive) -> dict:
    if "metadata_json" not in archive:
        return {
            "schema_version": "legacy",
            "quality": "UNVERIFIED_LEGACY_ARCHIVE",
            "physical_ground_state_verified": False,
            "functional_derivative_reliable": False,
        }
    try:
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("metadata_json must be a scalar JSON object") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must encode an object")
    if metadata.get("schema_version") not in {"1.0", "legacy"}:
        raise ValueError("unsupported density archive schema version")
    if metadata.get("kinetic_convention", "-Delta") != "-Delta":
        raise ValueError("archive must convert kinetic energies to -Delta convention")
    if metadata.get("spatial_units", "Bohr") != "Bohr":
        raise ValueError("density archive spatial units must be Bohr")
    return metadata


def _strict_scalar_int(value, name):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a scalar integer")
    return int(array.item())


def _real_array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real; imaginary components cannot be discarded")
    return np.asarray(value, dtype=float)


def load_density_npz(
    path,
    *,
    domain=None,
    device="cpu",
    dtype=torch.float64,
    target="pauli_target",
    mass_rtol=2e-3,
    normalize_electrons=False,
):
    """Load finite density samples, enforcing geometry and electron consistency.

    ``domain=None`` selects FrozenDE(); it does not disable gating. Optional
    derivative targets are loaded only with explicit reliable provenance and a
    density-gradient operator, making the density dependence testable.
    ``normalize_electrons=True`` explicitly changes the sampled density by a
    uniform numerical rescaling; raw archives and offline AO matrices remain
    unchanged. Its factor and raw mass error are recorded, not certified.
    """
    if not math.isfinite(mass_rtol) or mass_rtol <= 0:
        raise ValueError("mass_rtol must be finite and positive")
    selected_domain = FrozenDE() if domain is None else domain
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {
            "coords",
            "weights",
            "rho",
            "grad_u",
            "atomic_numbers",
            "positions_bohr",
            "charge",
        }
        if target is not None:
            required.add(target)
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"density archive missing fields: {sorted(missing)}")
        z = np.asarray(archive["atomic_numbers"])
        if z.ndim != 1 or z.dtype.kind not in "iu":
            raise ValueError("atomic_numbers must be an integer vector")
        charge = _strict_scalar_int(archive["charge"], "charge")
        spin = (
            _strict_scalar_int(archive["spin"], "spin")
            if "spin" in archive
            else (int(z.sum()) - charge) % 2
        )
        geometry = Geometry(
            tuple(int(v) for v in z),
            torch.as_tensor(
                _real_array(archive["positions_bohr"], "positions_bohr"), dtype=torch.float64
            ),
            charge,
            spin,
        )
        expected = selected_domain.validate(geometry)
        metadata = _metadata(archive)
        saved_hash = metadata.get("domain_hash")
        if saved_hash is not None and saved_hash != selected_domain.domain_hash:
            raise ValueError(
                "density archive belongs to a different FrozenDE version/configuration"
            )
        arrays = {
            key: _real_array(archive[key], key) for key in ("coords", "weights", "rho", "grad_u")
        }
        g = len(arrays["rho"]) if arrays["rho"].ndim == 1 else -1
        if (
            arrays["coords"].shape != (g, 3)
            or arrays["grad_u"].shape != (g, 3)
            or arrays["weights"].shape != (g,)
            or g < 1
        ):
            raise ValueError("expected coords/grad_u[G,3], rho/weights[G]")
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("density data must be finite")
        if (arrays["weights"] <= 0).any() or (arrays["rho"] < 0).any():
            raise ValueError("quadrature weights must be positive and density nonnegative")
        n_grid = float(arrays["weights"] @ arrays["rho"])
        if abs(n_grid - expected) > mass_rtol * max(1, expected):
            raise ValueError("sampled density integral is inconsistent with FrozenDE geometry")
        tensors = {
            key: torch.as_tensor(value.copy(), dtype=dtype, device=device)
            for key, value in arrays.items()
        }
        for key in (
            "pauli_target",
            "ts_target",
            "tw_target",
            "xc_target",
            "pauli_raw_difference",
            "pauli_empirical_error_budget",
        ):
            if key in archive:
                value = _real_array(archive[key], key)
                if value.size != 1 or not np.isfinite(value).all():
                    raise ValueError(f"{key} must be a finite scalar")
                tensors[key] = torch.as_tensor(value.reshape(()).copy(), dtype=dtype, device=device)
        if "pauli_target" in tensors and float(tensors["pauli_target"]) < 0:
            raise ValueError("negative Pauli target; investigate teacher or grid errors")
        derivative_fields = {"gradient_operator", "functional_derivative_target"}
        if derivative_fields.intersection(archive.files):
            if not derivative_fields.issubset(archive.files):
                raise ValueError("derivative targets require both gradient_operator and potential")
            if metadata.get("functional_derivative_reliable") is not True:
                raise ValueError("functional derivative target lacks reliable provenance")
            for key in derivative_fields:
                value = _real_array(archive[key], key)
                expected_shape = (g, 3, g) if key == "gradient_operator" else (g,)
                if value.shape != expected_shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid {key} shape/values")
                tensors[key] = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if normalize_electrons:
            represented_mass = float(torch.sum(tensors["weights"] * tensors["rho"]))
            normalization_factor = expected / represented_mass
            roundoff = 32 * torch.finfo(dtype).eps
            if (
                "functional_derivative_target" in tensors
                and abs(normalization_factor - 1) > roundoff
            ):
                raise ValueError(
                    "cannot normalize off-mass density with reliable derivative labels; "
                    "the target belongs to a different density"
                )
            # Repeat the global correction once to avoid a summation-order
            # artifact crossing the exact two-electron branch. No threshold
            # for genuine real-valued N>2 is widened.
            applied_factor = 1.0
            for _ in range(2):
                correction = expected / float(torch.sum(tensors["weights"] * tensors["rho"]))
                tensors["rho"] = tensors["rho"] * correction
                tensors["grad_u"] = tensors["grad_u"] * math.sqrt(correction)
                applied_factor *= correction
            if "gradient_operator" in tensors:
                tensors["grad_u"] = torch.einsum(
                    "gai,i->ga", tensors["gradient_operator"], torch.sqrt(tensors["rho"])
                )
            metadata.update(
                raw_grid_electrons=n_grid,
                raw_electron_mass_error=abs(n_grid - expected),
                electron_normalization_factor=applied_factor,
                density_representation="quadrature_mass_normalized",
                normalization_status="NUMERICAL_density_rescaling_not_a_certificate",
                labels_recomputed_at_normalized_density=False,
            )
        else:
            metadata.setdefault("density_representation", "raw_quadrature_samples")
        represented_mass = float(torch.sum(tensors["weights"] * tensors["rho"]))
    tensors.update(
        geometry=geometry,
        metadata=metadata,
        family_id=chemical_family(z, charge, spin),
        geometry_key=canonical_geometry_key(z, geometry.positions_bohr.numpy(), charge, spin),
        domain_hash=selected_domain.domain_hash,
        expected_electrons=expected,
        electron_mass_error=abs(represented_mass - expected),
        raw_electron_mass_error=abs(n_grid - expected),
    )
    return tensors


def save_reference_npz(path, reference, *, domain=None, overwrite=False):
    """Atomically save a non-pickled, domain-bound numerical reference archive."""
    selected_domain = FrozenDE() if domain is None else domain
    z = np.asarray(reference["atomic_numbers"])
    if z.ndim != 1 or z.dtype.kind not in "iu":
        raise ValueError("atomic_numbers must be an integer vector")
    charge = _strict_scalar_int(reference["charge"], "charge")
    spin = _strict_scalar_int(reference.get("spin", (int(z.sum()) - charge) % 2), "spin")
    geometry = Geometry(
        tuple(int(v) for v in z),
        torch.as_tensor(reference["positions_bohr"], dtype=torch.float64),
        charge,
        spin,
    )
    selected_domain.validate(geometry)
    metadata = dict(reference.get("metadata", {}))
    if metadata.get("domain_hash") not in {None, selected_domain.domain_hash}:
        raise ValueError("reference and requested save domain differ")
    metadata.update(
        schema_version=SCHEMA_VERSION,
        spatial_units="Bohr",
        kinetic_convention="-Delta",
        domain_version=selected_domain.version,
        domain_hash=selected_domain.domain_hash,
        physical_ground_state_verified=False,
    )
    values = {}
    for key, value in reference.items():
        if key == "metadata" or isinstance(value, dict):
            continue
        array = np.asarray(value)
        if array.dtype.kind == "O":
            raise ValueError(f"object arrays are not allowed: {key}")
        values[key] = array
    values["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing reference: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **values)
        # Validate what will actually be consumed before publishing the file.
        load_density_npz(temporary, domain=selected_domain, target=None)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output


def split_by_family(paths: Iterable, *, test_fraction=0.2, seed=17, domain=None):
    """Deterministic split with zero composition-family overlap.

    With only one family, test is intentionally empty. That is a training-only
    experiment, never evidence of chemical generalization.
    """
    if not 0 <= test_fraction < 1:
        raise ValueError("test_fraction must lie in [0,1)")
    families = {}
    for path in sorted(map(Path, paths)):
        sample = load_density_npz(path, domain=domain, target=None)
        families.setdefault(sample["family_id"], []).append(path)
    keys = sorted(families)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_test = (
        0
        if len(keys) < 2 or test_fraction == 0
        else min(len(keys) - 1, max(1, round(len(keys) * test_fraction)))
    )
    test_keys = set(keys[:n_test])
    train = sorted(path for key in keys if key not in test_keys for path in families[key])
    test = sorted(path for key in keys if key in test_keys for path in families[key])
    return train, test


class DensityDataset(torch.utils.data.Dataset):
    """Small variable-grid archives; callers choose a custom batching policy."""

    def __init__(self, paths, *, domain=None, device="cpu", target="pauli_target"):
        self.paths = tuple(map(Path, paths))
        self.domain = FrozenDE() if domain is None else domain
        self.device, self.target = device, target

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return load_density_npz(
            self.paths[index], domain=self.domain, device=self.device, target=self.target
        )
