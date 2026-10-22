"""The domain gate checks finite chemical parameters, never D_E provenance."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from pauliforge.density import DensityInput
from pauliforge.domain import FrozenDE, Geometry
from pauliforge.evaluation import ValidatedInference
from pauliforge.model import PauliKEDF

T = torch.float64


def geometry(z=(1, 1), separation=1.4, charge=0, spin=0):
    return Geometry(
        z, torch.tensor([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]], dtype=T), charge, spin
    )


def test_domain_serialization_and_immutable_hash():
    dom = FrozenDE(allowed_atomic_numbers=[8, 1, 7, 6])
    restored = FrozenDE.from_dict(json.loads(json.dumps(dom.to_dict())))
    assert restored == dom
    assert len(dom.domain_hash) == 64
    assert dom.domain_hash == restored.fingerprint
    assert dom.domain_hash != replace(dom, max_atoms=7).domain_hash
    assert dom.domain_hash != replace(dom, version="hcno-v2").domain_hash
    with pytest.raises(FrozenInstanceError):
        dom.max_atoms = 2
    with pytest.raises(ValueError, match="unknown"):
        FrozenDE.from_dict({"silent_extrapolation": True})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allowed_atomic_numbers": [1.5, 6]},
        {"allowed_atomic_numbers": [1, 1]},
        {"max_atoms": 1.1},
        {"min_separation_bohr": 0.0},
        {"max_centered_radius_bohr": float("inf")},
        {"min_electrons": 0},
        {"allowed_spins": [-1]},
        {"units": "Angstrom"},
        {"version": ""},
    ],
)
def test_invalid_domain_settings(kwargs):
    with pytest.raises(ValueError):
        FrozenDE(**kwargs)


@pytest.mark.parametrize(
    "geo",
    [
        geometry(z=(1, 2)),
        geometry(z=(1, 1.9)),
        geometry(charge=0.0),
        geometry(spin=1),
        geometry(spin=4),
        geometry(charge=1),
        geometry(separation=0.2),
        geometry(separation=30.0),
        Geometry((1, 1), torch.tensor([[float("nan"), 0, 0], [1, 0, 0]])),
    ],
)
def test_frozen_domain_refusals(geo):
    with pytest.raises(ValueError):
        FrozenDE().validate(geo)


def test_atom_and_electron_limits_and_explicit_extension():
    with pytest.raises(ValueError, match="atom count"):
        FrozenDE(max_atoms=1).validate(geometry())
    with pytest.raises(ValueError, match="electron"):
        FrozenDE(max_electrons=1).validate(geometry())
    helium = Geometry((2,), torch.zeros((1, 3), dtype=T), spin=0)
    with pytest.raises(ValueError, match="element"):
        FrozenDE().validate(helium)
    extended = FrozenDE(allowed_atomic_numbers=(1, 2, 6, 7, 8), version="hcno-he-v1")
    assert extended.validate(helium) == 2
    assert extended.domain_hash != FrozenDE().domain_hash


def test_geometry_inversion_translation_permutation_and_spin_parity():
    dom = FrozenDE()
    geo = geometry()
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=T))
    moved = Geometry(
        tuple(reversed(geo.nuclear_charges)), geo.positions_bohr.flip(0) @ reflection + 100, spin=0
    )
    assert dom.validate(geo) == dom.validate(moved) == 2
    assert dom.validate(Geometry((1,), torch.zeros((1, 3), dtype=T))) == 1
    with pytest.raises(ValueError, match="spin"):
        dom.validate(Geometry((1,), torch.zeros((1, 3), dtype=T), spin=0))


def test_valid_geometry_and_density_do_not_verify_ground_state():
    coords = torch.tensor([[-1.0, 0, 0], [1.0, 0, 0]], dtype=T)
    density = DensityInput(
        coords, torch.ones(2, dtype=T), torch.ones(2, dtype=T), torch.zeros((2, 3), dtype=T)
    )
    dom = FrozenDE()
    out = ValidatedInference(PauliKEDF().double(), dom)(geometry(), density)
    assert out.geometry_verified
    assert not out.physical_ground_state_verified
    assert out.density_domain == "D_trial"
    assert out.domain_hash == dom.domain_hash
    assert out.result.pauli.item() == 0
