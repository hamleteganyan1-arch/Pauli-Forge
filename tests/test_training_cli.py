"""Meaningful fit/reload, split isolation and derivative-supervision checks."""

import json
import math
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from pauliforge.data import save_reference_npz
from pauliforge.domain import FrozenDE
from pauliforge.training import (
    ExperimentConfig,
    evaluate_directory,
    export_checkpoint,
    load_checkpoint,
    read_npz,
    train_directory,
)


def _sample(path, electrons=3, *, derivatives=False):
    axis = torch.linspace(-2, 2, 3, dtype=torch.float64)
    coords = torch.cartesian_prod(axis, axis, axis)
    weights = torch.full((len(coords),), 8.0, dtype=torch.float64)
    rho = torch.exp(-0.5 * coords.square().sum(-1))
    rho = rho * electrons / (rho * weights).sum()
    grad_u = -0.5 * coords * rho.sqrt()[:, None]
    metadata = {"provenance": "synthetic_trial_density_test", "quality": "EMPIRICAL"}
    reference = {
        "coords": coords.numpy(),
        "weights": weights.numpy(),
        "rho": rho.numpy(),
        "grad_u": grad_u.numpy(),
        "pauli_target": np.array(0.7),
        "xc_target": np.array(-1.2),
        "atomic_numbers": np.ones(electrons, dtype=int),
        "positions_bohr": np.column_stack(
            (np.arange(electrons) * 1.5, np.zeros(electrons), np.zeros(electrons))
        ),
        "charge": np.array(0),
        "spin": np.array(electrons % 2),
        "metadata": metadata,
    }
    if derivatives:
        from pauliforge.density import (
            DensityInput,
            cartesian_gradient_operator,
            functional_derivative,
        )
        from pauliforge.model import PauliKEDF

        derivative_operator = cartesian_gradient_operator(coords)
        density = DensityInput.from_density_values(coords, weights, rho, derivative_operator)
        torch.manual_seed(13)
        teacher = PauliKEDF(hidden=8).double()
        derivative = functional_derivative(lambda d: teacher.forward_density(d).pauli, density)
        reference.update(
            gradient_operator=derivative_operator.numpy(),
            functional_derivative_target=derivative.detach().numpy(),
            grad_u=density.grad_u.numpy(),
        )
        metadata.update(
            functional_derivative_reliable=True,
            derivative_provenance="reliable_density_functional_derivative",
            derivative_target="pauli_target",
        )
    save_reference_npz(path, reference)
    return path


def _prediction(model, sample, mode="kinetic_hybrid"):
    with torch.no_grad():
        out = model(sample["coords"], sample["weights"], sample["rho"], sample["grad_u"])
    return out.pauli if mode == "kinetic_hybrid" else out.exc


@pytest.mark.parametrize(
    "settings",
    [
        {"epochs": 0},
        {"lr": math.nan},
        {"mode": "mixed"},
        {"derivative_weight": -1},
        {"device": "quantum"},
    ],
)
def test_invalid_experiment_settings(settings):
    with pytest.raises(ValueError):
        ExperimentConfig(**settings)


def test_training_checkpoint_reload_reproducibility_and_portable_export(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    sample_path = _sample(data / "h3.npz")
    output = tmp_path / "model.pt"
    settings = ExperimentConfig(epochs=8, hidden=8, lr=0.02, seed=17)
    model = train_directory(data, config=settings, output=output)
    loaded = load_checkpoint(output)
    sample = read_npz(sample_path)
    assert torch.equal(_prediction(model, sample), _prediction(loaded.model, sample))
    assert loaded.metadata["domain_hash"] == FrozenDE().domain_hash
    assert loaded.metadata["domain"]["version"] == "hcno-v1"
    assert loaded.metadata["validation_scope"] == "fit_only_no_holdout"
    history = loaded.metadata["history"]
    assert history[-1]["train_mse_per_electron"] < history[0]["train_mse_per_electron"]
    repeated = train_directory(data, config=settings, output=tmp_path / "repeated.pt")
    assert torch.equal(_prediction(model, sample), _prediction(repeated, sample))
    exported = export_checkpoint(output, tmp_path / "portable.json")
    assert exported["format"] == "pauliforge-portable-state-json-v1"
    assert exported["model_state"]
    portable = load_checkpoint(tmp_path / "portable.json")
    assert torch.equal(_prediction(model, sample), _prediction(portable.model, sample))
    assert (
        json.loads((tmp_path / "portable.json").read_text())["domain_hash"]
        == FrozenDE().domain_hash
    )
    assert not list(tmp_path.glob(".model.pt.*"))


def test_family_heldout_and_explicit_fit_metrics(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _sample(data / "h3.npz", 3)
    _sample(data / "h3-another-geometry.npz", 3)
    _sample(data / "h4.npz", 4)
    checkpoint = tmp_path / "model.pt"
    train_directory(data, epochs=1, output=checkpoint)
    loaded = load_checkpoint(checkpoint)
    assert loaded.metadata["validation_families"]
    assert not (
        set(loaded.metadata["training_families"]) & set(loaded.metadata["validation_families"])
    )
    with pytest.raises(ValueError, match="training molecule families"):
        evaluate_directory(checkpoint, data)
    heldout = tmp_path / "heldout"
    heldout.mkdir()
    for filename in loaded.metadata["validation_files"]:
        shutil.copy2(data / filename, heldout / filename)
    report = evaluate_directory(checkpoint, heldout)
    assert report["scope"] == "held_out_molecule_families"
    assert report["continuum_certified"] is False
    assert report["n_samples"] > 0
    assert (
        evaluate_directory(checkpoint, data, allow_training_families=True)["scope"]
        == "training_family_fit"
    )


def test_checkpoint_domain_tampering_rejected(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _sample(data / "sample.npz")
    checkpoint = tmp_path / "model.pt"
    train_directory(data, epochs=1, output=checkpoint)
    content = torch.load(checkpoint, weights_only=True)
    content["domain"]["max_atoms"] += 1
    torch.save(content, checkpoint)
    with pytest.raises(ValueError, match="domain hash"):
        load_checkpoint(checkpoint)


def test_xc_mode_uses_signed_separate_target(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _sample(data / "sample.npz")
    checkpoint = tmp_path / "xc.pt"
    model = train_directory(
        data,
        config=ExperimentConfig(mode="analytical_pauli_xc_ml", epochs=2, hidden=8),
        output=checkpoint,
    )
    loaded = load_checkpoint(checkpoint)
    assert loaded.metadata["target"] == "xc_target"
    assert loaded.config.mode == "analytical_pauli_xc_ml"
    sample = read_npz(data / "sample.npz", mode=loaded.config.mode)
    assert torch.equal(
        _prediction(model, sample, loaded.config.mode),
        _prediction(loaded.model, sample, loaded.config.mode),
    )


def test_derivative_training_requires_reliable_density_representation(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _sample(data / "plain.npz")
    with pytest.raises(ValueError, match="density gradient operator"):
        train_directory(data, epochs=1, derivative_weight=0.1, output=tmp_path / "bad.pt")
    assert not (tmp_path / "bad.pt").exists()
    (data / "plain.npz").unlink()
    _sample(data / "derivative.npz", derivatives=True)
    train_directory(
        data,
        config=ExperimentConfig(epochs=1, derivative_weight=0.1, hidden=8),
        output=tmp_path / "derivative.pt",
    )
    loaded = load_checkpoint(tmp_path / "derivative.pt")
    assert math.isfinite(loaded.metadata["history"][0]["train_derivative_mse"])
    assert loaded.metadata["history"][0]["train_derivative_mse"] > 0


def test_cli_demo_and_full_workflow(tmp_path):
    from pauliforge.cli import main

    data = tmp_path / "data"
    data.mkdir()
    _sample(data / "sample.npz")
    checkpoint = tmp_path / "model.pt"
    assert main(["train", str(data), "--epochs", "1", "--output", str(checkpoint)]) == 0
    assert (
        main(
            [
                "evaluate",
                str(checkpoint),
                str(data),
                "--allow-training-families",
                "--output",
                str(tmp_path / "metrics.json"),
            ]
        )
        == 0
    )
    assert main(["export", str(checkpoint), "--output", str(tmp_path / "model.json")]) == 0
    assert (
        main(
            [
                "sweep",
                str(data),
                "--epochs",
                "1",
                "--seeds",
                "17",
                "--learning-rates",
                "0.001",
                "--output-dir",
                str(tmp_path / "sweep"),
            ]
        )
        == 0
    )
    assert (tmp_path / "sweep/manifest.json").exists()
    process = subprocess.run(
        [sys.executable, "-m", "pauliforge.cli", "demo"], check=True, text=True, capture_output=True
    )
    demo = json.loads(process.stdout)
    assert demo["N"] == pytest.approx(3)
    assert demo["continuum_certified"] is False
    assert "trial" in demo["provenance"]
