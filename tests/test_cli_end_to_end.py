"""Separate-process CLI and loaded-console checks using explicit trial data."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pauliforge.data import save_reference_npz


def _make_archive(path):
    axis = np.linspace(-2, 2, 3)
    coords = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    weights = np.full(len(coords), 8.0)
    rho = np.exp(-0.5 * np.sum(coords**2, axis=1))
    rho *= 3 / (weights @ rho)
    return save_reference_npz(
        path,
        {
            "coords": coords,
            "weights": weights,
            "rho": rho,
            "grad_u": -0.5 * coords * np.sqrt(rho)[:, None],
            "pauli_target": np.array(0.7),
            "xc_target": np.array(-1.2),
            "atomic_numbers": np.ones(3, dtype=int),
            "positions_bohr": np.array([[-1.5, 0, 0], [0, 0, 0], [1.5, 0, 0]]),
            "charge": np.array(0),
            "spin": np.array(1),
            "metadata": {"provenance": "synthetic_cli_trial_density", "quality": "EMPIRICAL"},
        },
    )


def _cli(*arguments, success=True):
    result = subprocess.run(
        [sys.executable, "-m", "pauliforge.cli", *map(str, arguments)],
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    return result


def test_cli_separate_process_training_export_evaluation_and_sweep(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _make_archive(data / "trial.npz")
    checkpoint = tmp_path / "model.pt"
    _cli("train", data, "--epochs", "2", "--output", checkpoint)
    refused = _cli("evaluate", checkpoint, data, success=False)
    assert "training molecule families" in refused.stderr
    fit = json.loads(_cli("evaluate", checkpoint, data, "--allow-training-families").stdout)
    assert fit["scope"] == "training_family_fit"
    assert fit["continuum_certified"] is False
    portable = tmp_path / "model.json"
    _cli("export", checkpoint, "--output", portable)
    reloaded = json.loads(_cli("evaluate", portable, data, "--allow-training-families").stdout)
    assert reloaded == fit
    sweep = tmp_path / "sweep"
    _cli("sweep", data, "--epochs", "1", "--seeds", "17", "23", "--output-dir", sweep)
    manifest = json.loads((sweep / "manifest.json").read_text())
    assert len(manifest["runs"]) == 2
    assert {run["experiment"]["seed"] for run in manifest["runs"]} == {17, 23}
    assert all(Path(run["checkpoint"]).exists() for run in manifest["runs"])


@pytest.mark.parametrize("mode", ["kinetic_hybrid", "analytical_pauli_xc_ml"])
def test_console_loads_real_checkpoint_history_and_mode(tmp_path, mode):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import pauliforge.ui

    data = tmp_path / "data"
    data.mkdir()
    archive = data / "trial.npz"
    _make_archive(archive)
    checkpoint = tmp_path / "model.pt"
    _cli("train", data, "--epochs", "2", "--mode", mode, "--output", checkpoint)
    app = AppTest.from_file(str(Path(pauliforge.ui.__file__))).run(timeout=25)
    app.sidebar.text_input[0].set_value(str(checkpoint))
    app.sidebar.text_input[1].set_value(str(archive)).run(timeout=25)
    assert not list(app.exception)
    assert any("EMPIRICAL model prediction" in item.value for item in app.warning)
    app.sidebar.radio[0].set_value("Validation").run(timeout=25)
    summary = json.loads(app.json[0].value)
    assert summary["mode"] == mode
    assert summary["provenance"] == "synthetic_cli_trial_density"
    assert summary["continuum_certified"] is False
    assert ("Ts" in summary) is (mode == "kinetic_hybrid")
    assert ("Exc_hartree" in summary) is (mode == "analytical_pauli_xc_ml")
    app.sidebar.radio[0].set_value("Training").run(timeout=25)
    assert not list(app.exception)
    assert len(app.get("plotly_chart")) == 1
    assert len(app.dataframe[0].value) == 2
