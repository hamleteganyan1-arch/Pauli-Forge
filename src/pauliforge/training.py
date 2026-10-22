"""Reproducible domain-gated training; all learned errors are empirical.

No quantum chemistry package is imported during training or inference.  A low
validation error is neither a representability proof nor a continuum bound.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .domain import FrozenDE

MODES = ("kinetic_hybrid", "analytical_pauli_xc_ml")
CHECKPOINT_SCHEMA = "pauliforge-checkpoint-v1"


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "kinetic_hybrid"
    epochs: int = 30
    lr: float = 1e-3
    seed: int = 17
    device: str = "cpu"
    hidden: int = 48
    encoder: str = "moment"
    validation_fraction: float = 0.2
    derivative_weight: float = 0.0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not isinstance(self.hidden, int) or self.hidden < 1:
            raise ValueError("hidden must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("learning rate must be finite and positive")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0,1)")
        if not math.isfinite(self.derivative_weight) or self.derivative_weight < 0:
            raise ValueError("derivative_weight must be finite and nonnegative")
        if self.device not in ("auto", "cpu") and not self.device.startswith("cuda"):
            raise ValueError("device must be cpu, auto, or a CUDA device")


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    domain: FrozenDE
    config: ExperimentConfig
    metadata: dict[str, Any]


def configure_reproducibility(seed: int, device: str = "cpu") -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this runtime has no available GPU")
    if device.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    return torch.device(device)


def _atomic_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch_save(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _target_name(mode):
    return "pauli_target" if mode == "kinetic_hybrid" else "xc_target"


def build_model(mode, model_config=None):
    from .model import PauliKEDF
    from .modes import DensityXCModel

    if mode not in MODES:
        raise ValueError("unknown physical mode")
    model_type = PauliKEDF if mode == "kinetic_hybrid" else DensityXCModel
    return model_type(**(model_config or {})).double()


def read_npz(path: Path, device="cpu", domain=None, mode="kinetic_hybrid"):
    """Compatibility wrapper around the always domain-gated density loader."""
    from .data import load_density_npz

    data = load_density_npz(
        Path(path),
        domain=domain or FrozenDE(),
        device=device,
        dtype=torch.float64,
        target=_target_name(mode),
        normalize_electrons=True,
    )
    with np.load(path, allow_pickle=False) as archive:
        for key in ("gradient_operator", "functional_derivative_target"):
            if key in archive:
                data[key] = torch.as_tensor(archive[key], dtype=torch.float64, device=device)
    return data


def _value(model, sample, mode):
    out = model(sample["coords"], sample["weights"], sample["rho"], sample["grad_u"])
    return (out.pauli if mode == "kinetic_hybrid" else out.exc), out


def _derivative_loss(model, sample, mode):
    from .density import DensityInput, functional_derivative

    metadata = sample.get("metadata", {})
    if (
        metadata.get("derivative_provenance") != "reliable_density_functional_derivative"
        or metadata.get("derivative_target") != _target_name(mode)
        or "gradient_operator" not in sample
        or "functional_derivative_target" not in sample
    ):
        raise ValueError(
            "derivative loss requires a density gradient operator and reliable, "
            "mode-matched functional derivative labels"
        )
    density = DensityInput.from_density_values(
        sample["coords"], sample["weights"], sample["rho"], sample["gradient_operator"]
    )

    def functional(item):
        out = model.forward_density(item)
        return out.pauli if mode == "kinetic_hybrid" else out.exc

    potential = functional_derivative(functional, density, gauge="density_mean", create_graph=True)
    target = sample["functional_derivative_target"]
    if target.shape != potential.shape or not torch.isfinite(target).all():
        raise ValueError("functional derivative target must be finite and match the density grid")
    mass = sample["weights"] * sample["rho"]
    target = target - (target * mass).sum() / mass.sum()
    return (mass * (potential - target).square()).sum() / mass.sum()


def _sample_family(sample):
    return str(sample["family_id"])


def _metrics(model, samples, mode):
    if not samples:
        return {"n_samples": 0, "energy_mae_per_electron": None, "status": "UNRUN"}
    errors, mass_errors, violations = [], [], 0
    with torch.no_grad():
        for sample in samples:
            value, out = _value(model, sample, mode)
            n = float(out.n)
            errors.append(abs(float(value - sample[_target_name(mode)].reshape(()))) / max(1.0, n))
            geo = sample["geometry"]
            mass_errors.append(abs(n - (sum(geo.nuclear_charges) - geo.charge)))
            if mode == "kinetic_hybrid":
                violations += int(float(out.pauli) < -1e-12)
    return {
        "n_samples": len(samples),
        "target": _target_name(mode),
        "energy_mae_per_electron": sum(errors) / len(errors),
        "max_electron_number_error": max(mass_errors),
        "pauli_nonnegativity_violations": violations if mode == "kinetic_hybrid" else None,
        "status": "EMPIRICAL",
    }


def _domain_from_file(path):
    if path is None:
        return FrozenDE()
    with Path(path).open(encoding="utf-8") as stream:
        return FrozenDE.from_dict(json.load(stream))


def train_directory(
    directory,
    epochs=None,
    lr=None,
    output="model.pt",
    seed=None,
    domain_json=None,
    *,
    config=None,
    mode=None,
    device=None,
    validation_fraction=None,
    derivative_weight=None,
):
    """Fit one physical mode; group composition/charge/spin families together."""
    from .data import split_by_family

    raw = asdict(config) if isinstance(config, ExperimentConfig) else dict(config or {})
    for name, value in {
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "mode": mode,
        "device": device,
        "validation_fraction": validation_fraction,
        "derivative_weight": derivative_weight,
    }.items():
        if value is not None:
            raw[name] = value
    cfg = ExperimentConfig(**raw)
    runtime_device = configure_reproducibility(cfg.seed, cfg.device)
    domain = _domain_from_file(domain_json)
    paths = sorted(Path(directory).glob("*.npz"))
    if not paths:
        raise ValueError("no .npz reference samples found")
    samples = {p: read_npz(p, runtime_device, domain, cfg.mode) for p in paths}
    families = {_sample_family(s) for s in samples.values()}
    if cfg.validation_fraction and len(families) > 1:
        train_paths, val_paths = split_by_family(
            paths, test_fraction=cfg.validation_fraction, seed=cfg.seed, domain=domain
        )
    else:
        train_paths, val_paths = paths, []
    train_samples = [samples[Path(p)] for p in train_paths]
    val_samples = [samples[Path(p)] for p in val_paths]
    train_families = {_sample_family(s) for s in train_samples}
    val_families = {_sample_family(s) for s in val_samples}
    if train_families & val_families or not train_samples:
        raise ValueError("invalid molecule-family split")
    model = build_model(cfg.mode, {"hidden": cfg.hidden, "encoder": cfg.encoder}).to(runtime_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        losses, derivatives = [], []
        for sample in train_samples:
            optimizer.zero_grad(set_to_none=True)
            value, out = _value(model, sample, cfg.mode)
            target = sample[_target_name(cfg.mode)].reshape(())
            if not torch.isfinite(target):
                raise ValueError("reference target must be finite")
            if cfg.mode == "kinetic_hybrid" and float(target) < -1e-10:
                raise ValueError("negative Pauli label: inspect reference and quadrature errors")
            energy_loss = ((value - target) / out.n.clamp_min(1.0)).square()
            dloss = (
                _derivative_loss(model, sample, cfg.mode)
                if cfg.derivative_weight
                else torch.zeros_like(energy_loss)
            )
            loss = energy_loss + cfg.derivative_weight * dloss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(energy_loss.detach()))
            derivatives.append(float(dloss.detach()))
        model.eval()
        row = {
            "epoch": epoch + 1,
            "train_mse_per_electron": sum(losses) / len(losses),
            "train_derivative_mse": sum(derivatives) / len(derivatives),
            "validation": _metrics(model, val_samples, cfg.mode),
        }
        history.append(row)
        if epoch == 0 or (epoch + 1) % max(1, cfg.epochs // 5) == 0:
            print(json.dumps(row, allow_nan=False))
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": model.get_config(),
        "experiment": asdict(cfg),
        "domain": domain.to_dict(),
        "domain_hash": domain.domain_hash,
        "history": history,
        "scientific_status": "EMPIRICAL",
        "semantics": "density_only_unverified_accuracy_not_a_continuum_certificate",
        "training_families": sorted(train_families),
        "validation_families": sorted(val_families),
        "training_files": [Path(p).name for p in train_paths],
        "validation_files": [Path(p).name for p in val_paths],
        "validation_scope": "held_out_molecule_families" if val_samples else "fit_only_no_holdout",
        "target": _target_name(cfg.mode),
        "kinetic_operator": "-Delta",
        "density_units": "electrons/Bohr^3",
        "density_representation": "quadrature_mass_normalized_for_ML_only_not_raw_AO_target",
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "device": str(runtime_device),
            "pauliforge": importlib.metadata.version("pauliforge"),
        },
    }
    _atomic_torch_save(output, checkpoint)
    _atomic_json(
        str(output) + ".history.json", {k: v for k, v in checkpoint.items() if k != "model_state"}
    )
    return model


def load_checkpoint(path, device="cpu"):
    """Load weights-only data; arbitrary Python checkpoint objects are refused."""
    if Path(path).suffix.lower() == ".json":
        with Path(path).open(encoding="utf-8") as stream:
            checkpoint = json.load(stream)
        if checkpoint.get("format") != "pauliforge-portable-state-json-v1":
            raise ValueError("unsupported portable model format")
        dtype_map = {
            "torch.float64": torch.float64,
            "torch.float32": torch.float32,
            "torch.int64": torch.int64,
            "torch.int32": torch.int32,
            "torch.bool": torch.bool,
        }
        tensors = {}
        for name, value in checkpoint["model_state"].items():
            if value["dtype"] not in dtype_map:
                raise ValueError("unsupported portable tensor dtype")
            tensor = torch.tensor(value["values"], dtype=dtype_map[value["dtype"]])
            if list(tensor.shape) != value["shape"] or not torch.isfinite(tensor).all():
                raise ValueError("invalid portable tensor shape or values")
            tensors[name] = tensor
        checkpoint["model_state"] = tensors
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported PauliForge checkpoint schema")
    domain = FrozenDE.from_dict(checkpoint["domain"])
    if domain.domain_hash != checkpoint["domain_hash"]:
        raise ValueError("checkpoint domain hash disagrees with frozen configuration")
    config = ExperimentConfig(**checkpoint["experiment"])
    runtime_device = configure_reproducibility(config.seed, device)
    model = build_model(config.mode, checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(runtime_device).eval()
    return LoadedCheckpoint(model, domain, config, checkpoint)


def evaluate_directory(
    checkpoint, directory, *, device="cpu", allow_training_families=False, output=None
):
    loaded = load_checkpoint(checkpoint, device)
    paths = sorted(Path(directory).glob("*.npz"))
    if not paths:
        raise ValueError("no .npz evaluation samples found")
    runtime_device = next(loaded.model.parameters()).device
    samples = [read_npz(p, runtime_device, loaded.domain, loaded.config.mode) for p in paths]
    overlap = set(loaded.metadata["training_families"]) & {_sample_family(s) for s in samples}
    if overlap and not allow_training_families:
        raise ValueError(
            "evaluation overlaps training molecule families; explicitly allow fit metrics"
        )
    report = _metrics(loaded.model, samples, loaded.config.mode)
    report.update(
        mode=loaded.config.mode,
        domain_hash=loaded.domain.domain_hash,
        scope="training_family_fit" if overlap else "held_out_molecule_families",
        continuum_certified=False,
    )
    if output is not None:
        _atomic_json(output, report)
    return report


def export_checkpoint(checkpoint, output):
    """Portable JSON weights, preserving dynamic electron-number branching."""
    loaded = load_checkpoint(checkpoint)
    report = {k: v for k, v in loaded.metadata.items() if k != "model_state"}
    report["format"] = "pauliforge-portable-state-json-v1"
    report["model_state"] = {
        name: {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
        for name, value in loaded.model.state_dict().items()
    }
    _atomic_json(output, report)
    return report
