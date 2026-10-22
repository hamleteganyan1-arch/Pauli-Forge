"""Research trainer from precomputed density samples; no hidden orbital inference."""
from pathlib import Path
import numpy as np
import torch
from .model import PauliKEDF
from .domain import FrozenDE, Geometry
import json


def read_npz(path: Path, device="cpu", domain=None):
    with np.load(path, allow_pickle=False) as z:
        data = {k: torch.tensor(z[k], dtype=torch.float64, device=device)
                for k in ("coords", "weights", "rho", "grad_u", "pauli_target")}
        if domain is not None:
            required = ("atomic_numbers", "positions_bohr", "charge")
            if not all(k in z for k in required):
                raise ValueError(f"domain metadata missing in {path}; expected {required}")
            geo = Geometry(tuple(int(v) for v in z["atomic_numbers"].tolist()),
                           torch.tensor(z["positions_bohr"], dtype=torch.float64),
                           int(z["charge"].item()))
            n_expected = domain.validate(geo)
            n_grid = float((data["rho"] * data["weights"]).sum())
            if abs(n_grid - n_expected) > 2e-3 * max(1,n_expected):
                raise ValueError(f"quadrature misses electron mass: {path}: {n_grid} vs {n_expected}")
    return data


def train_directory(directory: str, epochs=30, lr=1e-3, output="model.pt", seed=17, domain_json=None):
    torch.manual_seed(seed)
    paths = sorted(Path(directory).glob("*.npz"))
    if not paths:
        raise ValueError("no .npz samples; generate reference data first")
    cfg = json.load(open(domain_json, encoding="utf-8")) if domain_json else {}
    domain_args = {k:v for k,v in cfg.items() if k in FrozenDE.__dataclass_fields__}
    domain = FrozenDE(**domain_args)
    samples = [read_npz(p,domain=domain) for p in paths]
    model = PauliKEDF().double()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        losses=[]
        # Never claim random split validation proves chemistry-generalization.
        for s in samples:
            optim.zero_grad()
            out = model(s["coords"], s["weights"], s["rho"], s["grad_u"])
            target = s["pauli_target"].reshape(())
            if target < -1e-7:
                raise ValueError("negative Pauli label: inspect quadrature/reference")
            loss = ((out.pauli - target)/torch.clamp(out.n, min=1.0))**2
            loss.backward()
            optim.step()
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch+1) % max(1, epochs//5) == 0:
            print(f"epoch={epoch+1} train_MSE_per_electron={sum(losses)/len(losses):.6g}")
    torch.save({"model_state": model.state_dict(), "architecture": "PauliKEDF_v0",
                "semantics": "uncertified_research_model", "n_samples": len(paths)}, output)
    return model
