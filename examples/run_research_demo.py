"""End-to-end numerical demonstration plus two explicitly scoped exact checks.

Generate the reference banks and train the two checkpoints first, following
README.md. This script never promotes numerical QC or ML outputs to proofs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from pauliforge.data import load_density_npz
from pauliforge.decoders.packets import PolynomialPacket, decode_low_mass_packets
from pauliforge.density import DensityInput
from pauliforge.evaluation import ValidatedInference
from pauliforge.modes import AnalyticalPauliXCFunctional
from pauliforge.teachers.dual import verify_rational_primal_dual
from pauliforge.teachers.matrix_sdp import solve_finite_basis
from pauliforge.training import load_checkpoint, read_npz


def source_digest():
    digest = hashlib.sha256()
    for path in sorted(Path("src/pauliforge").rglob("*.py")):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinetic-checkpoint", default="results/demo/kinetic-heldout.pt")
    parser.add_argument("--xc-checkpoint", default="results/demo/xc-heldout.pt")
    parser.add_argument("--density", default="data/processed/reference_demo/H2O.npz")
    parser.add_argument("--output", default="results/demo/research-demo.json")
    args = parser.parse_args(argv)
    kinetic = load_checkpoint(args.kinetic_checkpoint)
    xc = load_checkpoint(args.xc_checkpoint)
    if kinetic.domain.domain_hash != xc.domain.domain_hash:
        raise ValueError("The two demonstration models must use the same FrozenDE")
    sample = read_npz(Path(args.density), domain=kinetic.domain)
    with torch.no_grad():
        predicted = ValidatedInference(kinetic.model, kinetic.domain)(
            sample["geometry"], sample["coords"], sample["weights"], sample["rho"], sample["grad_u"]
        ).result
    raw = load_density_npz(args.density, domain=kinetic.domain, target=None)
    density = DensityInput(raw["coords"], raw["weights"], raw["rho"], raw["grad_u"])
    with np.load(args.density, allow_pickle=False) as archive:
        teacher = solve_finite_basis(
            archive["rho"],
            archive["ao_values_orth"],
            archive["weights"],
            archive["kinetic_orth_minus_delta"],
            float(archive["electrons"]),
            basis_metric="analytic",
        )
        with torch.no_grad():
            mode_b = AnalyticalPauliXCFunctional(xc.model)(
                density,
                teacher,
                float(archive["hartree_target"]),
                float(archive["external_energy_hartree"]),
                float(archive["nuclear_repulsion_hartree"]),
            )
        reference = {
            name: float(archive[name])
            for name in ("ts_target", "tw_target", "pauli_target", "xc_target")
        }
    rational = verify_rational_primal_dual(
        gamma=[["3/2", 0], [0, "1/2"]],
        kinetic=[[1, 0], [0, 3]],
        ao_values=[[1, 0], [0, 1]],
        target_density=["3/2", "1/2"],
        electrons=2,
        dual_density=[-1, -3],
    )
    packets = decode_low_mass_packets(
        [
            PolynomialPacket((0, 0, 0), (1, 1, 1), 1),
            PolynomialPacket((4, 0, 0), (1, 1, 1), "3/2"),
        ]
    )
    report = {
        "status": "NUMERICAL_RESEARCH_DEMONSTRATION",
        "source_sha256": source_digest(),
        "domain_version": kinetic.domain.version,
        "domain_hash": kinetic.domain.domain_hash,
        "continuum_certified": False,
        "reference_water": reference,
        "kinetic_hybrid": {
            "N": float(predicted.n),
            "TW_minus_delta": float(predicted.tw),
            "Pauli_ML": float(predicted.pauli),
            "Ts_ML_minus_delta": float(predicted.ts),
            "matrix_representability_certified": False,
            "mass_normalization": sample["metadata"]["electron_normalization_factor"],
        },
        "water_joint_sdp": {
            key: teacher[key]
            for key in (
                "ts_minus_delta",
                "constraint_residual",
                "dual_lower_numerical",
                "gap_numerical",
                "certified",
                "continuum_certified",
            )
        },
        "analytical_pauli_xc_ml": {
            "total_hartree": float(mode_b.total_hartree),
            "ts_finite_basis_minus_delta": float(mode_b.ts_finite_basis),
            "pauli_finite_basis_minus_delta": float(mode_b.pauli_finite_basis),
            "xc_ml_hartree": float(mode_b.exc_hartree),
            "status": mode_b.status,
            "total_density_derivative_available": mode_b.kinetic_derivative_available,
        },
        "exact_rational_finite_sdp": rational.to_dict(),
        "exact_disjoint_packet_subclass": packets.to_dict(),
        "training": {},
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pyscf", "cvxpy", "e3nn")
        },
        "cuda_test_status": "UNRUN_no_device" if not torch.cuda.is_available() else "UNRUN",
        "limitations": [
            "Small STO-3G/PBE demonstration; no broad chemical accuracy claim.",
            "CH4 is one held-out composition, not a statistically adequate benchmark.",
            "ML-normalized grid differs slightly from raw AO density; labels are numerical.",
            "Mode B's offline kinetic value has no validated total density response.",
            "The exact packet certificate covers only explicit disjoint packets of mass <=2 each.",
        ],
    }
    for name, loaded in [("kinetic_hybrid", kinetic), ("xc", xc)]:
        history = loaded.metadata["history"]
        report["training"][name] = {
            "epochs": len(history),
            "first_train_mse_per_electron": history[0]["train_mse_per_electron"],
            "last_train_mse_per_electron": history[-1]["train_mse_per_electron"],
            "held_out": loaded.metadata["validation_files"],
            "validation": history[-1]["validation"],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    main()
