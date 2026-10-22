"""Build a small numerical reference bank, with disjoint domain manifests.

This runs real PySCF calculations; it does not certify continuum labels or
claim a nontrivial trained accuracy from a handful of molecules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pauliforge.data import save_reference_npz
from pauliforge.domain import FrozenDE
from pauliforge.teachers.pyscf_reference import density_grid_convergence, pyscf_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed/reference_demo"))
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument(
        "--xc-labels",
        action="store_true",
        help="Use UKS/RKS for every sample to create a separate XC-labelled bank",
    )
    parser.add_argument(
        "--extended-atoms", action="store_true", help="Separate explicit He/Li benchmark domain"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    domain = FrozenDE()
    bond = 2.06 / 3**0.5
    methane = [("C", (0, 0, 0))] + [
        ("H", tuple(bond * v for v in direction))
        for direction in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    ]
    examples = [
        ("H", [("H", (0, 0, 0))], "UHF", 1),
        ("H2", [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))], "RHF", 0),
        ("H2O", [("O", (0, 0, 0)), ("H", (0, 1.430, 1.107)), ("H", (0, -1.430, 1.107))], "RKS", 0),
        ("CH4", methane, "RKS", 0),
    ]
    for separation in (3.0, 6.0, 10.0):
        examples.append(
            (
                f"H2_r{separation:g}",
                [("H", (0, 0, -separation / 2)), ("H", (0, 0, separation / 2))],
                "RHF",
                0,
            )
        )
    report = {"status": "NUMERICAL_REFERENCE", "domain": domain.to_dict(), "examples": []}
    for name, atoms, method, spin in examples:
        if args.xc_labels:
            method = "UKS" if spin else "RKS"
        reference = pyscf_reference(
            atoms, method=method, spin=spin, grid_level=args.grid_level, domain=domain
        )
        save_reference_npz(args.output / f"{name}.npz", reference, domain=domain, overwrite=True)
        report["examples"].append(
            {
                "name": name,
                "method": method,
                "N": reference["electrons"],
                "Ts_minus_delta": reference["ts_target"],
                "TW_minus_delta": reference["tw_target"],
                "Pauli_target": reference["pauli_target"],
                "raw_Pauli_grid_difference": reference["pauli_raw_difference"],
                "electron_mass_error": reference["metadata"]["electron_mass_error"],
                "scf_solver_strategy": reference["metadata"]["scf_solver_strategy"],
                "physical_ground_state_verified": False,
            }
        )
    report["H2_fixed_density_grid_refinement"] = density_grid_convergence(
        examples[1][1], levels=(0, 1, 2), domain=domain
    )
    if args.extended_atoms:
        expanded = FrozenDE(allowed_atomic_numbers=(1, 2, 3), version="h-he-li-benchmark-v1")
        extended = args.output / "separate_h_he_li_domain"
        extended.mkdir(exist_ok=True)
        (extended / "frozen_de.json").write_text(json.dumps(expanded.to_dict(), indent=2) + "\n")
        report["extended_examples"] = []
        for name, method, spin in [("He", "RHF", 0), ("Li", "UHF", 1)]:
            reference = pyscf_reference(
                [(name, (0, 0, 0))],
                method=method,
                spin=spin,
                grid_level=args.grid_level,
                domain=expanded,
            )
            save_reference_npz(extended / f"{name}.npz", reference, domain=expanded, overwrite=True)
            report["extended_examples"].append(
                {
                    "name": name,
                    "domain_version": expanded.version,
                    "Pauli_target": reference["pauli_target"],
                }
            )
    output = args.output / "reference_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
