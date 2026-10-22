"""Export numerical RKS water labels with units and frozen-domain provenance."""

from pathlib import Path

from pauliforge.data import save_reference_npz
from pauliforge.teachers.pyscf_reference import pyscf_reference


def main():
    atoms = [("O", (0, 0, 0)), ("H", (0, 1.430, 1.107)), ("H", (0, -1.430, 1.107))]
    reference = pyscf_reference(atoms, method="RKS", basis="sto-3g", xc="pbe", grid_level=2)
    path = save_reference_npz(Path("data/processed/water_reference.npz"), reference, overwrite=True)
    print(
        {
            "path": str(path),
            "pauli_target_minus_delta": reference["pauli_target"],
            "status": reference["metadata"]["quality"],
            "domain": reference["metadata"]["domain_version"],
        }
    )


if __name__ == "__main__":
    main()
