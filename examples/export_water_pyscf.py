"""Run only after installing PySCF; outputs approximate RKS reference, not exact Ts.

python examples/export_water_pyscf.py
"""
import numpy as np
from pauliforge.teachers.pyscf_reference import closed_shell_ks_reference

# Bohr, neutral H2O; 3 atoms, 10 electrons.
atoms=[("O",(0,0,0)),("H",(0,1.430,1.107)),("H",(0,-1.430,1.107))]
r=closed_shell_ks_reference(atoms, basis="sto-3g", xc="pbe")
np.savez_compressed("water_reference.npz", coords=r["coords"], weights=r["weights"],
    rho=r["rho"],grad_u=r["grad_u"],pauli_target=np.asarray(r["pauli_target"]),
    atomic_numbers=np.array([8,1,1]),positions_bohr=np.array([a[1] for a in atoms]),
    charge=np.array(0))
print("Saved water_reference.npz", "Pauli label:",r["pauli_target"],r["provenance"])
