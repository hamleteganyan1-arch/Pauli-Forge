"""PySCF spin-restricted KS reference extractor, requires optional dependency.

This provides *approximate KS reference* labels, not certified exact Ts[rho].
PySCF kinetic operator is -Delta/2, multiplied by 2 to match our -Delta.
"""
import numpy as np


def closed_shell_ks_reference(atoms_bohr, basis="sto-3g", xc="pbe", charge=0,
                               grid_level=3):
    from pyscf import gto, dft
    mol = gto.M(atom=atoms_bohr, unit="Bohr", basis=basis, charge=charge, spin=0, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = grid_level
    e = mf.kernel()
    if not mf.converged:
        raise RuntimeError("KS SCF did not converge; reference rejected")
    dm = mf.make_rdm1()
    t_ks_minus_delta = 2.0 * np.einsum("ij,ji->", dm, mol.intor("int1e_kin"))
    coords = mf.grids.coords
    weights = mf.grids.weights
    from pyscf.dft import numint
    ao = numint.eval_ao(mol, coords, deriv=1)
    rho4 = numint.eval_rho(mol, ao, dm, xctype="GGA")
    rho = np.maximum(rho4[0], 0.)
    grad_rho = rho4[1:4].T
    # near-tail floor is a numerical approximation and must be error-budgeted
    grad_u = grad_rho / (2. * np.sqrt(np.maximum(rho, 1e-16))[:, None])
    tw_grid = float(np.sum(weights * np.sum(grad_u**2, axis=1)))
    pauli_label = float(t_ks_minus_delta - tw_grid)
    if pauli_label < -1e-3:
        raise RuntimeError("negative Pauli label beyond quadrature tolerance")
    return dict(coords=coords, weights=weights, rho=rho, grad_u=grad_u,
                ts_target=float(t_ks_minus_delta), pauli_target=max(0.,pauli_label),
                tw_target=tw_grid, ks_total_energy_hartree=float(e),
                provenance="RKS_PySCF_reference_not_exact_constrained_search",
                basis=basis, xc=xc)
