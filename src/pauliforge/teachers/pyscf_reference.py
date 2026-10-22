"""Offline PySCF references, never an ingredient of density-only inference.

These are approximate finite-basis HF/KS references, not exact constrained
search labels or continuum certificates. PySCF's ``-Delta/2`` kinetic values
are multiplied by two for the project's ``-Delta`` convention. Coulomb, XC,
and SCF-total quantities explicitly retain their ordinary Hartree convention.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _orthogonal_matrices(dm: np.ndarray, overlap: np.ndarray, kinetic: np.ndarray):
    eigenvalues, vectors = np.linalg.eigh(overlap)
    if eigenvalues.min() <= 1e-12:
        raise ValueError("AO overlap is numerically singular; choose a better-conditioned basis")
    sqrt_s = (vectors * np.sqrt(eigenvalues)) @ vectors.T
    invsqrt_s = (vectors / np.sqrt(eigenvalues)) @ vectors.T
    gamma = sqrt_s @ dm @ sqrt_s
    return (gamma + gamma.T) / 2, invsqrt_s, invsqrt_s @ kinetic @ invsqrt_s


def _prepare_scf(
    atoms_bohr,
    *,
    method,
    basis,
    xc,
    charge,
    spin,
    scf_grid_level,
    conv_tol,
    max_cycle,
    domain,
):
    # PySCF remains a strictly optional, lazy offline dependency.
    try:
        from pyscf import dft, gto, scf
    except ImportError as exc:
        raise RuntimeError("Install pauliforge[qc] to generate offline PySCF references") from exc
    from pauliforge.domain import FrozenDE, Geometry

    name = str(method).upper()
    if name not in {"RHF", "RKS", "UHF", "UKS"}:
        raise ValueError("method must be RHF, RKS, UHF, or UKS")
    if isinstance(charge, bool) or int(charge) != charge:
        raise ValueError("charge must be an integer")
    if spin is not None and (isinstance(spin, bool) or int(spin) != spin):
        raise ValueError("spin must be an integer N_alpha-N_beta")
    # Build the molecule to resolve element names; no SCF has run at this point.
    mol = gto.M(
        atom=atoms_bohr,
        unit="Bohr",
        basis=basis,
        charge=int(charge),
        spin=None if spin is None else int(spin),
        verbose=0,
    )
    selected_domain = FrozenDE() if domain is None else domain
    geometry = Geometry(
        tuple(int(z) for z in mol.atom_charges()),
        mol.atom_coords(),
        charge=int(charge),
        spin=int(mol.spin),
    )
    electrons = selected_domain.validate(geometry)
    if name in {"RHF", "RKS"} and mol.spin != 0:
        raise ValueError("restricted RHF/RKS requires spin=0; use UHF/UKS for open shells")
    if not 0 <= scf_grid_level <= 9:
        raise ValueError("SCF grid level must lie in 0..9")
    if not np.isfinite(conv_tol) or conv_tol <= 0 or max_cycle < 1:
        raise ValueError("invalid SCF convergence controls")
    factory = {"RHF": scf.RHF, "UHF": scf.UHF, "RKS": dft.RKS, "UKS": dft.UKS}[name]
    mf = factory(mol)
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    if name.endswith("KS"):
        mf.xc = xc
        mf.grids.level = scf_grid_level
    energy = float(mf.kernel())
    strategy = "diis"
    if not mf.converged:
        # Near-degenerate stretched molecules can oscillate under DIIS. The
        # public second-order solver changes the numerical SCF strategy, not
        # the Hamiltonian or spin sector. Its convergence is still checked.
        mf = mf.newton()
        mf.max_cycle = max_cycle
        energy = float(mf.kernel())
        strategy = "newton_fallback_after_unconverged_diis"
    if not mf.converged or not np.isfinite(energy):
        raise RuntimeError("SCF did not converge; reference rejected")
    mf._pauliforge_scf_strategy = strategy
    dm = np.asarray(mf.make_rdm1(), dtype=float)
    dm_total = dm.sum(axis=0) if dm.ndim == 3 else dm
    return mol, mf, dm, dm_total, geometry, selected_domain, electrons


def _evaluate_density(mol, dm_total, grid_level, *, density_floor, positive_weights_required=True):
    from pyscf import dft
    from pyscf.dft import numint

    if not 0 <= grid_level <= 9:
        raise ValueError("density grid level must lie in 0..9")
    if not np.isfinite(density_floor) or density_floor <= 0:
        raise ValueError("density_floor must be finite and positive")
    grids = dft.gen_grid.Grids(mol)
    grids.level = int(grid_level)
    grids.build()
    coords = np.asarray(grids.coords)
    weights = np.asarray(grids.weights)
    # PySCF pads grids with zero-weight entries. Some coarse Lebedev rules
    # also have genuine NEGATIVE weights: never silently drop those nodes.
    negative_weights = int(np.count_nonzero(weights < 0))
    if negative_weights and positive_weights_required:
        raise ValueError(
            "PySCF grid contains negative quadrature weights; use a positive-weight finer grid"
        )
    nonzero = weights != 0
    coords, weights = coords[nonzero], weights[nonzero]
    ao = numint.eval_ao(mol, coords, deriv=1)
    rho4 = np.asarray(numint.eval_rho(mol, ao, dm_total, xctype="GGA", hermi=1))
    rho_raw = rho4[0]
    if not np.isfinite(rho4).all() or rho_raw.min(initial=0.0) < -1e-12:
        raise RuntimeError("nonfinite or substantially negative reference density")
    rho = np.maximum(rho_raw, 0.0)
    grad_rho = rho4[1:4].T
    grad_u = grad_rho / (2 * np.sqrt(np.maximum(rho, density_floor))[:, None])
    grad_u[rho == 0] = 0
    tw = float(np.sum(weights * np.sum(grad_u**2, axis=1)))
    return {
        "coords": coords,
        "weights": weights,
        "rho": rho,
        "grad_u": grad_u,
        "grad_rho": grad_rho,
        "ao_values": ao[0],
        "tw_target": tw,
        "grid_electrons": float(weights @ rho),
        "negative_quadrature_weights": negative_weights,
        "tail_mass_below_floor": float(weights[rho < density_floor] @ rho[rho < density_floor]),
        "negative_density_roundoff_mass": float(weights @ np.maximum(-rho_raw, 0)),
    }


def pyscf_reference(
    atoms_bohr: Sequence[Any],
    *,
    method: str = "RKS",
    basis: str = "sto-3g",
    xc: str = "pbe",
    charge: int = 0,
    spin: int | None = None,
    grid_level: int = 2,
    scf_grid_level: int = 3,
    conv_tol: float = 1e-10,
    max_cycle: int = 100,
    density_floor: float = 1e-24,
    mass_rtol: float = 2e-3,
    pauli_error_tolerance: float = 2e-3,
    include_ao: bool = True,
    domain=None,
) -> dict[str, Any]:
    """Generate a provenance-rich RHF/RKS/UHF/UKS reference.

    ``domain=None`` uses the default frozen HCNO class, never an unrestricted
    chemistry bypass. He/Li benchmarks need an explicitly expanded NEW domain
    version. ``spin`` is N_alpha-N_beta, matching PySCF. No orbitals are used
    by inference: AO quantities returned here are offline teacher diagnostics.

    A tiny negative raw Pauli difference can be projected to zero ONLY within
    the explicit empirical error tolerance and is recorded as such. For N<=2
    the continuum Pauli target is exactly zero and the finite-grid discrepancy
    is retained separately. Neither policy turns numerical data into a proof.
    """
    if mass_rtol <= 0 or pauli_error_tolerance < 0:
        raise ValueError("mass tolerance must be positive and Pauli tolerance nonnegative")
    mol, mf, dm, dm_total, geometry, domain, electrons = _prepare_scf(
        atoms_bohr,
        method=method,
        basis=basis,
        xc=xc,
        charge=charge,
        spin=spin,
        scf_grid_level=scf_grid_level,
        conv_tol=conv_tol,
        max_cycle=max_cycle,
        domain=domain,
    )
    result = _evaluate_density(mol, dm_total, grid_level, density_floor=density_floor)
    overlap = np.asarray(mol.intor("int1e_ovlp"))
    kinetic_half = np.asarray(mol.intor("int1e_kin"))
    kinetic = 2 * kinetic_half
    gamma, inverse_sqrt_s, kinetic_orth = _orthogonal_matrices(dm_total, overlap, kinetic)
    ts = float(np.einsum("ij,ji->", dm_total, kinetic))
    mass_error = abs(result["grid_electrons"] - electrons)
    if mass_error > mass_rtol * max(1, electrons):
        raise RuntimeError("density quadrature misses electron mass; increase grid_level")
    pauli_raw = ts - result["tw_target"]
    error_budget = pauli_error_tolerance * max(1, electrons)
    if pauli_raw < -error_budget:
        raise RuntimeError("negative raw Pauli difference exceeds empirical grid tolerance")
    if electrons <= 2:
        pauli = 0.0
        pauli_policy = "exact_N_le_2_continuum_sector; grid_difference_retained"
    elif pauli_raw < 0:
        pauli = 0.0
        pauli_policy = "zero_projection_within_explicit_empirical_grid_tolerance"
    else:
        pauli, pauli_policy = pauli_raw, "raw_finite_basis_kinetic_minus_grid_TW"
    import pyscf

    metadata = {
        "schema_version": "1.0",
        "method": str(method).upper(),
        "basis": basis,
        "xc": xc if str(method).upper().endswith("KS") else None,
        "pyscf_version": pyscf.__version__,
        "quality": "NUMERICAL_REFERENCE",
        "physical_ground_state_verified": False,
        "provenance": "approximate_finite_basis_SCF_not_exact_constrained_search",
        "kinetic_convention": "-Delta",
        "spatial_units": "Bohr",
        "other_energy_convention": "ordinary_Hartree",
        "grid_level": int(grid_level),
        "scf_grid_level": int(scf_grid_level),
        "quadrature_certified": False,
        "basis_error_certified": False,
        "scf_converged": True,
        "scf_solver_strategy": mf._pauliforge_scf_strategy,
        "scf_conv_tol": conv_tol,
        "electron_mass_error": mass_error,
        "tail_mass_below_floor": result["tail_mass_below_floor"],
        "density_floor": density_floor,
        "gradient_source": "offline_AO_density_derivative_not_prediction_orbitals",
        "functional_derivative_reliable": False,
        "pauli_target_policy": pauli_policy,
        "pauli_empirical_error_budget": error_budget,
        "domain_version": domain.version,
        "domain_hash": domain.domain_hash,
    }
    result.update(
        ts_target=ts,
        kinetic_half_laplacian_target=ts / 2,
        pauli_target=pauli,
        pauli_raw_difference=pauli_raw,
        pauli_empirical_error_budget=error_budget,
        scf_total_energy_hartree=float(mf.e_tot),
        ks_total_energy_hartree=float(mf.e_tot),
        nuclear_repulsion_hartree=float(mol.energy_nuc()),
        external_energy_hartree=float(np.einsum("ij,ji->", dm_total, mol.intor("int1e_nuc"))),
        dm_ao=dm_total,
        overlap=overlap,
        gamma=gamma,
        kinetic_ao_minus_delta=kinetic,
        kinetic_orth_minus_delta=kinetic_orth,
        atomic_numbers=np.asarray(geometry.nuclear_charges, dtype=np.int64),
        positions_bohr=np.asarray(mol.atom_coords()),
        charge=int(charge),
        spin=int(mol.spin),
        electrons=electrons,
        metadata=metadata,
        provenance=metadata["provenance"],
        basis=basis,
        xc=xc if str(method).upper().endswith("KS") else "not_applicable_HF",
    )
    if dm.ndim == 3:
        result["dm_alpha"], result["dm_beta"] = dm
        result["gamma_alpha"] = _orthogonal_matrices(dm[0], overlap, kinetic)[0]
        result["gamma_beta"] = _orthogonal_matrices(dm[1], overlap, kinetic)[0]
    if include_ao:
        result["ao_values_orth"] = result["ao_values"] @ inverse_sqrt_s
    else:
        result.pop("ao_values")
    if str(method).upper().endswith("KS"):
        veff = mf.get_veff(mol, dm)
        result["xc_target"] = float(veff.exc)
        result["hartree_target"] = float(veff.ecoul)
        metadata["xc_label_available"] = True
        metadata["xc_label_convention"] = "ordinary_Hartree_approximate_DFT_functional"
    else:
        # HF exchange is orbital-dependent and is not mislabeled as an exact
        # density-only exchange-correlation constrained-search functional.
        metadata["xc_label_available"] = False
    return result


def closed_shell_ks_reference(
    atoms_bohr,
    basis="sto-3g",
    xc="pbe",
    charge=0,
    grid_level=3,
    **kwargs,
):
    """Backward-compatible RKS adapter, with all v1 provenance checks."""
    return pyscf_reference(
        atoms_bohr,
        method="RKS",
        basis=basis,
        xc=xc,
        charge=charge,
        spin=0,
        grid_level=grid_level,
        **kwargs,
    )


def density_grid_convergence(
    atoms_bohr,
    *,
    levels=(0, 1, 2),
    method="RHF",
    basis="sto-3g",
    xc="pbe",
    charge=0,
    spin=None,
    scf_grid_level=3,
    domain=None,
):
    """Compare quadratures of ONE fixed converged AO density.

    Reported differences are EMPIRICAL diagnostics, never rigorous continuum
    quadrature error certificates or proof of monotonic convergence.
    """
    if not levels or any(isinstance(level, bool) or int(level) != level for level in levels):
        raise ValueError("nonempty integer grid levels required")
    mol, _, _, dm_total, _, _, electrons = _prepare_scf(
        atoms_bohr,
        method=method,
        basis=basis,
        xc=xc,
        charge=charge,
        spin=spin,
        scf_grid_level=scf_grid_level,
        conv_tol=1e-10,
        max_cycle=100,
        domain=domain,
    )
    rows = []
    previous_tw = None
    for level in levels:
        data = _evaluate_density(
            mol, dm_total, int(level), density_floor=1e-24, positive_weights_required=False
        )
        rows.append(
            {
                "grid_level": int(level),
                "points": len(data["rho"]),
                "negative_quadrature_weights": data["negative_quadrature_weights"],
                "electrons": data["grid_electrons"],
                "electron_mass_error": abs(data["grid_electrons"] - electrons),
                "tw_minus_delta": data["tw_target"],
                "tw_change": None if previous_tw is None else abs(data["tw_target"] - previous_tw),
            }
        )
        previous_tw = data["tw_target"]
    return {"status": "EMPIRICAL", "same_scf_density": True, "certified": False, "rows": rows}
