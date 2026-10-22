"""Exact *matrix-level* Pauli admissibility; numerical tolerance at evaluation.

For spin summed natural orbitals, 0 <= Gamma <= 2I and Tr Gamma=N.
In a nonorthogonal AO basis D with overlap S, enforce these constraints on
S^(1/2) D S^(1/2), NOT the raw coefficient matrix D.
Projection does not preserve a prescribed spatial density: use joint SDP.
"""
import torch


def _hermitian(a: torch.Tensor) -> torch.Tensor:
    return (a + a.mH) / 2


def orthonormal_density_matrix(dm: torch.Tensor, overlap: torch.Tensor) -> torch.Tensor:
    s = _hermitian(overlap)
    e, v = torch.linalg.eigh(s)
    if torch.any(e <= 0):
        raise ValueError("overlap must be strictly positive definite")
    sqrt_s = (v * torch.sqrt(e).unsqueeze(0)) @ v.mH
    return _hermitian(sqrt_s @ dm @ sqrt_s)


def check_pauli_matrix(gamma: torch.Tensor, n_electrons: float,
                       max_occupation: float = 2.0, tol: float = 1e-7) -> bool:
    if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1]:
        return False
    if not torch.isfinite(gamma).all():
        return False
    if torch.max(torch.abs(gamma - gamma.mH)).item() > tol:
        return False
    v = torch.linalg.eigvalsh(_hermitian(gamma))
    return (v.min().item() >= -tol and v.max().item() <= max_occupation + tol
            and abs(torch.sum(v).item() - n_electrons) <= tol * max(1.0, n_electrons))


def project_occupations(matrix: torch.Tensor, n_electrons: float,
                        max_occupation: float = 2.0, iterations: int = 100) -> torch.Tensor:
    """Frobenius projection onto PSD + cap + trace in orthonormal basis.

    WARNING: this DOES NOT preserve diag(A Gamma A^*)=rho. It is not a
    density-constrained-search solution; intended as initialization/test.
    """
    mat = _hermitian(matrix)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square")
    if n_electrons < 0 or n_electrons > max_occupation * mat.shape[0]:
        raise ValueError("infeasible electron number for matrix dimension")
    vals, vecs = torch.linalg.eigh(mat)
    low = (vals.min() - max_occupation).item()
    high = vals.max().item()
    for _ in range(iterations):
        mid = (low + high) / 2
        if torch.clamp(vals - mid, 0.0, max_occupation).sum().item() > n_electrons:
            low = mid
        else:
            high = mid
    occ = torch.clamp(vals - (low + high) / 2, 0.0, max_occupation)
    return _hermitian((vecs * occ.unsqueeze(0)) @ vecs.mH)
