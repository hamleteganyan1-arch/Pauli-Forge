"""Numerical matrix Pauli constraints and finite-basis density transformations.

Spectral projection is initialization only: it does not preserve a prescribed
spatial density and is not a constrained-search solution.
"""

import math

import torch


def _hermitian(a: torch.Tensor) -> torch.Tensor:
    return (a + a.mH) / 2


def _square_tensor(a, name: str) -> torch.Tensor:
    value = torch.as_tensor(a)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not value.shape[0]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not value.is_floating_point() and not value.is_complex():
        value = value.to(torch.float64)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _overlap_roots(overlap: torch.Tensor):
    s = _square_tensor(overlap, "overlap")
    if not torch.allclose(s, s.mH, atol=1e-10, rtol=1e-10):
        raise ValueError("overlap must be Hermitian")
    e, v = torch.linalg.eigh(_hermitian(s))
    if torch.any(e <= 0):
        raise ValueError("overlap must be strictly positive definite")
    root = (v * torch.sqrt(e).unsqueeze(0)) @ v.mH
    inverse_root = (v * torch.rsqrt(e).unsqueeze(0)) @ v.mH
    return root, inverse_root


def orthonormal_density_matrix(dm: torch.Tensor, overlap: torch.Tensor) -> torch.Tensor:
    """Return ``S**(1/2) D S**(1/2)``; its trace is ``Tr(S D)``."""
    d = _square_tensor(dm, "density matrix")
    s = _square_tensor(overlap, "overlap").to(device=d.device)
    dtype = torch.promote_types(d.dtype, s.dtype)
    d, s = d.to(dtype), s.to(dtype)
    if d.shape != s.shape:
        raise ValueError("density matrix and overlap shapes differ")
    if not torch.allclose(d, d.mH, atol=1e-10, rtol=1e-10):
        raise ValueError("density matrix must be Hermitian")
    root, _ = _overlap_roots(s)
    return _hermitian(root @ d @ root)


def ao_density_matrix(gamma: torch.Tensor, overlap: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`orthonormal_density_matrix` for a positive overlap."""
    g = _square_tensor(gamma, "gamma")
    s = _square_tensor(overlap, "overlap").to(device=g.device)
    dtype = torch.promote_types(g.dtype, s.dtype)
    g, s = g.to(dtype), s.to(dtype)
    if g.shape != s.shape:
        raise ValueError("gamma and overlap shapes differ")
    if not torch.allclose(g, g.mH, atol=1e-10, rtol=1e-10):
        raise ValueError("gamma must be Hermitian")
    _, inverse_root = _overlap_roots(s)
    return _hermitian(inverse_root @ g @ inverse_root)


def sampled_density(gamma: torch.Tensor, basis_values: torch.Tensor) -> torch.Tensor:
    """Evaluate ``diag(A Gamma A*)`` without discarding complex amplitudes."""
    g = _square_tensor(gamma, "gamma")
    a = torch.as_tensor(basis_values, device=g.device)
    dtype = torch.promote_types(g.dtype, a.dtype)
    g, a = g.to(dtype), a.to(dtype)
    if a.ndim != 2 or a.shape[1] != g.shape[0] or not torch.isfinite(a).all():
        raise ValueError("basis_values must be finite [grid,basis]")
    return ((a @ g) * a.conj()).sum(-1).real


def matrix_residuals(gamma: torch.Tensor, n_electrons: float, max_occupation=2.0):
    """Independent finite precision Hermiticity, occupation and trace residuals."""
    g = _square_tensor(gamma, "gamma")
    if not math.isfinite(float(n_electrons)) or n_electrons < 0:
        raise ValueError("electron number must be finite and nonnegative")
    if not math.isfinite(float(max_occupation)) or max_occupation <= 0:
        raise ValueError("occupation cap must be finite and positive")
    values = torch.linalg.eigvalsh(_hermitian(g))
    return {
        "hermiticity": float(torch.max(torch.abs(g - g.mH)).detach()),
        "psd": max(0.0, -float(values.min().detach())),
        "cap": max(0.0, float(values.max().detach()) - max_occupation),
        "trace": abs(float(torch.trace(g).real.detach()) - n_electrons),
    }


def check_pauli_matrix(
    gamma: torch.Tensor, n_electrons: float, max_occupation: float = 2.0, tol: float = 1e-7
) -> bool:
    """Check a spin-summed (cap 2) or one-spin (cap 1) occupation matrix."""
    if not math.isfinite(float(tol)) or tol <= 0:
        return False
    try:
        residuals = matrix_residuals(gamma, n_electrons, max_occupation)
    except (ValueError, TypeError, RuntimeError):
        return False
    return max(residuals[k] for k in ("hermiticity", "psd", "cap")) <= tol and residuals[
        "trace"
    ] <= tol * max(1.0, n_electrons)


def project_occupations(
    matrix: torch.Tensor,
    n_electrons: float,
    max_occupation: float = 2.0,
    iterations: int = 100,
) -> torch.Tensor:
    """Frobenius projection onto PSD, cap and trace, without density recovery."""
    mat = _hermitian(_square_tensor(matrix, "matrix"))
    if not math.isfinite(float(max_occupation)) or max_occupation <= 0:
        raise ValueError("occupation cap must be finite and positive")
    if (
        not math.isfinite(float(n_electrons))
        or n_electrons < 0
        or n_electrons > max_occupation * mat.shape[0]
    ):
        raise ValueError("infeasible electron number for matrix dimension")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if n_electrons == 0:
        return torch.zeros_like(mat)
    if n_electrons == max_occupation * mat.shape[0]:
        return max_occupation * torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
    vals, vecs = torch.linalg.eigh(mat)
    low = float(vals.min().detach()) - max_occupation
    high = float(vals.max().detach())
    for _ in range(iterations):
        mid = (low + high) / 2
        if float(torch.clamp(vals - mid, 0.0, max_occupation).sum().detach()) > n_electrons:
            low = mid
        else:
            high = mid
    occ = torch.clamp(vals - (low + high) / 2, 0.0, max_occupation)
    return _hermitian((vecs * occ.unsqueeze(0)) @ vecs.mH)
