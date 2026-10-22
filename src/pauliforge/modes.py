"""Explicit separation of kinetic learning and finite-basis-Pauli/XC learning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .model import DensityXCModel, PauliKEDF
from .physics import von_weizsaecker


@dataclass
class AnalyticalXCOutput:
    total_hartree: torch.Tensor
    total_project_units: torch.Tensor
    ts_finite_basis: torch.Tensor
    tw: torch.Tensor
    pauli_finite_basis: torch.Tensor
    hartree_energy_hartree: torch.Tensor
    external_energy_hartree: torch.Tensor
    exc_hartree: torch.Tensor
    nuclear_energy_hartree: torch.Tensor
    n: torch.Tensor
    status: str = "NUMERICAL_finite_basis_not_continuum_certified"
    continuum_certified: bool = False
    kinetic_derivative_available: bool = False
    derivative_status: str = "BLOCKED_offline_Ts_is_not_a_differentiable_constrained_search"


class _OfflineKineticAutogradGuard(torch.autograd.Function):
    """Prevent an XC-only derivative being mistaken for d(total energy)/d rho."""

    @staticmethod
    def forward(ctx, rho, kinetic):
        return kinetic.clone()

    @staticmethod
    def backward(ctx, gradient):
        raise RuntimeError(
            "Mode B total density derivative unavailable: offline constrained-search "
            "Ts has no validated response; differentiate xc_model separately"
        )


def _checked_teacher_kinetic(result, density, electrons):
    required = ("gamma", "ao_values", "kinetic", "weights", "ts_minus_delta")
    if not all(name in result for name in required):
        raise ValueError("finite-basis teacher needs gamma/A/K/weights for independent validation")
    values = [
        torch.as_tensor(result[name], device=density.rho.device)
        for name in ("gamma", "ao_values", "kinetic")
    ]
    dtype = density.rho.dtype
    for value in values:
        dtype = torch.promote_types(dtype, value.dtype)
    gamma, ao, kinetic = [value.to(dtype=dtype) for value in values]
    weights = torch.as_tensor(result["weights"], device=density.rho.device, dtype=density.rho.dtype)
    dimension = gamma.shape[0] if gamma.ndim == 2 else 0
    if (
        gamma.shape != (dimension, dimension)
        or dimension == 0
        or ao.shape != (len(density.rho), dimension)
        or kinetic.shape != gamma.shape
        or weights.shape != density.weights.shape
    ):
        raise ValueError("finite-basis teacher source matrix/grid shapes are inconsistent")
    if not all(torch.isfinite(value).all() for value in (gamma, ao, kinetic, weights)):
        raise ValueError("finite-basis teacher arrays must be finite")
    tolerance = 2e-6
    if not torch.allclose(weights, density.weights, rtol=1e-10, atol=1e-12):
        raise ValueError("finite-basis teacher quadrature differs from supplied density")
    if not torch.allclose(gamma, gamma.mH, rtol=0, atol=tolerance):
        raise ValueError("finite-basis teacher gamma is not Hermitian")
    if not torch.allclose(kinetic, kinetic.mH, rtol=0, atol=1e-9):
        raise ValueError("finite-basis teacher kinetic is not Hermitian")
    if torch.linalg.eigvalsh((kinetic + kinetic.mH) / 2).min() < -1e-9:
        raise ValueError("finite-basis -Delta kinetic matrix must be positive semidefinite")
    cap = float(result.get("max_occupation", 2.0))
    occupation = torch.linalg.eigvalsh((gamma + gamma.mH) / 2)
    if (
        not math.isfinite(cap)
        or not 0 < cap <= 2
        or not 0 <= electrons <= cap * dimension + tolerance
        or occupation.min() < -tolerance
        or occupation.max() > cap + tolerance
        or abs(torch.trace(gamma).real.item() - electrons) > tolerance * max(1, electrons)
    ):
        raise ValueError("finite-basis teacher violates PSD/cap/electron trace")
    if "gamma_alpha" in result or "gamma_beta" in result:
        if not all(name in result for name in ("gamma_alpha", "gamma_beta", "spin_electrons")):
            raise ValueError("both spin matrices and channel electron counts are required")
        channels = [
            torch.as_tensor(result[name], device=gamma.device, dtype=dtype)
            for name in ("gamma_alpha", "gamma_beta")
        ]
        counts = result["spin_electrons"]
        if len(counts) != 2:
            raise ValueError("two spin channel electron counts required")
        for channel, count in zip(channels, counts):
            if (
                channel.shape != gamma.shape
                or not torch.isfinite(channel).all()
                or not torch.allclose(channel, channel.mH, rtol=0, atol=tolerance)
                or not math.isfinite(float(count))
            ):
                raise ValueError("invalid finite-basis spin channel")
            eig = torch.linalg.eigvalsh((channel + channel.mH) / 2)
            if (
                eig.min() < -tolerance
                or eig.max() > 1 + tolerance
                or abs(torch.trace(channel).real.item() - count) > tolerance * max(1, count)
            ):
                raise ValueError("finite-basis spin channel violates cap I or trace")
        if not torch.allclose(channels[0] + channels[1], gamma, rtol=0, atol=tolerance):
            raise ValueError("spin channel sum does not match total gamma")
    reconstructed = torch.sum((ao @ gamma) * ao.conj(), dim=-1)
    if reconstructed.is_complex() and reconstructed.imag.abs().max() > tolerance:
        raise ValueError("finite-basis teacher density has an imaginary residual")
    if not torch.allclose(reconstructed.real, density.rho, rtol=1e-7, atol=tolerance):
        raise ValueError("finite-basis teacher does not jointly reproduce density")
    objective = torch.trace(kinetic @ gamma)
    if objective.is_complex() and objective.imag.abs() > tolerance:
        raise ValueError("finite-basis teacher energy has an imaginary residual")
    supplied = density.rho.new_tensor(result["ts_minus_delta"])
    if (
        supplied.numel() != 1
        or not torch.isfinite(supplied)
        or not torch.allclose(objective.real, supplied, rtol=1e-7, atol=tolerance)
    ):
        raise ValueError("finite-basis teacher kinetic label disagrees with Tr(K gamma)")
    return _OfflineKineticAutogradGuard.apply(density.rho, objective.real.detach())


class AnalyticalPauliXCFunctional(nn.Module):
    """Mode B assembles an offline matrix teacher and a separate XC surrogate.

    Project kinetic labels use -Delta, i.e. twice physical Hartree kinetic
    energy. Full Hartree totals therefore use Ts_project/2, and doubled-project
    totals multiply *all* Coulomb and XC terms by two. No KS total is added,
    preventing double counting of XC. Supplied Hartree/external terms are
    independent numerical inputs, not secretly inferred from orbitals here.
    """

    mode = "analytical_pauli_xc_ml"

    def __init__(self, xc_model=None):
        super().__init__()
        self.xc_model = xc_model if xc_model is not None else DensityXCModel()
        if not isinstance(self.xc_model, DensityXCModel):
            raise TypeError("Mode B requires its separate DensityXCModel")

    def forward(
        self,
        density,
        finite_basis_result,
        hartree_energy_hartree,
        external_energy_hartree,
        nuclear_energy_hartree=0.0,
    ):
        density.validate()
        result = finite_basis_result
        if not result.get("numerical_admissible", False):
            raise ValueError("independently checked joint finite-basis teacher required")
        target = torch.as_tensor(
            result.get("density_target", []), dtype=density.rho.dtype, device=density.rho.device
        )
        if target.shape != density.rho.shape or not torch.allclose(
            target, density.rho, rtol=1e-7, atol=1e-9
        ):
            raise ValueError("finite-basis teacher target does not match supplied density")
        electrons = float(result.get("electrons", float("nan")))
        n = torch.sum(density.rho * density.weights)
        expected = (
            density.expected_electrons
            if density.expected_electrons is not None
            else float(n.detach())
        )
        mass_error = density.mass_tolerance * max(1, expected)
        if (
            not math.isfinite(electrons)
            or electrons <= 0
            or abs(electrons - expected) > max(2e-6, mass_error)
        ):
            raise ValueError("finite-basis teacher electron number does not match density")
        ts = _checked_teacher_kinetic(result, density, electrons)
        tw = von_weizsaecker(density.grad_u, density.weights)
        pauli = ts - tw
        if pauli.detach().item() < 0:
            raise ValueError(
                "finite-basis kinetic energy is below TW: inspect basis/grid consistency"
            )
        xc = self.xc_model.forward_density(density)
        coulomb = [
            torch.as_tensor(value, dtype=ts.dtype, device=ts.device)
            for value in (hartree_energy_hartree, external_energy_hartree, nuclear_energy_hartree)
        ]
        if any(v.numel() != 1 or not torch.isfinite(v) for v in coulomb):
            raise ValueError("finite scalar Hartree-unit energy components required")
        eh, ev, enn = coulomb
        total = ts / 2 + eh + ev + xc.exc + enn
        return AnalyticalXCOutput(total, 2 * total, ts, tw, pauli, eh, ev, xc.exc, enn, n)


def build_functional(mode="kinetic_hybrid", **kwargs):
    if mode == "kinetic_hybrid":
        return PauliKEDF(**kwargs)
    if mode == "analytical_pauli_xc_ml":
        return AnalyticalPauliXCFunctional(DensityXCModel(**kwargs))
    raise ValueError(f"unknown scientific mode: {mode}")


@dataclass(frozen=True)
class CompactPolynomialFragment:
    """Analytically compact supported *synthetic* density for disjoint tests.

    u=A max(1-|x-center|²/R²,0)^p (p>=2) has an explicit support ball.
    A is normalized on the provided quadrature; its continuum electron number
    is exposed separately. Sampling zeros alone is never a support proof.
    """

    density: object
    center: torch.Tensor
    radius: float
    amplitude: torch.Tensor
    exponent: int = 2

    @staticmethod
    def _evaluate(coords, center, radius, amplitude, exponent):
        displacement = coords - center
        polynomial = torch.clamp(1 - displacement.square().sum(-1) / radius**2, min=0)
        u = amplitude * polynomial**exponent
        grad_u = (
            -2
            * exponent
            * amplitude
            / radius**2
            * polynomial[:, None] ** (exponent - 1)
            * displacement
        )
        return u.square(), grad_u

    @classmethod
    def from_grid(cls, coords, weights, center, radius, electrons=3.0, exponent=2):
        from .density import DensityInput

        center = torch.as_tensor(center, dtype=coords.dtype, device=coords.device)
        radius = float(radius)
        if (
            center.shape != (3,)
            or not torch.isfinite(center).all()
            or not math.isfinite(radius)
            or radius <= 0
            or not isinstance(exponent, int)
            or isinstance(exponent, bool)
            or exponent < 2
            or not math.isfinite(electrons)
            or electrons <= 0
        ):
            raise ValueError(
                "finite compact ball, integer p>=2 and positive electron mass required"
            )
        raw, _ = cls._evaluate(coords, center, radius, coords.new_tensor(1.0), exponent)
        mass = torch.sum(raw * weights)
        if mass.detach().item() <= 0:
            raise ValueError("grid misses compact support")
        amplitude = torch.sqrt(coords.new_tensor(electrons) / mass)
        rho, grad_u = cls._evaluate(coords, center, radius, amplitude, exponent)
        density = DensityInput(
            coords=coords,
            weights=weights,
            rho=rho,
            grad_u=grad_u,
            gradient_source="analytic_compact_polynomial",
        )
        return cls(density, center, radius, amplitude, exponent)

    @property
    def continuum_electrons(self):
        beta = math.exp(
            math.lgamma(1.5)
            + math.lgamma(2 * self.exponent + 1)
            - math.lgamma(2 * self.exponent + 2.5)
        )
        return self.amplitude.square() * (2 * math.pi * self.radius**3 * beta)

    def validate(self):
        if (
            not isinstance(self.center, torch.Tensor)
            or self.center.shape != (3,)
            or not torch.isfinite(self.center).all()
            or not self.center.is_floating_point()
            or not math.isfinite(self.radius)
            or self.radius <= 0
            or not isinstance(self.exponent, int)
            or isinstance(self.exponent, bool)
            or self.exponent < 2
            or not isinstance(self.amplitude, torch.Tensor)
            or self.amplitude.numel() != 1
            or not torch.isfinite(self.amplitude)
            or self.amplitude.detach().item() <= 0
        ):
            raise ValueError("invalid analytic compact support parameters")
        self.density.validate()
        rho, grad_u = self._evaluate(
            self.density.coords, self.center, self.radius, self.amplitude, self.exponent
        )
        if not torch.allclose(rho, self.density.rho, rtol=1e-10, atol=1e-12):
            raise ValueError("fragment samples differ from analytically supported density")
        if not torch.allclose(grad_u, self.density.grad_u, rtol=1e-10, atol=1e-12):
            raise ValueError("fragment gradient differs from its analytic density representation")
