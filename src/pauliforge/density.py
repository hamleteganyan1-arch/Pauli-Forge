"""Density inputs and derivatives of an explicitly defined discrete functional.

A sampled finite grid is not a certified strong H1 name. Numerical quadrature
errors and supplied error bounds are distinct from continuum certificates.
The dense derivative helper is for small validation grids, not production.
"""

import math
from dataclasses import dataclass, replace
from typing import Callable

import torch


@dataclass(frozen=True)
class QuadratureCertificate:
    """Error-budget metadata, not independent proof of the supplied bounds."""

    electron_error_bound: float | None = None
    tw_error_bound: float | None = None
    h1_error_bound: float | None = None
    tail_mass_bound: float | None = None
    tail_energy_bound: float | None = None
    evidence_kind: str = "empirical"
    source: str = "unspecified"

    def __post_init__(self):
        for name in (
            "electron_error_bound",
            "tw_error_bound",
            "h1_error_bound",
            "tail_mass_bound",
            "tail_energy_bound",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.evidence_kind not in ("empirical", "conditional_bound"):
            raise ValueError("quadrature evidence is empirical or a supplied conditional_bound")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("quadrature evidence needs a source")

    @property
    def continuum_certified(self) -> bool:
        return False

    def scaled(self, factor: float) -> "QuadratureCertificate":
        def energy(value):
            return None if value is None else factor**2 * value

        return replace(
            self,
            tw_error_bound=energy(self.tw_error_bound),
            tail_energy_bound=energy(self.tail_energy_bound),
            h1_error_bound=None
            if self.h1_error_bound is None
            else max(1.0, factor) * self.h1_error_bound,
        )


QuadratureError = QuadratureCertificate


@dataclass(frozen=True)
class DensityInput:
    coords: torch.Tensor
    weights: torch.Tensor
    rho: torch.Tensor
    grad_u: torch.Tensor
    expected_electrons: float | None = None
    mass_tolerance: float = 2e-3
    quadrature: QuadratureCertificate | None = None
    gradient_source: str = "density_input"
    gradient_operator: torch.Tensor | None = None
    units: str = "atomic"

    def __post_init__(self):
        for name in ("coords", "weights", "rho", "grad_u"):
            value = torch.as_tensor(getattr(self, name))
            if not value.is_floating_point():
                raise ValueError(f"{name} must be a real floating-point tensor")
            object.__setattr__(self, name, value)
        if self.gradient_operator is not None:
            object.__setattr__(self, "gradient_operator", torch.as_tensor(self.gradient_operator))
        self.validate()

    def validate(self) -> "DensityInput":
        if self.units != "atomic":
            raise ValueError(
                "density input requires atomic units: Bohr, Bohr^3, electrons/Bohr^3, -Delta"
            )
        if not math.isfinite(self.mass_tolerance) or self.mass_tolerance < 0:
            raise ValueError("mass tolerance must be finite and nonnegative")
        g = self.rho.numel()
        if (
            self.rho.ndim != 1
            or g == 0
            or self.coords.shape != (g, 3)
            or self.weights.shape != (g,)
            or self.grad_u.shape != (g, 3)
        ):
            raise ValueError("density shapes must be coords[G,3], weights[G], rho[G], grad_u[G,3]")
        if len({value.device for value in (self.coords, self.weights, self.rho, self.grad_u)}) != 1:
            raise ValueError("density tensors must share a device")
        if len({value.dtype for value in (self.coords, self.weights, self.rho, self.grad_u)}) != 1:
            raise ValueError("density tensors must share a floating dtype")
        if not all(
            torch.isfinite(value).all()
            for value in (self.coords, self.weights, self.rho, self.grad_u)
        ):
            raise ValueError("density, coordinates, weights and density gradient must be finite")
        if torch.any(self.rho < 0) or torch.any(self.weights <= 0):
            raise ValueError("density must be nonnegative and quadrature weights positive")
        if (
            self.n.item() <= 0
            or not torch.isfinite(self.tw)
            or not torch.isfinite(self.energy_mass)
        ):
            raise ValueError(
                "density must have positive mass and finite H1/critical energy samples"
            )
        if self.expected_electrons is not None:
            expected = float(self.expected_electrons)
            if not math.isfinite(expected) or expected <= 0:
                raise ValueError("expected electron number must be finite and positive")
            bound = self.mass_tolerance * max(1.0, expected)
            if abs(self.n.detach().item() - expected) > bound:
                raise ValueError(
                    "sampled density integral inconsistent with expected electron number"
                )
        if not isinstance(self.gradient_source, str) or not self.gradient_source:
            raise ValueError("density gradient needs source provenance")
        if self.quadrature is not None and not isinstance(self.quadrature, QuadratureCertificate):
            raise ValueError("quadrature must be explicit error-budget metadata")
        if self.gradient_operator is not None:
            op = self.gradient_operator
            if (
                op.shape != (g, 3, g)
                or op.dtype != self.rho.dtype
                or op.device != self.rho.device
                or not torch.isfinite(op).all()
            ):
                raise ValueError(
                    "gradient_operator must be finite [G,3,G], sharing density dtype/device"
                )
            reconstructed = torch.einsum("gai,i->ga", op, torch.sqrt(self.rho))
            if not torch.allclose(reconstructed, self.grad_u, rtol=1e-9, atol=1e-11):
                raise ValueError("grad_u is inconsistent with the supplied density representation")
        return self

    @property
    def n(self) -> torch.Tensor:
        return torch.sum(self.rho * self.weights)

    @property
    def tw(self) -> torch.Tensor:
        return torch.sum(self.weights * self.grad_u.square().sum(-1))

    @property
    def energy_mass(self) -> torch.Tensor:
        return torch.sum(self.weights * self.rho.pow(5.0 / 3.0))

    @property
    def grad_sqrt_rho(self) -> torch.Tensor:
        return self.grad_u

    @property
    def strong_h1_certified(self) -> bool:
        return False

    @classmethod
    def from_density_values(
        cls, coords, weights, rho, gradient_operator, **kwargs
    ) -> "DensityInput":
        rho = torch.as_tensor(rho)
        operator = torch.as_tensor(gradient_operator, dtype=rho.dtype, device=rho.device)
        if rho.ndim != 1 or torch.any(rho < 0):
            raise ValueError("density values must be nonnegative [G]")
        if operator.shape != (rho.numel(), 3, rho.numel()):
            raise ValueError("gradient_operator must have shape [G,3,G]")
        grad_u = torch.einsum("gai,i->ga", operator, torch.sqrt(rho))
        kwargs.setdefault("gradient_source", "discrete_density_representation")
        return cls(coords, weights, rho, grad_u, gradient_operator=operator, **kwargs)

    @classmethod
    def from_representation(
        cls, representation: Callable, coords, weights, **kwargs
    ) -> "DensityInput":
        """Evaluate a pointwise differentiable density without orbital input.

        The callable must not couple different grid points. Spatial autograd
        gradients do not define a full density-potential representation; use
        from_density_values for the derivative of a discrete density functional.
        """
        points = torch.as_tensor(coords)
        if not points.requires_grad:
            points = points.detach().clone().requires_grad_(True)
        rho = representation(points)
        if not torch.is_tensor(rho) or rho.shape != (points.shape[0],) or torch.any(rho <= 0):
            raise ValueError("pointwise differentiable representation must return positive rho[G]")
        grad_u = torch.autograd.grad(torch.sqrt(rho).sum(), points, create_graph=True)[0]
        kwargs.setdefault("gradient_source", "pointwise_density_representation")
        return cls(points, weights, rho, grad_u, **kwargs)

    def with_rho(self, rho: torch.Tensor) -> "DensityInput":
        if self.gradient_operator is None:
            raise ValueError(
                "a density-derived gradient_operator is required; holding grad_u fixed is incorrect"
            )
        return self.from_density_values(
            self.coords,
            self.weights,
            rho,
            self.gradient_operator,
            expected_electrons=self.expected_electrons,
            mass_tolerance=self.mass_tolerance,
            units=self.units,
        )

    def scaled(self, factor: float) -> "DensityInput":
        factor = float(factor)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("coordinate scaling factor must be finite and positive")
        return replace(
            self,
            coords=self.coords / factor,
            weights=self.weights / factor**3,
            rho=self.rho * factor**3,
            grad_u=self.grad_u * factor**2.5,
            gradient_operator=None
            if self.gradient_operator is None
            else self.gradient_operator * factor,
            quadrature=None if self.quadrature is None else self.quadrature.scaled(factor),
        )


def functional_derivative(
    functional: Callable[[DensityInput], torch.Tensor],
    density: DensityInput,
    gauge: str | None = "density_mean",
    create_graph: bool = False,
) -> torch.Tensor:
    """Discrete dE=sum_g weights[g]*potential[g]*d_rho[g], up to N-gauge.

    This differentiates the supplied density representation, not an independent
    fixed grad_u array. It does not certify continuum potential accuracy.
    """
    if density.gradient_operator is None:
        raise ValueError("functional derivative requires a density-derived gradient_operator")
    if torch.any(density.rho <= 0):
        raise ValueError("density derivative requires the positive-density interior")
    rho = (
        density.rho
        if density.rho.requires_grad
        else density.rho.detach().clone().requires_grad_(True)
    )
    represented = density.with_rho(rho)
    energy = functional(represented)
    if not torch.is_tensor(energy) or energy.numel() != 1:
        raise ValueError("functional must return one scalar energy tensor")
    derivative = (
        torch.autograd.grad(energy.reshape(()), rho, create_graph=create_graph)[0] / density.weights
    )
    if gauge == "density_mean":
        derivative = derivative - torch.sum(density.weights * rho * derivative) / represented.n
    elif gauge not in (None, "none"):
        raise ValueError("gauge must be density_mean or none")
    if not torch.isfinite(derivative).all():
        raise ValueError("functional derivative is nonfinite")
    return derivative


def cartesian_gradient_operator(coords: torch.Tensor, max_points: int = 4096) -> torch.Tensor:
    """Second-order differences on a full uniform Cartesian grid, any ordering.

    O(G^2) memory is for small derivative checks, not a production decoder.
    """
    coords = torch.as_tensor(coords)
    if (
        coords.ndim != 2
        or coords.shape[1] != 3
        or not coords.is_floating_point()
        or not torch.isfinite(coords).all()
    ):
        raise ValueError("finite floating Cartesian coordinates [G,3] required")
    g = len(coords)
    if not 1 <= g <= max_points:
        raise ValueError("dense derivative operator restricted to small grids")
    axes = [torch.unique(coords[:, axis], sorted=True) for axis in range(3)]
    if any(len(axis) < 3 for axis in axes) or math.prod(len(axis) for axis in axes) != g:
        raise ValueError("complete Cartesian grid with at least three points per axis required")
    steps = [axis[1] - axis[0] for axis in axes]
    if any(
        not torch.allclose(axis[1:] - axis[:-1], step.expand(len(axis) - 1), rtol=1e-7, atol=1e-12)
        for axis, step in zip(axes, steps)
    ):
        raise ValueError("uniform Cartesian grid required")
    indices = [
        tuple(int(torch.searchsorted(axes[a], point[a])) for a in range(3)) for point in coords
    ]
    lookup = {index: i for i, index in enumerate(indices)}
    if len(lookup) != g:
        raise ValueError("duplicate Cartesian grid points")
    operator = coords.new_zeros((g, 3, g))
    for i, index in enumerate(indices):
        for axis in range(3):
            position = index[axis]
            size = len(axes[axis])
            if position == 0:
                stencil = ((0, -3.0), (1, 4.0), (2, -1.0))
            elif position == size - 1:
                stencil = ((position, 3.0), (position - 1, -4.0), (position - 2, 1.0))
            else:
                stencil = ((position - 1, -1.0), (position + 1, 1.0))
            for neighbor, coefficient in stencil:
                shifted = list(index)
                shifted[axis] = neighbor
                operator[i, axis, lookup[tuple(shifted)]] = coefficient / (2 * steps[axis])
    return operator


@dataclass(frozen=True)
class GaussianDensity:
    """Analytic synthetic density, never labeled a Coulomb ground state."""

    electrons: float = 3.0
    exponent: float = 0.5
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if (
            not math.isfinite(self.electrons)
            or self.electrons <= 0
            or not math.isfinite(self.exponent)
            or self.exponent <= 0
        ):
            raise ValueError("Gaussian mass and exponent must be finite and positive")
        if len(self.center) != 3 or not all(math.isfinite(value) for value in self.center):
            raise ValueError("Gaussian center must be finite [3]")

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        center = coords.new_tensor(self.center)
        radius2 = (coords - center).square().sum(-1)
        return (
            self.electrons * (self.exponent / math.pi) ** 1.5 * torch.exp(-self.exponent * radius2)
        )
