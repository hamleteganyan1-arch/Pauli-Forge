"""Public inference gate: valid geometry is not ground-state D_E membership."""

from dataclasses import dataclass

from .density import DensityInput
from .domain import FrozenDE, Geometry


@dataclass
class DomainResult:
    result: object
    geometry_verified: bool
    physical_ground_state_verified: bool = False
    density_domain: str = "D_trial"
    domain_version: str = ""
    domain_hash: str = ""
    status: str = "EMPIRICAL"


class ValidatedInference:
    def __init__(self, model, domain: FrozenDE, mass_tolerance: float = 2e-3):
        self.model, self.domain, self.mass_tolerance = model, domain, mass_tolerance

    def __call__(self, geometry: Geometry, coords, weights=None, rho=None, grad_sqrt_rho=None):
        expected = self.domain.validate(geometry)
        if isinstance(coords, DensityInput):
            if any(value is not None for value in (weights, rho, grad_sqrt_rho)):
                raise ValueError("pass DensityInput or raw tensors, not both")
            density = coords
            density.validate()
        else:
            density = DensityInput(
                coords,
                weights,
                rho,
                grad_sqrt_rho,
                expected_electrons=expected,
                mass_tolerance=self.mass_tolerance,
            )
        if abs(density.n.detach().item() - expected) > self.mass_tolerance * max(1, expected):
            raise ValueError("sampled density integral inconsistent with allowed geometry")
        if hasattr(self.model, "forward_density"):
            result = self.model.forward_density(density)
        else:
            result = self.model(density.coords, density.weights, density.rho, density.grad_u)
        return DomainResult(
            result=result,
            geometry_verified=True,
            domain_version=self.domain.version,
            domain_hash=self.domain.domain_hash,
        )
