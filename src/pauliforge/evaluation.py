"""Domain-gated prediction: membership in K geometries, not proof of ground state."""
from dataclasses import dataclass
from .domain import FrozenDE, Geometry
from .model import PauliKEDF

@dataclass
class DomainResult:
    result: object
    geometry_verified: bool
    physical_ground_state_verified: bool = False  # not decidable by this gate

class ValidatedInference:
    def __init__(self, model: PauliKEDF, domain: FrozenDE, mass_tolerance: float = 2e-3):
        self.model, self.domain, self.mass_tolerance = model, domain, mass_tolerance

    def __call__(self, geometry: Geometry, coords, weights, rho, grad_sqrt_rho):
        n_expected = self.domain.validate(geometry)
        result = self.model(coords, weights, rho, grad_sqrt_rho)
        if abs(float(result.n.detach()) - n_expected) > self.mass_tolerance * max(1,n_expected):
            raise ValueError("sampled density integral inconsistent with allowed geometry")
        return DomainResult(result=result, geometry_verified=True)
