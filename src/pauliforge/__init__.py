"""Density-only research library; no universal continuum guarantee is asserted."""

from .density import DensityInput, QuadratureCertificate, functional_derivative
from .domain import FrozenDE, Geometry
from .evaluation import ValidatedInference
from .matrix import check_pauli_matrix, project_occupations
from .model import DensityXCModel, PauliKEDF
from .physics import electron_count, von_weizsaecker

__version__ = "1.0.0"

__all__ = [
    "DensityInput",
    "QuadratureCertificate",
    "functional_derivative",
    "FrozenDE",
    "Geometry",
    "PauliKEDF",
    "DensityXCModel",
    "ValidatedInference",
    "check_pauli_matrix",
    "project_occupations",
    "von_weizsaecker",
    "electron_count",
]
