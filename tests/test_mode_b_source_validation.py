import pytest
import torch

from pauliforge.density import DensityInput
from pauliforge.model import DensityXCModel
from pauliforge.modes import AnalyticalPauliXCFunctional


@pytest.mark.parametrize("defect", ["cap", "negative_kinetic"])
def test_mode_b_rechecks_supplied_cap_and_kinetic_operator(defect):
    density = DensityInput(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        torch.full((2,), 1.5, dtype=torch.float64),
        torch.zeros(2, 3, dtype=torch.float64),
    )
    teacher = {
        "numerical_admissible": True,
        "density_target": density.rho,
        "electrons": 3.0,
        "ts_minus_delta": 1.2,
        "gamma": torch.eye(2, dtype=torch.float64) * 1.5,
        "ao_values": torch.eye(2, dtype=torch.float64),
        "kinetic": torch.eye(2, dtype=torch.float64) * 0.4,
        "weights": density.weights,
    }
    if defect == "cap":
        teacher["max_occupation"] = 1.0
        reason = "cap"
    else:
        teacher["kinetic"] = torch.diag(torch.tensor([-0.1, 0.9], dtype=torch.float64))
        reason = "positive semidefinite"
    with pytest.raises(ValueError, match=reason):
        AnalyticalPauliXCFunctional(DensityXCModel().double())(density, teacher, 1.0, -2.0)
