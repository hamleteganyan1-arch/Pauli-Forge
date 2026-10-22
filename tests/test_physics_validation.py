import math

import pytest
import torch

from pauliforge.physics import (
    C_TF_MINUS_DELTA,
    LTConstantEvidence,
    analytic_pauli_lower_bound,
    electron_count,
    von_weizsaecker,
)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_density_integrals_reject_invalid_values(bad):
    with pytest.raises(ValueError):
        electron_count(torch.tensor([1.0, bad]), torch.ones(2))


def test_gradient_and_weights_require_finite_positive_data():
    with pytest.raises(ValueError):
        von_weizsaecker(torch.zeros((2, 3)), torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError):
        von_weizsaecker(torch.full((2, 3), math.nan), torch.ones(2))


def test_semiclassical_coefficient_and_boolean_are_not_theorem_evidence():
    rho, weights, tw = torch.ones(3), torch.ones(3), torch.tensor(1.0)
    with pytest.raises(ValueError, match="structured"):
        analytic_pauli_lower_bound(rho, weights, tw, C_TF_MINUS_DELTA, True)
    assert analytic_pauli_lower_bound(rho, weights, tw).item() == 0.0
    with pytest.raises(ValueError, match="conventions"):
        LTConstantEvidence(0.1, "test-only hypothetical theorem", "-Delta/2", 2, True)


def test_explicit_external_hypotheses_remain_conditional():
    # This tests evidence plumbing, not the validity of an LT theorem.
    evidence = LTConstantEvidence(0.1, "test-only conditional hypothesis", assumptions_checked=True)
    result = analytic_pauli_lower_bound(
        torch.ones(3), torch.ones(3), torch.tensor(0.1), 0.1, evidence
    )
    assert result.item() == pytest.approx(0.2)
