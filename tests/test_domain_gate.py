import pytest
import torch
from pauliforge.domain import FrozenDE, Geometry
from pauliforge.model import PauliKEDF
from pauliforge.evaluation import ValidatedInference


def test_domain_gate_rejects_mismatch():
    xyz=torch.tensor([[0.,0.,0.],[1.,0.,0.]],dtype=torch.float64)
    geometry=Geometry((1,1),xyz)
    coords=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=torch.float64)
    w=torch.ones(3,dtype=torch.float64)
    rho=torch.ones(3,dtype=torch.float64)
    gu=torch.zeros_like(coords)
    gate=ValidatedInference(PauliKEDF().double(),FrozenDE())
    with pytest.raises(ValueError, match="integral"):
        gate(geometry, coords,w,rho,gu)
