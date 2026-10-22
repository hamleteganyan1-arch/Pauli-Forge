"""A toy Gaussian density, NOT a physical Coulomb ground-state benchmark."""

import torch

from pauliforge.model import PauliKEDF

x = torch.linspace(-5, 5, 11, dtype=torch.float64)
coords = torch.cartesian_prod(x, x, x)
weights = torch.full((len(coords),), float((x[1] - x[0]) ** 3), dtype=torch.float64)
a = 0.5
rho0 = torch.exp(-a * coords.square().sum(-1))
rho = (3.0 / torch.sum(weights * rho0)) * rho0
u = torch.sqrt(rho)
grad_u = -a * coords * u[:, None]
m = PauliKEDF().double()
y = m(coords, weights, rho, grad_u)
print(
    {
        "N": float(y.n),
        "TW": float(y.tw),
        "Pauli_ML_untrained": float(y.pauli.detach()),
        "Ts_ML_untrained": float(y.ts.detach()),
        "label": "toy_untrained_not_exact",
    }
)
