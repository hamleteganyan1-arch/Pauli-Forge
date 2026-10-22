from fractions import Fraction

import pytest
import torch

from pauliforge.atlas import (
    WINDOW_DILATION,
    AdaptiveAtlas,
    AtlasResolutionError,
    DyadicCube,
    IntegralEstimate,
    PauliBlock,
)


def test_dyadic_geometry_and_strong_balance_conserve_volume():
    root = DyadicCube()
    atlas = AdaptiveAtlas([root, DyadicCube(index=(1, 0, 0))])
    # A narrow fine region touches the unchanged neighboring root.
    leaves = set(root.children())
    chosen = DyadicCube(1, (1, 0, 0))
    leaves.remove(chosen)
    leaves.update(chosen.children())
    next_cube = DyadicCube(2, (3, 0, 0))
    leaves.remove(next_cube)
    leaves.update(next_cube.children())
    atlas.leaves = tuple(leaves | {DyadicCube(index=(1, 0, 0))})
    assert not atlas.is_strongly_balanced()
    volume = sum((q.side**3 for q in atlas.leaves), Fraction())
    atlas.balance()
    assert atlas.is_strongly_balanced()
    assert sum((q.side**3 for q in atlas.leaves), Fraction()) == volume
    assert not atlas.continuum_certified


def test_source_robust_thresholds_and_active_superset():
    def oracle(cube, dilation, field):
        if field == "kinetic":
            return IntegralEstimate(0, 0)
        if dilation == 32:
            return IntegralEstimate(5 if cube.level == 0 else 1, 0.1)
        return IntegralEstimate(2.05 - 0.124, 0.125)

    atlas = AdaptiveAtlas([DyadicCube()]).build(oracle)
    assert len(atlas.leaves) == 8
    assert len(atlas.active_blocks) == 8
    assert all(
        block.alpha == (WINDOW_DILATION * block.cube.side) ** -2 for block in atlas.active_blocks
    )
    assert all(block.full_diagonal for block in atlas.active_blocks)
    assert atlas.to_dict()["complete_space"] is False


def test_unknown_tail_never_makes_global_certificate():
    atlas = AdaptiveAtlas([DyadicCube()]).build(lambda *args: IntegralEstimate(0, 0, rigorous=True))
    assert not atlas.complete_space
    assert not atlas.continuum_certified
    assert "certified_global_H1_name_and_tail" in atlas.to_dict()["missing"]


def test_refinement_resource_limit_is_explicit_failure():
    atlas = AdaptiveAtlas([DyadicCube()], max_depth=1)
    with pytest.raises(AtlasResolutionError):
        atlas.build(lambda *args: IntegralEstimate(10, 0))


def test_oracle_error_must_close_source_comparison_budget():
    atlas = AdaptiveAtlas([DyadicCube()])
    with pytest.raises(ValueError, match="error"):
        atlas.build(lambda *args: IntegralEstimate(0, 1))


def test_windows_partition_derivative_and_full_diagonal_density():
    atlas = AdaptiveAtlas([DyadicCube()])
    atlas.leaves = DyadicCube().children()
    coords = torch.tensor(
        [[0.48, 0.49, 0.50], [0.2, 0.3, 0.4]], dtype=torch.float64, requires_grad=True
    )
    chi, gradient = atlas.normalized_windows(coords)
    assert torch.allclose(chi.square().sum(1), torch.ones(2, dtype=torch.float64), atol=1e-14)
    assert torch.allclose(
        (chi[..., None] * gradient).sum(1), torch.zeros((2, 3), dtype=torch.float64), atol=1e-13
    )
    automatic = torch.autograd.grad(chi[:, 0].sum(), coords)[0]
    assert torch.allclose(automatic, gradient[:, 0], atol=1e-12)
    cube = DyadicCube()
    block = PauliBlock(cube, IntegralEstimate(3, 0.01), WINDOW_DILATION**-2)
    rho = torch.tensor([5.0, 7.0], dtype=torch.float64)
    _, restricted = block.restrict_density(coords, rho)
    assert torch.equal(restricted, rho)
    with pytest.raises(ValueError, match="full"):
        PauliBlock(cube, IntegralEstimate(3, 0), WINDOW_DILATION**-2, full_diagonal=False)


def test_uncovered_points_raise_instead_of_silently_ignoring_density():
    atlas = AdaptiveAtlas([DyadicCube()])
    with pytest.raises(ValueError, match="cover"):
        atlas.normalized_windows(torch.tensor([[20.0, 0.0, 0.0]], dtype=torch.float64))
