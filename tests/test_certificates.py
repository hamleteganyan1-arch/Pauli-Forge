import math
from fractions import Fraction

import pytest

from pauliforge.certificates import (
    ErrorBudget,
    LocalInterval,
    assemble,
    bounded_decoder_error,
    histogram_l2_budget,
)
from pauliforge.decoders.packets import PolynomialPacket, decode_low_mass_packets


def test_small_numeric_gap_is_not_a_proof():
    result = assemble([LocalInterval(1, 1.01, 1)], 2, 2.01, 0.01, 3, 0.1)
    assert result.budget_satisfied
    assert not result.certified_at_epsilon_n
    assert result.status == "CONDITIONAL"
    assert "complete_full_diagonal_cover" in result.missing_assumptions


@pytest.mark.parametrize(
    "args",
    [(math.nan, 1, 1), (0, math.inf, 1), (0, 1, math.nan), (-1e-15, 1, 1), (1, 0, 1), (0, 1, -1)],
)
def test_invalid_intervals_are_rejected(args):
    with pytest.raises(ValueError):
        LocalInterval(*args)


def test_all_named_errors_are_included_and_missing_budget_blocks_proof():
    result = assemble(
        [LocalInterval(0, 1, 2)],
        3,
        3.1,
        0.2,
        4,
        1,
        errors=ErrorBudget(basis=0.3, tail=0.4, missing=("continuum_tail",)),
    )
    assert result.lower <= 2.3
    assert result.upper >= 6.0
    assert not result.certified_at_epsilon_n
    assert "continuum_tail" in result.missing_assumptions


def test_exact_disjoint_packet_certificate_is_positive_case():
    packets = [
        PolynomialPacket((0, 0, 0), (1, 1, 1), 1),
        PolynomialPacket((4, 0, 0), (1, 2, 1), Fraction(3, 2)),
    ]
    result = decode_low_mass_packets(packets)
    interval = result.global_interval()
    assert interval.certified_at_epsilon_n
    assert interval.lower <= float(result.tw_exact) <= interval.upper
    assert interval.scope == "continuum_explicit_packet_subclass"
    assert result.n_exact == Fraction(5, 2)
    # Exact P=0 does not require the total number to be <=2.
    assert result.pauli_exact == 0
    with pytest.raises(ValueError):
        decode_low_mass_packets([packets[0], packets[0]])


def test_forged_callback_cannot_certify():
    class Unchecked:
        scope = "continuum"

        def verify_global(self, *args):
            return True

    result = assemble([LocalInterval(0, 0, 1)], 1, 1, 0, 1, 0.1, proof=Unchecked())
    assert not result.certified_at_epsilon_n


def test_wrong_tw_bracket_invalidates_actual_packet_proof():
    decoded = decode_low_mass_packets([PolynomialPacket((0, 0, 0), (1, 1, 1), 1)])
    result = assemble(decoded.intervals, 0, 1, 0, 1, 10, proof=decoded.proof)
    assert result.budget_satisfied and not result.certified_at_epsilon_n


def test_aggregate_theorem_allows_one_bad_block_without_individual_cutoffs():
    # true [1,0], predictions [1.1,3], weights [.5,.1]; second is outside tau.
    actual = abs(0.5 * (1.1 - 1) + 0.1 * (3 - 0))
    bound = bounded_decoder_error(
        delta=0.2, weight_sum_bound=0.6, local_energy_upper=3, tau=0.5, aggregate_l2_squared=0.1
    )
    assert actual <= bound
    assert histogram_l2_budget(
        n_electrons=4,
        subdivisions=8,
        overlap=2,
        kinetic_per_electron=1,
        beta=0.001,
        blocks_per_electron=3,
    ) == pytest.approx((36 / 64 + 0.018) * 4)
    with pytest.raises(ValueError):
        bounded_decoder_error(
            delta=0.1, weight_sum_bound=1, local_energy_upper=1, tau=0, aggregate_l2_squared=1
        )
