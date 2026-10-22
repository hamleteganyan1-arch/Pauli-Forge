"""Aggregated guaranteed bounds; no claim that the ML network itself certifies them."""
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class LocalInterval:
    lower: float
    upper: float
    alpha: float
    def __post_init__(self):
        if self.alpha < 0 or self.lower < -1e-10 or self.upper < self.lower:
            raise ValueError("invalid local Pauli interval")

@dataclass(frozen=True)
class GlobalInterval:
    lower: float
    upper: float
    width: float
    certified_at_epsilon_n: bool


def assemble(intervals: Sequence[LocalInterval], tw_lower: float,
             tw_upper: float, ims_upper: float, n: float, epsilon: float) -> GlobalInterval:
    """If 0 <= true P-sum residual <= ims_upper, and local intervals honest,
    then T_s lies within [TW^-+sum alpha L, TW^++sum alpha U+IMS].
    """
    if tw_lower < 0 or tw_upper < tw_lower or ims_upper < 0 or n <= 0 or epsilon <= 0:
        raise ValueError("invalid global input interval")
    lower = tw_lower + sum(a.alpha * a.lower for a in intervals)
    upper = tw_upper + sum(a.alpha * a.upper for a in intervals) + ims_upper
    width = upper - lower
    return GlobalInterval(lower, upper, width, width <= epsilon * n)
