"""Certified rational Fourier moment enclosures for a narrow polynomial class.

No quadrature is used. Nonnegative monomial coefficients with positive constant
term give explicit positivity and global log-Lipschitz bounds on [0,1]^3.
The exact mass and directed rational arithmetic supply the moment-input branch
of the Gram source; they do not enclose arbitrary sampled physical densities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral


@dataclass(frozen=True)
class RationalInterval:
    lower: Fraction
    upper: Fraction

    def __post_init__(self):
        object.__setattr__(self, "lower", Fraction(self.lower))
        object.__setattr__(self, "upper", Fraction(self.upper))
        if self.lower > self.upper:
            raise ValueError("reversed rational interval")

    @staticmethod
    def point(value):
        return RationalInterval(Fraction(value), Fraction(value))

    def __add__(self, other):
        other = other if isinstance(other, RationalInterval) else self.point(other)
        return RationalInterval(self.lower + other.lower, self.upper + other.upper)

    __radd__ = __add__

    def __neg__(self):
        return RationalInterval(-self.upper, -self.lower)

    def __sub__(self, other):
        return self + -(other if isinstance(other, RationalInterval) else self.point(other))

    def __mul__(self, other):
        other = other if isinstance(other, RationalInterval) else self.point(other)
        values = [a * b for a in (self.lower, self.upper) for b in (other.lower, other.upper)]
        return RationalInterval(min(values), max(values))

    __rmul__ = __mul__

    def reciprocal(self):
        if self.lower <= 0 <= self.upper:
            raise ValueError("interval division crosses zero")
        return RationalInterval(1 / self.upper, 1 / self.lower)


@dataclass(frozen=True)
class ComplexRationalInterval:
    real: RationalInterval
    imag: RationalInterval

    @staticmethod
    def point(value):
        return ComplexRationalInterval(RationalInterval.point(value), RationalInterval.point(0))

    def __add__(self, other):
        other = other if isinstance(other, ComplexRationalInterval) else self.point(other)
        return ComplexRationalInterval(self.real + other.real, self.imag + other.imag)

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, ComplexRationalInterval) else self.point(other)
        return ComplexRationalInterval(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    __rmul__ = __mul__

    def midpoint(self):
        return complex(
            float((self.real.lower + self.real.upper) / 2),
            float((self.imag.lower + self.imag.upper) / 2),
        )

    def absolute_error(self, approximation=None):
        approximation = self.midpoint() if approximation is None else complex(approximation)
        real_error = max(
            abs(Fraction.from_float(approximation.real) - v)
            for v in (self.real.lower, self.real.upper)
        )
        imag_error = max(
            abs(Fraction.from_float(approximation.imag) - v)
            for v in (self.imag.lower, self.imag.upper)
        )
        # l1 bounds complex absolute error, avoiding a floating sqrt certificate.
        return real_error + imag_error


def pi_interval(bits=100):
    """Machin identity and alternating-series remainder, exact Fractions."""
    if not isinstance(bits, int) or bits < 16:
        raise ValueError("at least 16 enclosure bits required")
    target = Fraction(1, 2 ** (bits + 8))

    def atan_inverse(denominator):
        total, index = Fraction(), 0
        while True:
            term = Fraction(1, (2 * index + 1) * denominator ** (2 * index + 1))
            signed = term if index % 2 == 0 else -term
            total += signed
            index += 1
            next_term = Fraction(1, (2 * index + 1) * denominator ** (2 * index + 1))
            if next_term <= target:
                next_value = total + (next_term if index % 2 == 0 else -next_term)
                return RationalInterval(min(total, next_value), max(total, next_value))

    return 16 * atan_inverse(5) - 4 * atan_inverse(239)


class PolynomialMomentOracle:
    """Exact known polynomial density/window, never inferred from nodal values.

    Mapping keys are exponent triples; values are exact nonnegative rational
    coefficients. Positive constant coefficient implies positive rho and a.
    """

    def __init__(self, density_coefficients, window_coefficients=None, *, bits=100):
        self.density = self._coefficients(density_coefficients)
        self.window = self._coefficients(window_coefficients or {(0, 0, 0): 1})
        self.pi = pi_interval(bits)
        self.bits = bits
        self.weighted = {}
        for left, a in self.density.items():
            for right, b in self.window.items():
                key = tuple(i + j for i, j in zip(left, right))
                self.weighted[key] = self.weighted.get(key, Fraction()) + a * b
        self.mass_exact = sum(
            (a * math.prod(Fraction(1, j + 1) for j in key) for key, a in self.density.items()),
            Fraction(),
        )
        self._cache = {}

    @staticmethod
    def _coefficients(values):
        result = {}
        for key, value in values.items():
            if len(key) != 3 or any(
                isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in key
            ):
                raise ValueError("polynomial exponents must be nonnegative integer triples")
            if isinstance(value, float) or isinstance(value, bool):
                raise ValueError("polynomial coefficients must be exact rationals")
            result[tuple(key)] = Fraction(value)
        if result.get((0, 0, 0), 0) <= 0 or any(value < 0 for value in result.values()):
            raise ValueError(
                "certified branch requires nonnegative coefficients and positive constant term"
            )
        return result

    def _monomial(self, exponent, frequency):
        if frequency == 0:
            return ComplexRationalInterval.point(Fraction(1, exponent + 1))
        reciprocal = (self.pi * frequency).reciprocal()
        # 1/(i*pi*m)=-i/(pi*m); exp(i*pi*m)=(-1)^m exactly.
        inverse_ia = ComplexRationalInterval(RationalInterval.point(0), -reciprocal)
        endpoint = -1 if frequency % 2 else 1
        value = inverse_ia * (endpoint - 1)
        for degree in range(1, exponent + 1):
            value = inverse_ia * endpoint + inverse_ia * value * (-degree)
        return value

    def enclosure(self, frequency, *, weighted=False):
        frequency = tuple(int(i) if isinstance(i, Integral) else i for i in frequency)
        if len(frequency) != 3 or any(
            isinstance(i, bool) or not isinstance(i, Integral) for i in frequency
        ):
            raise ValueError("integer frequency triple required")
        key = frequency, weighted
        if key not in self._cache:
            result = ComplexRationalInterval.point(0)
            for exponent, coefficient in (self.weighted if weighted else self.density).items():
                term = ComplexRationalInterval.point(coefficient)
                for power, mode in zip(exponent, frequency):
                    term = term * self._monomial(power, mode)
                result = result + term
            self._cache[key] = result
        return self._cache[key]

    def __call__(self, frequency):
        return self.enclosure(frequency).midpoint()

    def weighted_moment(self, frequency):
        return self.enclosure(frequency, weighted=True).midpoint()

    def regularity(self):
        from .gram_sdp import RegularityAssumptions

        derivative_bound = sum(value * sum(key) for key, value in self.density.items())
        loglip = derivative_bound / self.density[(0, 0, 0)]
        window_lip = sum(value * sum(key) for key, value in self.window.items())

        def up(value):
            return math.nextafter(float(value), math.inf)

        return RegularityAssumptions(
            up(self.mass_exact),
            up(loglip),
            math.nextafter(float(self.window[(0, 0, 0)]), -math.inf),
            up(sum(self.window.values())),
            up(window_lip),
            "exact_positive_polynomial_coefficients",
            True,
        )

    def solve(self, frequencies, **kwargs):
        """Analytically enclosed moments; solver/truncation remain numerical."""
        import numpy as np

        from .gram_sdp import _frequencies, solve_gram_sdp

        frequencies = _frequencies(frequencies)
        g_error, e_error = Fraction(), Fraction()
        for left in frequencies:
            for right in frequencies:
                difference = tuple(int(v) for v in right - left)
                g_error = max(g_error, self.enclosure(difference).absolute_error())
                weighted = self.enclosure(difference, weighted=True)
                exact_e = weighted * ComplexRationalInterval(
                    self.pi * self.pi * int(np.dot(left, right)), RationalInterval.point(0)
                )
                numerical_e = math.pi**2 * int(np.dot(left, right)) * weighted.midpoint()
                e_error = max(e_error, exact_e.absolute_error(numerical_e))
        dimension = len(frequencies)
        g_bound = math.nextafter(float(dimension * g_error), math.inf)
        e_bound = math.nextafter(float(dimension * e_error), math.inf)
        return solve_gram_sdp(
            frequencies,
            self,
            self.weighted_moment,
            self.regularity(),
            moment_operator_error=g_bound,
            objective_error=e_bound,
            **kwargs,
        )
