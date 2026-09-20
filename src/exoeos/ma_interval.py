"""Conservative composition curvature for the existing Ma Fe-Si-O-H control.

This numerical bound describes the implemented scalar, not experimental
validity, pressure dependence, or missing alloy-H interactions.
"""

from dataclasses import dataclass
import math

import numpy as np

from exoeos.ma_fe_si_o import MaFeSiOLiquid
from exoeos.ma_fe_si_o_h import MaFeSiOHLiquid


@dataclass(frozen=True)
class _Interval:
    lo: float
    hi: float

    @staticmethod
    def rounded(lo, hi):
        return _Interval(np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf))

    def __add__(self, other):
        other = _interval(other)
        return self.rounded(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return _Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -_interval(other)

    def __rsub__(self, other):
        return _interval(other) + -self

    def __mul__(self, other):
        other = _interval(other)
        products = [a * b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return self.rounded(min(products), max(products))

    __rmul__ = __mul__

    def reciprocal(self):
        if self.lo <= 0 <= self.hi:
            raise ValueError("The composition box crosses a singular denominator.")
        return self.rounded(1 / self.hi, 1 / self.lo)

    def __truediv__(self, other):
        return self * _interval(other).reciprocal()

    def log(self):
        if self.lo <= 0:
            raise ValueError("The composition box crosses a logarithmic endpoint.")
        return _Interval(_log_point(self.lo).lo, _log_point(self.hi).hi)


def _interval(value):
    return value if isinstance(value, _Interval) else _Interval(float(value), float(value))


def _log_series(value):
    """Enclose log(value) for 1 <= value <= 2, with a geometric tail bound."""
    z = (value - 1) / (value + 1)
    squared, power, total = z * z, z, _interval(0.)
    for index in range(20):
        total = total + 2 * power / (2 * index + 1)
        power = power * squared
    radius = _interval(max(abs(z.lo), abs(z.hi)))
    tail = 2 * _interval(max(abs(power.lo), abs(power.hi))) / (41 * (1 - radius * radius))
    return total + _Interval(-tail.hi, tail.hi)


_LOG_TWO = _log_series(_interval(2.))


def _log_point(value):
    # frexp is exact for binary64 inputs. The convergent atanh series avoids
    # assuming any platform-specific libm logarithm error bound.
    mantissa, exponent = math.frexp(value)
    return _log_series(_interval(2 * mantissa)) + (exponent - 1) * _LOG_TWO


class _SecondOrder:
    """Three composition derivatives with interval-valued coefficients."""

    def __init__(self, value, index=None):
        self.value = _interval(value)
        self.gradient = [_interval(float(i == index)) for i in range(3)]
        self.hessian = [[_interval(0.) for _ in range(3)] for _ in range(3)]

    def __add__(self, other):
        other = _second(other)
        result = _SecondOrder(self.value + other.value)
        result.gradient = [a + b for a, b in zip(self.gradient, other.gradient)]
        result.hessian = [[self.hessian[i][j] + other.hessian[i][j]
                           for j in range(3)] for i in range(3)]
        return result

    __radd__ = __add__

    def __mul__(self, other):
        other = _second(other)
        result = _SecondOrder(self.value * other.value)
        result.gradient = [self.gradient[i] * other.value + self.value * other.gradient[i]
                           for i in range(3)]
        result.hessian = [[
            self.hessian[i][j] * other.value + self.gradient[i] * other.gradient[j]
            + self.gradient[j] * other.gradient[i] + self.value * other.hessian[i][j]
            for j in range(3)] for i in range(3)]
        return result

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -_second(other)

    def __rsub__(self, other):
        return _second(other) + -self

    def transform(self, value, first, second):
        result = _SecondOrder(value)
        result.gradient = [first * item for item in self.gradient]
        result.hessian = [[first * self.hessian[i][j]
                           + second * self.gradient[i] * self.gradient[j]
                           for j in range(3)] for i in range(3)]
        return result

    def reciprocal(self):
        inverse = self.value.reciprocal()
        return self.transform(inverse, -inverse * inverse, 2 * inverse * inverse * inverse)

    def __truediv__(self, other):
        return self * _second(other).reciprocal()

    def __rtruediv__(self, other):
        return _second(other) * self.reciprocal()

    def log(self):
        inverse = self.value.reciprocal()
        return self.transform(self.value.log(), inverse, -inverse * inverse)


def _second(value):
    return value if isinstance(value, _SecondOrder) else _SecondOrder(value)


def ma_alloy_curvature_lower_bound(
    model: MaFeSiOHLiquid, temperature_k: float, lower: np.ndarray, upper: np.ndarray,
) -> float:
    """Bound the ideal-plus-Ma molar Hessian below on a composition domain.

    Bounds follow Fe, Si, O, H order and intersect ``sum(x)=1``. Fe is
    eliminated, so the Hessian acts on (Si, O, H); fixed coordinates are
    removed. Standards are linear and contribute no curvature. The result
    bounds every eigenvalue in units of G/(RT) per mole of components.

    Interval differentiation encloses the Ma excess Hessian on the enclosing
    solute box. Add the ideal diagonal lower bounds ``1/upper[i]`` and omit
    the positive-semidefinite Fe ideal term. Gershgorin then gives a global
    lower bound. A nonnegative result certifies convexity for this mathematical
    model on the declared domain. A negative result is inconclusive, not proof
    of instability. No composition sampling is used to claim convexity.

    The enclosing box must stay away from pure-H and dry pure-solute
    singularities. Narrow the explicit domain if interval bounds cross them;
    this does not establish a physical calibration domain. Binary64 arithmetic
    is widened outwards. Logarithms use exact binary range reduction and a
    convergent series with an explicit geometric remainder bound.
    """
    if type(model) is not MaFeSiOHLiquid or type(model.dry_model) is not MaFeSiOLiquid:
        raise TypeError("This bound applies to the unmodified MaFeSiOHLiquid and MaFeSiOLiquid scalars only.")
    raw = tuple(map(np.asarray, (lower, upper, model.dry_model.interaction_K, temperature_k)))
    if not all(np.isrealobj(value) for value in raw) or raw[-1].ndim != 0:
        raise ValueError("The composition box, interactions, and temperature must be real.")
    lo, hi, coefficients = (value.astype(float) for value in raw[:3])
    if (not np.isfinite(temperature_k) or temperature_k <= 0
            or lo.shape != (4,) or hi.shape != (4,)
            or coefficients.shape != (3,)
            or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi))
            or not np.all(np.isfinite(coefficients))
            or np.any(lo < 0) or np.any(hi > 1) or np.any(lo > hi)
            or lo.sum() > 1 or hi.sum() < 1 or lo[0] <= 0):
        raise ValueError("Supply positive T and a feasible finite Fe-rich composition box.")
    silicon, oxygen, hydrogen = [_SecondOrder(_Interval(lo[i + 1], hi[i + 1]), i)
                                for i in range(3)]
    dry = 1 - hydrogen
    si, ox = silicon / dry, oxygen / dry
    a, b = 1 - si, 1 - ox
    product = si * ox
    pair = -product - si * b.log() - ox * a.log() + product * product * (1 / a + 1 / b - 1) / 2
    excess = dry * (coefficients[0] * a * a.log() + coefficients[1] * b * b.log()
                    + coefficients[2] * pair) / temperature_k
    free = [i for i in range(3) if hi[i + 1] > lo[i + 1]]
    if not free:
        return 0.
    bounds = []
    for i in free:
        row = excess.hessian[i][i] + _interval(hi[i + 1]).reciprocal()
        for j in free:
            if i != j:
                cross = excess.hessian[i][j]
                row = row - max(abs(cross.lo), abs(cross.hi))
        bounds.append(row.lo)
    bound = float(min(bounds))
    if not np.isfinite(bound):
        raise ValueError("The composition box does not provide a finite curvature bound.")
    return bound
