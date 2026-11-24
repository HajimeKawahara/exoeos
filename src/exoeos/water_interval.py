"""Outward bounds for the two-power water quadrature's mathematical error.

This post-evaluation tool leaves the material scalar and its JAX derivatives
unchanged. It does not enclose pressure inversion or backend roundoff.
"""

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np

from . import _water_isotherm
from .ma_interval import _Interval, _interval


@dataclass(frozen=True)
class WaterQuadratureBound:
    """Dimensionless integral-error upper bound at a common driving pressure."""

    integral_error_upper: float
    finite_error_upper: float
    continuous_tail_upper: float
    quadrature_tail_upper: float
    moment_error_upper: tuple
    water_fraction_max: float
    series_orders: int
    transform_scale: float
    transformed_exponents: tuple


def _positive(value):
    """Intersect an interval with the known nonnegative range."""
    if not np.isfinite(value.lo) or not np.isfinite(value.hi) or value.hi < 0:
        raise ValueError("The water bound left the finite nonnegative domain.")
    return _Interval(max(0., value.lo), value.hi)


def _sum(values):
    result = _interval(0.)
    for value in values:
        result = result + value
    return result


def _exp_point_nonpositive(value):
    """Enclose exp(value) without a libm transcendental-error assumption."""
    if not math.isfinite(value) or value > 0:
        raise ValueError("Require a finite nonpositive exponential argument.")
    if value == 0:
        return _interval(1.)
    # Exact binary range reduction, followed by a positive Taylor series.
    # For x <= 1/8, the remaining terms are at most next/(1-x/22).
    squarings = max(0, math.frexp(-value)[1] + 3)
    reduced = _interval(-value) / math.ldexp(1., squarings)
    term, total = _interval(1.), _interval(1.)
    for order in range(1, 21):
        term = term * reduced / order
        total = total + term
    next_term = term * reduced / 21
    remainder = next_term / (1 - reduced / 22)
    positive_exp = total + _Interval(0., remainder.hi)
    result = _positive(positive_exp.reciprocal())
    for _ in range(squarings):
        result = _positive(result * result)
    return result


def _power_unit_interval(base, exponent):
    """Enclose base**exponent for exact binary64 base in (0, 1)."""
    argument = _interval(base).log() * exponent
    if argument.hi > 0:
        # Exact exponent zero is common when the two powers coincide.
        if exponent == 0:
            return _interval(1.)
        raise ValueError("Require a nonnegative transformed exponent.")
    return _Interval(_exp_point_nonpositive(argument.lo).lo,
                     _exp_point_nonpositive(argument.hi).hi)


def two_power_water_quadrature_bound(exponents, *, water_fraction_max=.5,
                                     series_orders=64):
    """Bound the stored 64-node rule uniformly over both capacity mixtures.

    Let ``t_i=C_i*p**a_i >= 0`` and ``sum(t_i) <= water_fraction_max`` at
    one common driving pressure. This bounds the absolute difference between
    the exact continuous integral and the real-arithmetic quadrature using
    the provider's stored binary64 nodes, weights, scale and transformed
    exponents. The capacity split is unrestricted, including either zero
    term as a limit. The result is dimensionless; multiply it outwards by
    ``S = dry_mass / water_molar_mass`` for the extensive Gibbs/RT error.

    Every geometric-series moment through ``series_orders`` is enclosed,
    and both nonnegative tails are bounded. Arithmetic is widened outwards;
    logarithms and exponentials use convergent series with explicit tails.
    No high-precision numerical agreement is used as an error certificate.

    This is a quadrature-error bound, not a bound on a complete JAX scalar
    evaluation. A caller must separately bound driving-pressure inversion,
    computed capacities, backend operations and any derivatives used in a
    phase proof. In particular ``water_fraction_max`` bounds the actual
    positive-term sum at that pressure, not merely the requested h/(S+h).
    Global minimization over dry composition remains a separate problem.
    """
    raw = np.asarray(exponents)
    if not np.isrealobj(raw) or raw.shape != (2,):
        raise ValueError("Require two real water exponents.")
    a = raw.astype(float)
    limit = np.asarray(water_fraction_max)
    if (not np.isrealobj(limit) or limit.ndim != 0
            or not np.all(np.isfinite(a)) or np.any(a < .25) or np.any(a > 1)
            or not np.isfinite(limit) or not 0 <= float(limit) <= .5
            or isinstance(series_orders, bool) or not isinstance(series_orders, Integral)
            or series_orders < 1):
        raise ValueError("Require 0.25<=a<=1, 0<=water_fraction_max<=0.5 and positive integer orders.")
    a = np.sort(a)
    w = _interval(float(limit))
    scale = float(8 / a[0])
    beta = scale * a
    u, q = np.asarray(_water_isotherm._U), np.asarray(_water_isotherm._Q)
    if (not np.all(np.isfinite(np.r_[u, q])) or np.any(u <= 0)
            or np.any(u >= 1) or np.any(q <= 0)):
        raise ValueError("Require finite positive weights and nodes strictly inside (0, 1).")
    # Recurrence retains intervals for every moment. It does not replace
    # the stored rule by an ideal exact-coefficient Gauss rule.
    advance = [_power_unit_interval(float(node), float(beta[0])) for node in u]
    ratio_exponent = _interval(float(beta[1])) - float(beta[0])
    ratio = []
    for node in u:
        if beta[0] == beta[1]:
            ratio.append(_interval(1.))
        else:
            lo = _power_unit_interval(float(node), ratio_exponent.hi).lo
            hi = _power_unit_interval(float(node), ratio_exponent.lo).hi
            ratio.append(_Interval(lo, hi))
    base = [_interval(float(weight)) / float(node) for weight, node in zip(q, u)]
    errors = []
    finite, w_power = _interval(0.), _interval(1.)
    for order in range(1, int(series_orders) + 1):
        base = [_positive(x * y) for x, y in zip(base, advance)]
        terms = list(base)
        worst = 0.
        for high_count in range(order + 1):
            quadrature = scale * _sum(terms)
            continuous = 1 / (high_count * _interval(a[1])
                              + (order - high_count) * _interval(a[0]))
            error = quadrature - continuous
            worst = max(worst, abs(error.lo), abs(error.hi))
            terms = [_positive(x * y) for x, y in zip(terms, ratio)]
        errors.append(worst)
        w_power = _positive(w_power * w)
        finite = _positive(finite + worst * w_power)
    next_power = _positive(w_power * w)
    continuous_tail = _positive(next_power / ((int(series_orders) + 1) * _interval(a[0]) * (1 - w)))
    tail_moment = _sum(_positive(x * y) for x, y in zip(base, advance))
    quadrature_tail = _positive(scale * next_power * tail_moment / (1 - w))
    total = _positive(finite + continuous_tail + quadrature_tail)
    return WaterQuadratureBound(total.hi, finite.hi, continuous_tail.hi,
        quadrature_tail.hi, tuple(errors), float(limit), int(series_orders),
        scale, tuple(map(float, beta)))
