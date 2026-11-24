"""Independent moment references and scope checks for the water value bound."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from exoeos import _water_isotherm
from exoeos.water_interval import (
    _exp_point_nonpositive, two_power_water_quadrature_bound,
)


@pytest.fixture(scope="module")
def bound():
    return two_power_water_quadrature_bound([.807805, .449387])


@pytest.mark.parametrize("value", [0., -1e-12, -.125, -1., -40., -700., -10000.])
def test_exponential_enclosure_contains_independent_decimal_reference(value):
    actual = _exp_point_nonpositive(value)
    with localcontext() as context:
        context.prec = 100
        expected = Decimal.from_float(value).exp()
        assert Decimal.from_float(actual.lo) <= expected <= Decimal.from_float(actual.hi)


def test_mixed_fractional_moments_are_enclosed(bound):
    # Decimal references are a regression check, not the global-bound proof.
    with localcontext() as context:
        context.prec = 80
        d = Decimal.from_float
        a = list(map(d, [.449387, .807805]))
        beta = list(map(d, bound.transformed_exponents))
        for order in (1, 3, 16, 64):
            for high in (0, order // 2, order):
                power = high * beta[1] + (order - high) * beta[0] - 1
                quadrature = d(bound.transform_scale) * sum(
                    d(float(q)) * (d(float(u)).ln() * power).exp()
                    for u, q in zip(_water_isotherm._U, _water_isotherm._Q))
                integral = 1 / (high * a[1] + (order - high) * a[0])
                assert abs(quadrature - integral) <= d(bound.moment_error_upper[order - 1])


def test_s2_uniform_quadrature_budget_is_below_phase_tolerance(bound):
    assert 0 < bound.integral_error_upper < 1e-10
    # 18 exceeds the largest dry-component/water molar-mass ratio in S2.
    assert 18 * bound.integral_error_upper < 1e-8
    assert bound.integral_error_upper >= (
        bound.finite_error_upper + bound.continuous_tail_upper
        + bound.quadrature_tail_upper)


@pytest.mark.parametrize("split", [0., .1, .5, .99, 1.])
def test_complete_positive_series_bounds_capacity_mixtures(split, bound):
    with localcontext() as context:
        context.prec = 80
        d = Decimal.from_float
        a = list(map(d, [.449387, .807805]))
        beta = list(map(d, bound.transformed_exponents))
        t = [Decimal('.5') * d(1 - split), Decimal('.5') * d(split)]
        # The rounded split can slightly exceed one; renormalize exactly.
        t = [x * Decimal('.5') / sum(t) for x in t]
        integral, coefficients = Decimal(0), [Decimal(1)]
        for order in range(1, 321):
            updated = [Decimal(0)] * (order + 1)
            for k, coefficient in enumerate(coefficients):
                updated[k] += coefficient * t[0]
                updated[k + 1] += coefficient * t[1]
            coefficients = updated
            integral += sum(value / (k * a[1] + (order - k) * a[0])
                            for k, value in enumerate(coefficients))
        quadrature = Decimal(0)
        for u, q in zip(_water_isotherm._U, _water_isotherm._Q):
            node = d(float(u))
            at_node = sum(value * (node.ln() * power).exp() for value, power in zip(t, beta))
            quadrature += d(float(q)) * at_node / (node * (1 - at_node))
        quadrature *= d(bound.transform_scale)
        assert abs(quadrature - integral) < d(bound.integral_error_upper)


def test_stored_weight_changes_are_included(monkeypatch):
    # A defective rule must not receive an ideal-Gauss exactness assertion.
    monkeypatch.setattr(_water_isotherm, "_Q", _water_isotherm._Q * 1.01)
    actual = two_power_water_quadrature_bound([.5, .5], series_orders=32)
    with localcontext() as context:
        context.prec = 70
        expected = Decimal('.01') * Decimal(2).ln() / Decimal('.5')
        assert Decimal.from_float(actual.integral_error_upper) > expected
    assert actual.integral_error_upper > .01


@pytest.mark.parametrize("kwargs", [
    {"exponents": [0., .5]}, {"exponents": [1.01, .5]},
    {"exponents": [np.nan, .5]}, {"exponents": [.5]},
    {"exponents": [1j, .5]}, {"water_fraction_max": .5001},
    {"water_fraction_max": -1.}, {"water_fraction_max": np.nan},
    {"series_orders": 0}, {"series_orders": 1.5}, {"series_orders": True},
])
def test_unsupported_inputs_do_not_produce_a_certificate(kwargs):
    inputs = dict(exponents=[.5, .5])
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        two_power_water_quadrature_bound(**inputs)
