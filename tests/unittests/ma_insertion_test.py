"""Independent high-precision controls for the fixed-plane insertion bound."""

from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos.ma_fe_si_o import MaFeSiOLiquid
from exoeos.ma_fe_si_o_h import MaFeSiOHLiquid
from exoeos.ma_interval import ma_alloy_insertion_lower_bound


LOWER = np.array([.86, 0., 0., 0.])
UPPER = np.array([1., .08, .02, .04])
POINT = np.array([.01, .01, .02])
WEIGHTS = np.array([.96, .01, .01, .02])
IDEAL = MaFeSiOHLiquid(MaFeSiOLiquid((0., 0., 0.)))


def decimal_value(value):
    return Decimal.from_float(float(value))


def ideal_minimum(standards, plane, fixed_h=None, offsets=None):
    with localcontext() as context:
        context.prec = 70
        costs = [decimal_value(s) - decimal_value(p) for s, p in zip(standards, plane)]
        if offsets is not None:
            costs = [cost + decimal_value(offset) for cost, offset in zip(costs, offsets)]
        if fixed_h is None:
            return -sum((-cost).exp() for cost in costs).ln()
        h = decimal_value(fixed_h)
        dry = 1 - h
        return dry * (dry / sum((-cost).exp() for cost in costs[:3])).ln() + h * (h.ln() + costs[3])


@pytest.mark.parametrize("offset", [-.125, 0., .125])
def test_ideal_global_minimum_against_decimal_partition_function(offset):
    standards, plane = -np.log(WEIGHTS), np.full(4, offset)
    actual = ideal_minimum(standards, plane)
    bound = ma_alloy_insertion_lower_bound(IDEAL, 2173.15, LOWER, UPPER, POINT, standards, plane)
    assert decimal_value(bound) <= actual
    assert abs(float(actual) - bound) < 1e-11
    if offset > 0:
        assert bound < -.12  # A genuinely negative lower bound is preserved.
    elif offset < 0:
        assert bound > .12


def test_nonstationary_point_still_bounds_the_ideal_minimum():
    standards, plane = -np.log(WEIGHTS), np.zeros(4)
    bound = ma_alloy_insertion_lower_bound(IDEAL, 2173.15, LOWER, UPPER,
                                          [.05, .015, .03], standards, plane)
    assert decimal_value(bound) <= ideal_minimum(standards, plane)


def test_linear_offsets_remain_separate_exact_constants():
    standards = -np.log(WEIGHTS)
    offsets = np.full(4, .125)
    plane = np.zeros(4)
    actual = ideal_minimum(standards, plane, offsets=offsets)
    bound = ma_alloy_insertion_lower_bound(IDEAL, 2173.15, LOWER, UPPER, POINT,
                                          standards, plane, standard_offsets_rt=offsets)
    assert decimal_value(bound) <= actual
    assert abs(float(actual) - bound) < 1e-11


def test_fixed_coordinate_does_not_enter_the_gradient_norm():
    lower, upper = LOWER.copy(), UPPER.copy()
    lower[3] = upper[3] = POINT[2]
    standards, plane = -np.log(WEIGHTS), np.array([0., 0., 0., 5.])
    actual = ideal_minimum(standards, plane, fixed_h=POINT[2])
    bound = ma_alloy_insertion_lower_bound(IDEAL, 2173.15, lower, upper, POINT, standards, plane)
    assert decimal_value(bound) <= actual
    assert abs(float(actual) - bound) < 1e-11


def test_ma_insertion_near_independent_ad_tangent_against_decimal_scalar():
    model, temperature = MaFeSiOHLiquid(), 2173.15
    standards = np.array([-8.08, 1.63, -10.89, -7.2])

    def extensive(amounts):
        total = amounts.sum()
        x = amounts / total
        return total * (jnp.dot(x, standards) + jnp.sum(x * jnp.log(x))
                        + model.gex_RT(temperature, 1e5, x))

    plane = np.asarray(jax.grad(extensive)(jnp.array(WEIGHTS)))
    bound = ma_alloy_insertion_lower_bound(model, temperature, LOWER, UPPER, POINT, standards, plane)
    with localcontext() as context:
        context.prec = 70
        si, ox, h = map(decimal_value, POINT)
        x = [1 - si - ox - h, si, ox, h]
        dry = 1 - h
        si, ox = si / dry, ox / dry
        a, b, product = 1 - si, 1 - ox, si * ox
        pair = -product - si * b.ln() - ox * a.ln() + product**2 * (1/a + 1/b - 1) / 2
        c = list(map(decimal_value, model.dry_model.interaction_K))
        scalar = dry * (c[0] * a * a.ln() + c[1] * b * b.ln() + c[2] * pair) / decimal_value(temperature)
        scalar += sum(v * (v.ln() + decimal_value(s) - decimal_value(p))
                      for v, s, p in zip(x, standards, plane))
        assert decimal_value(bound) <= scalar
        assert abs(float(scalar) - bound) < 1e-11


@pytest.mark.parametrize("field", ["point", "standard", "plane", "offsets"])
@pytest.mark.parametrize("bad", [complex(1, 1), np.nan, np.inf])
def test_invalid_values_cannot_produce_a_certificate(field, bad):
    values = {"point": POINT.copy(), "standard": np.zeros(4), "plane": np.zeros(4), "offsets": np.zeros(4)}
    values[field] = np.asarray(values[field], dtype=complex if isinstance(bad, complex) else float)
    values[field][0] = bad
    with pytest.raises(ValueError):
        ma_alloy_insertion_lower_bound(IDEAL, 2173.15, LOWER, UPPER,
                                       values["point"], values["standard"], values["plane"],
                                       standard_offsets_rt=values["offsets"])


@pytest.mark.parametrize("point", [[0., .01, .02], [.1, .01, .02], [.01, .01], [.01, .01, .02, .03]])
def test_invalid_or_unenclosed_point_is_rejected(point):
    with pytest.raises(ValueError):
        ma_alloy_insertion_lower_bound(IDEAL, 2173.15, LOWER, UPPER, point, np.zeros(4), np.zeros(4))


def test_nonpositive_curvature_is_not_an_insertion_certificate():
    with pytest.raises(ValueError, match="positive global"):
        ma_alloy_insertion_lower_bound(MaFeSiOHLiquid(), 2173.15,
                                       [.1, 0., 0., 0.], [1., .4, .4, .3], POINT,
                                       np.zeros(4), np.zeros(4))
