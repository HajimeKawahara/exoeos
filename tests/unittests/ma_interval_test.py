"""Independent AD comparisons for the Ma alloy's conservative domain bound."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos.ma_fe_si_o import MaFeSiOLiquid
from exoeos.ma_fe_si_o_h import MaFeSiOHLiquid
from exoeos.ma_interval import ma_alloy_curvature_lower_bound


def test_fe_rich_box_has_positive_bound_below_independent_hessians():
    model = MaFeSiOHLiquid()
    lower, upper = np.array([.86, 0., 0., 0.]), np.array([1., .08, .02, .04])
    bound = ma_alloy_curvature_lower_bound(model, 2173.15, lower, upper)
    assert bound > 0

    def molar_energy(solutes):
        x = jnp.concatenate((jnp.array([1 - solutes.sum()]), solutes))
        return jnp.sum(x * jnp.log(x)) + model.gex_RT(2173.15, 1e5, x)

    hessian = jax.jit(jax.hessian(molar_energy))
    # Independent AD checks include near-zero and nearly maximal solutes.
    # The proof covers the entire box; these points guard the implementation.
    for point in np.random.default_rng(74).uniform(.001, .999, (40, 3)) * upper[1:]:
        actual = np.asarray(hessian(point))
        assert np.linalg.eigvalsh(actual).min() >= bound


def test_zero_interactions_reduce_to_a_conservative_ideal_mixing_bound():
    model = MaFeSiOHLiquid(MaFeSiOLiquid((0., 0., 0.)))
    lower, upper = [.7, 0., 0., 0.], [1., .1, .1, .1]
    bound = ma_alloy_curvature_lower_bound(model, 2000., lower, upper)
    assert 9.999999 < bound <= 10.


def test_fixed_zero_hydrogen_is_removed_without_a_trace_floor():
    model = MaFeSiOHLiquid()
    lower, upper = [.9, 0., 0., 0.], [1., .08, .02, 0.]
    bound = ma_alloy_curvature_lower_bound(model, 2173.15, lower, upper)
    assert np.isfinite(bound) and bound > 0
    assert ma_alloy_curvature_lower_bound(model, 2173.15, [1., 0., 0., 0.],
                                         [1., 0., 0., 0.]) == 0


def test_broad_domain_is_not_automatically_certified_convex():
    bound = ma_alloy_curvature_lower_bound(MaFeSiOHLiquid(), 2173.15,
                                          [.1, 0., 0., 0.], [1., .4, .4, .3])
    assert bound < 0


def test_modified_model_scalar_cannot_inherit_the_original_curvature_certificate():
    class ModifiedDry(MaFeSiOLiquid):
        def gex_RT(self, temperature, pressure, x):
            return super().gex_RT(temperature, pressure, x) - 1000 * x[1]**2

    class ModifiedAlloy(MaFeSiOHLiquid):
        def gex_RT(self, temperature, pressure, x):
            return super().gex_RT(temperature, pressure, x) - 1000 * x[1]**2

    for model in (ModifiedAlloy(), MaFeSiOHLiquid(ModifiedDry())):
        with pytest.raises(TypeError, match="unmodified"):
            ma_alloy_curvature_lower_bound(model, 2173.15, [.86, 0., 0., 0.],
                                           [1., .08, .02, .04])


@pytest.mark.parametrize("item", ["lower", "upper", "interactions", "temperature"])
def test_complex_inputs_are_rejected_before_any_real_cast(item):
    model = MaFeSiOHLiquid()
    lower, upper, temperature = np.array([.86, 0., 0., 0.]), np.array([1., .08, .02, .04]), 2173.15
    if item == "lower":
        lower = lower + 1j
    elif item == "upper":
        upper = upper + 1j
    elif item == "temperature":
        temperature += 1j
    else:
        model = MaFeSiOHLiquid(MaFeSiOLiquid([1j, 0., 0.]))
    with pytest.raises(ValueError, match="real"):
        ma_alloy_curvature_lower_bound(model, temperature, lower, upper)


@pytest.mark.parametrize("lower,upper", [
    ([0., 0., 0., 0.], [1., .1, .1, .1]),
    ([.9, .2, 0., 0.], [1., .3, .1, .1]),
    ([.5, 0., 0., 0.], [.6, .1, .1, .1]),
    ([.8, -.1, 0., 0.], [1., .1, .1, .1]),
    ([.8, 0., 0., 0.], [1., float("nan"), .1, .1]),
    ([.1, 0., 0., 0.], [1., .6, .6, .5]),
])
def test_invalid_or_singular_box_cannot_supply_a_bound(lower, upper):
    with pytest.raises(ValueError):
        ma_alloy_curvature_lower_bound(MaFeSiOHLiquid(), 2173.15, lower, upper)
