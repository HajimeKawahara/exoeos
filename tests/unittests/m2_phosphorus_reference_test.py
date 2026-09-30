"""Shared-gauge and finite-scalar checks for the primary P continuation."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import MaFeSiOHLiquid, total_solution_gibbs_RT, total_solution_state

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/m2_material"))
from phosphorus_reference import (ATOMIC_MASSES, R, MaPhosphorusLiquid,
    mass_percent_to_mole_interaction, phosphorus_interactions, phosphorus_standard_rt,
    phosphorus_curvature_lower_bound, phosphorus_excess)


@pytest.mark.parametrize("species,fraction,a,b", [("P1", 1., -95.3, .0155), ("P2", .5, -37.7, .0013)])
def test_dissolution_standard_and_pressure_and_mass_conversions(species, fraction, a, b):
    t, gas, pressure_atm, wt_activity = 1873.15, -6.4, .031, .7
    report = phosphorus_standard_rt(t, gas, gas_species=species)
    expected = fraction * (gas + np.log(1.01325)) + 4184 * (a + b * t) / (R * t)
    assert report["standard_rt"] - report["mole_fraction_minus_mass_percent_standard_rt"] == pytest.approx(expected)
    # Equality of potentials yields the measured 1 wt% equilibrium quotient.
    activity = pressure_atm ** fraction * np.exp(-4184 * (a + b * t) / (R * t))
    mole_activity = activity * ATOMIC_MASSES["Fe"] / (100 * ATOMIC_MASSES["P"])
    assert report["standard_rt"] + np.log(mole_activity) == pytest.approx(fraction * (gas + np.log(pressure_atm * 1.01325)))
    shifted = phosphorus_standard_rt(t, gas + 2.5 / fraction, gas_species=species)
    assert shifted["standard_rt"] - report["standard_rt"] == pytest.approx(2.5)
    assert report["empirical_coupled_error_bound"] is None
    assert not report["temperature_continued"]


def test_temperature_continuation_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="opt in"):
        phosphorus_standard_rt(2173.15, -5., gas_species="P1")
    report = phosphorus_standard_rt(2173.15, -5., gas_species="P1", allow_temperature_continuation=True)
    assert report["temperature_continued"]
    assert not report["pressure_response_calibrated"]
    assert phosphorus_interactions(2173.15, temperature_policy="constant")["epsilon"][0] == 7.3
    assert phosphorus_interactions(2173.15, temperature_policy="enthalpic")["epsilon"][0] == pytest.approx(7.3 * 1873.15 / 2173.15)


def test_mass_percent_coefficient_conversion_uses_mean_mass_and_reciprocity():
    epsilon = mass_percent_to_mole_interaction(.015, ATOMIC_MASSES["P"])
    # Independent composition perturbation of Henry mass activity.
    def gamma(xp):
        mean = (1 - xp) * ATOMIC_MASSES["Fe"] + xp * ATOMIC_MASSES["P"]
        wp = 100 * xp * ATOMIC_MASSES["P"] / mean
        return .015 * wp * np.log(10.) + np.log(ATOMIC_MASSES["Fe"] / mean)
    step = 1e-6
    derivative = (gamma(step) - gamma(-step)) / (2 * step)
    assert epsilon == pytest.approx(derivative, rel=1e-9)


def test_finite_scalar_extensivity_reciprocity_host_limit_and_independent_derivatives():
    epsilon = np.asarray(phosphorus_interactions(2173.15, temperature_policy="constant")["epsilon"])
    model = MaPhosphorusLiquid(epsilon)
    t, p = 2173.15, 269.65e5
    n = jnp.array([.96, .003, .007, .025, .005])
    standards = jnp.array([-7., -8., -9., -10., -11.])
    energy = lambda v: total_solution_gibbs_RT(model, t, p, v, standards)
    model.validate_state(t, p, n)
    mu = jax.grad(energy)(n)
    hessian = jax.hessian(energy)(n)
    assert energy(7 * n) == pytest.approx(7 * energy(n), abs=5e-13)
    assert n @ mu == pytest.approx(energy(n), abs=5e-13)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-11, rtol=0)
    np.testing.assert_allclose(n @ hessian, 0, atol=1e-11, rtol=0)
    step = 1e-5
    fd = []
    for delta in step * np.eye(5):
        values = [energy(n + k * delta) for k in (-2, -1, 1, 2)]
        fd.append((values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step))
    np.testing.assert_allclose(mu, fd, atol=2e-9, rtol=2e-9)
    old = jnp.array([.96, .003, .007, .03])
    expected = total_solution_state(MaFeSiOHLiquid(), t, p, old, standards[:4])
    actual = total_solution_state(model, t, p, jnp.r_[old, 0.], standards)
    np.testing.assert_allclose(actual.mu_RT[:4], expected.mu_RT, atol=1e-13, rtol=0)
    assert actual.gibbs_RT == pytest.approx(expected.gibbs_RT, abs=1e-13)
    np.testing.assert_array_equal(model.standard_state_shift_RT(t)[:4], MaFeSiOHLiquid().standard_state_shift_RT(t))
    assert model.standard_state_shift_RT(t)[4] == 0


def test_jit_and_dilute_coefficients_match_source_values():
    epsilon = np.array([7.3, 11.9, 8.1, 2.36])
    model = MaPhosphorusLiquid(epsilon)
    # On the normalized composition simplex, derivative of mu_P excess is
    # the reported epsilon at infinite dilution; standards and ideal terms
    # do not enter this comparison.
    from exoeos import solution_state
    def potential(y):
        x = jnp.r_[1 - y.sum(), y]
        return solution_state(model, 1873.15, 1e5, x).lngamma[4]
    derivative = jax.jacfwd(potential)(jnp.zeros(4))
    np.testing.assert_allclose(derivative, epsilon[[1, 2, 3, 0]], atol=1e-12, rtol=0)
    compiled = jax.jit(lambda m, x: solution_state(m, 1873.15, 1e5, x).gex_RT)
    x = jnp.array([.96, .01, .01, .01, .01])
    assert compiled(model, x) == pytest.approx(model.gex_RT(1873.15, 1e5, x))


@pytest.mark.parametrize("x", [[1, 0, 0, 0, 1], [0, 0, 0, 0, 1], [1, 0, 0, 0, -1], [np.nan, 0, 0, 0, 0]])
def test_invalid_states_are_not_renormalized(x):
    with pytest.raises(ValueError):
        MaPhosphorusLiquid(np.zeros(4)).validate_state(2173.15, 1e5, x)


def test_interval_curvature_and_independent_scalar_match():
    from exoeos.ma_interval import _Interval
    epsilon = np.asarray(phosphorus_interactions(2173.15, temperature_policy="constant")["epsilon"])
    model = MaPhosphorusLiquid(epsilon)
    lo, hi = np.array([.86, 0, 0, 0, 0]), np.array([1, .08, .02, .04, .02])
    bound = phosphorus_curvature_lower_bound(model, 2173.15, lo, hi)
    assert 3. < bound < 3.1
    coefficients = np.asarray(model.host.dry_model.interaction_K)
    for point in ([.001, .003, .027, .006], [.04, .01, .025, .01], [.07, .018, .033, .019]):
        y = jnp.array(point)
        full = jnp.r_[1 - y.sum(), y]
        enclosed = phosphorus_excess(2173.15, coefficients, epsilon, [_Interval(v, v) for v in point])
        value = float(model.gex_RT(2173.15, 1e5, full))
        assert enclosed.lo <= value <= enclosed.hi
        hessian = jax.hessian(lambda y: total_solution_gibbs_RT(
            model, 2173.15, 1e5, jnp.r_[1 - y.sum(), y], jnp.zeros(5)))(y)
        assert np.linalg.eigvalsh(hessian).min() > bound
    fixed_lo, fixed_hi = lo.copy(), hi.copy()
    fixed_hi[4] = 0
    assert phosphorus_curvature_lower_bound(model, 2173.15, fixed_lo, fixed_hi) > 0
    with pytest.raises(ValueError):
        phosphorus_curvature_lower_bound(model, 2173.15, lo, [1, .8, .8, .9, .8])
