"""Water substitutions preserve the gas gauge and integrable host response."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import magma, magma_carbonate, magma_fluid, magma_volatiles as liquid
from exoeos._conditional_data import R

T, P = 2173.15, 2.7e7
DRY = jnp.linspace(.01, .05, 11)
MASSES = jnp.asarray(liquid.water_parameters()["dry_molar_masses_kg_mol"])
GAS = jnp.array([-30., -20., -15.])
MODELS = ("thompson2025", "power_law", "melts_mode1", "melts_mode4")


def energy(n, model, gas=GAS):
    return liquid.gibbs_j(T, P, n, gas, MASSES, water_model=model,
                         water_capacity_1bar=.001, water_pressure_exponent=1/1.747)/(R*T)


@pytest.mark.parametrize("pressure_bar", [.02, 1., 440., 8000.])
def test_power_law_recovers_water_fugacity_and_dry_host_response(pressure_bar):
    c, alpha = .001, 1/1.747
    w = c*pressure_bar**alpha
    equivalent = MASSES @ DRY/liquid.water_parameters()["water_molar_mass_kg_mol"]
    water = equivalent*w/(1-w)
    # No molecular H2: test the stated binary dry-host/water law exactly.
    fun = lambda n: energy(jnp.r_[n, 0., 0.], "power_law")
    n = jnp.r_[DRY, water]
    mu = jax.grad(fun)(n)
    assert float(mu[-1]-GAS[0]) == pytest.approx(np.log(pressure_bar), abs=2e-13)
    dry_mu = jax.grad(lambda n: magma.liquid_gibbs(T, P, n)/(R*T))(DRY)
    expected = MASSES/liquid.water_parameters()["water_molar_mass_kg_mol"]*np.log1p(-w)/alpha
    np.testing.assert_allclose(mu[:11]-dry_mu, expected, atol=2e-13, rtol=0)


@pytest.mark.parametrize("model", MODELS)
def test_extensive_scalar_gauge_curvature_and_unchanged_guests(model):
    n = jnp.r_[DRY, .004, .0002, 1e-6]
    fun = lambda n: energy(n, model)
    mu, hessian = jax.grad(fun)(n), jax.hessian(fun)(n)
    assert np.all(np.isfinite(hessian))
    assert float(n @ mu) == pytest.approx(float(fun(n)), abs=2e-13)
    np.testing.assert_allclose(hessian @ n, 0., atol=2e-12, rtol=0)
    np.testing.assert_allclose(mu[-2:], jax.grad(lambda n: energy(n, "thompson2025"))(n)[-2:], atol=2e-13)
    shifted = jax.grad(lambda n: energy(n, model, GAS+jnp.array([3., 0., 0.])))(n)
    np.testing.assert_allclose(shifted-mu, np.eye(14)[11]*3, atol=2e-13, rtol=0)
    direction = n*jnp.linspace(-.2, .3, 14)
    step = 1e-4
    numeric = (jax.grad(fun)(n+step*direction)-jax.grad(fun)(n-step*direction))/(2*step)
    np.testing.assert_allclose(hessian @ direction, numeric, atol=2e-9, rtol=2e-7)


@pytest.mark.parametrize("model", ["melts_mode1", "melts_mode4"])
def test_native_water_has_one_hydrous_contribution(model):
    n = jnp.r_[DRY, .004]
    hydrous = magma.liquid_gibbs if model == "melts_mode1" else magma_carbonate.hydrous_liquid_gibbs
    expected = (hydrous(T, P, n)-n[-1]*magma_fluid.water_gas_standard_gibbs(T))/(R*T)+n[-1]*GAS[0]
    assert float(energy(jnp.r_[n, 0., 0.], model)) == pytest.approx(float(expected), abs=2e-13)


@pytest.mark.parametrize("t,p", [(1000., 1e5), (1673.15, 1e8), (2173.15, 4.4e7)])
def test_mode4_carbon_free_face_matches_full_energy_and_tangent_derivatives(t, p):
    n = jnp.r_[DRY, .004]
    fun = lambda n: magma_carbonate.hydrous_liquid_gibbs(t, p, n)
    assert float(fun(n)) == pytest.approx(float(magma_carbonate.liquid_gibbs(t, p, jnp.r_[n, 0.])), abs=1e-9)
    mu, hessian = jax.grad(fun)(n), jax.hessian(fun)(n)
    assert np.all(np.isfinite(hessian))
    assert float(mu @ n) == pytest.approx(float(fun(n)), abs=2e-9)
    step, direction = 1e-4, n*jnp.linspace(-.3, .4, 12)
    numeric = (jax.grad(fun)(n+step*direction)-jax.grad(fun)(n-step*direction))/(2*step)
    np.testing.assert_allclose(hessian @ direction, numeric, atol=2e-6, rtol=2e-7)
    assert float(magma_carbonate.hydrous_liquid_gibbs(t, p, jnp.zeros(12))) == 0.


def test_water_selection_requires_explicit_valid_power_law():
    n = jnp.r_[DRY, .004, .0002, 1e-6]
    with pytest.raises(ValueError, match="Unknown"):
        energy(n, "unknown")
    with pytest.raises(ValueError, match="Supply"):
        liquid.gibbs_j(T, P, n, GAS, MASSES, water_model="power_law")
    assert np.isnan(liquid.gibbs_j(T, P, n, GAS, MASSES, water_model="power_law",
                                 water_capacity_1bar=-.1, water_pressure_exponent=.5))
    assert np.isnan(liquid.gibbs_j(T, P, n, GAS, MASSES, water_model="power_law",
                                 water_capacity_1bar=.001, water_pressure_exponent=0.))
    with pytest.raises(ValueError, match="shape"):
        magma_carbonate.hydrous_liquid_gibbs(T, P, jnp.ones(13))
