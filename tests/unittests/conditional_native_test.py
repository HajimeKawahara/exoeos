"""Packaged native scalars against the independent pinned example formulas."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import associated_metal as metal, magma, magma_volatiles as liquid, total_gex_RT

ROOT = Path(__file__).resolve().parents[2]


def load(relative):
    spec = importlib.util.spec_from_file_location("conditional_reference", ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("temperature", [2101., 2173.15, 2399.])
@pytest.mark.parametrize("policy", ["constant", "enthalpic"])
def test_twenty_species_scalar_and_potentials_match_reference(temperature, policy):
    reference = load("examples/m2_material/sodium_metal.py")
    model, _ = reference.make_associated_model(temperature, temperature_policy=policy)
    n = jnp.array([.9, .002, .003, .03, .004, 1e-6, 1e-9, 2e-6, .03, 1e-7,
                   .0004, 5e-7, 3e-6, .001, 1e-7, 5e-10, .0001, 1e-12, .0005, .0008])
    epsilon, matrix = metal.interaction_parameters(temperature, temperature_policy=policy)
    native = lambda n: metal.excess_gibbs_rt(temperature, 1e7, n, epsilon, matrix)
    old = lambda n: total_gex_RT(model, temperature, 1e7, n)
    np.testing.assert_allclose(native(n), old(n), atol=2e-16, rtol=0)
    np.testing.assert_allclose(jax.grad(native)(n), jax.grad(old)(n), atol=2e-13, rtol=0)
    np.testing.assert_allclose(jax.hessian(native)(n), jax.hessian(old)(n), atol=2e-11, rtol=0)


def test_full_volatile_scalar_matches_old_water_and_trace_helium():
    basis = load("examples/melts_liquid_evaluator.py")
    mixing = load("examples/melts_liquid_mixing.py")
    water = load("examples/melts_water_reconstruction.py")
    helium = load("examples/m2_material/helium_dissolution.py")
    def dry(t, p, n, **kwargs):
        state = mixing.liquid_mixing_state(mixing.liquid_mixing_parameters(t, p), n)
        return dict(component_moles=list(n), gibbs_RT=state["mixing_gibbs_rt"], mu_RT=state["mixing_mu_rt"],
                    basis=dict(common_R_J_mol_K=metal.R), provenance={})
    stub = SimpleNamespace(MODEL_ID=mixing.PUBLISHED_MODEL_ID, evaluate_liquid=dry,
        **{name: getattr(basis, name) for name in ("COMPONENTS", "OXIDES", "NU", "OXIDE_MASSES", "ELEMENTS", "FORMULA_MATRIX")})
    old = water.make_reconstructed_water_evaluator(stub, lambda t, p: -30.)
    n = np.r_[np.linspace(.01, .05, 11), .002, .0001, 1e-6]
    indices = [basis.COMPONENTS.index(name.lower()) for name in magma.DRY11_COMPONENTS+("H2O",)]
    native_n = np.zeros(19); native_n[indices] = n[:12]
    t, p = 2173.15, 1e7
    standards = np.asarray(magma.liquid_standard_gibbs(t, p, include_dry11=True))/(metal.R*t)
    masses = np.linspace(.04, .2, 11)
    def old_value(n):
        native_n[indices] = np.asarray(n[:12])
        value = old.evaluate_liquid(t, p, native_n)["gibbs_RT"]+standards @ n[:11]
        host, h = np.sum(n[:12]), n[12]
        value += h*(-20.+11.403+.76*p/1e9)+host*np.log(host/(host+h))+h*np.log(h/(host+h))
        value += float(helium.helium_dissolution_gibbs_rt(n, np.r_[masses, 0., 0.], liquid.helium_capacity(t), -15.))
        return value
    fun = lambda n: liquid.gibbs_j(t, p, n, jnp.array([-30., -20., -15.]), masses)/(metal.R*t)
    assert float(fun(n)) == pytest.approx(old_value(n), abs=2e-13)
    gradient = np.asarray(jax.grad(fun)(n))
    for i in range(14):
        step = n[i]*1e-4
        delta = np.eye(14)[i]*step
        numeric = (old_value(n+delta)-old_value(n-delta))/(2*step)
        assert gradient[i] == pytest.approx(numeric, abs=8e-5)
    assert np.all(np.isfinite(jax.hessian(fun)(n)))
    assert np.isnan(liquid.helium_capacity(2399.))
    assert float(liquid.helium_capacity(t)) == pytest.approx(2.458350594263779e-5, rel=2e-14)


@pytest.mark.parametrize("temperature", [2101., 2173.15, 2399.])
def test_twenty_actual_standards_match_independent_reference(temperature):
    reference = load("examples/m2_material/sodium_metal.py")
    host = np.array([-25., -40., -45., -15.])
    gas = np.array([-20., -30., -10., -15., -30., -40., -50., -10.])
    pressure = 1e7
    options = dict(projection="remove_potassium_0.25", temperature_policy="published_exchange_slope")
    delta = metal.sodium_reference_parameters(**options)
    native = lambda t: metal.standard_potentials_rt(t, pressure, host, gas,
        potassium_offset_rt=-.09476121345453237, sodium_delta=delta)
    parent = reference._potassium()
    phosphorus = parent._associated()._phosphorus().phosphorus_standard_rt(
        temperature, gas[0], gas_species="P1", allow_temperature_continuation=True)["standard_rt"]
    old, _ = parent.associated_standards_rt(temperature, np.r_[host, phosphorus],
        dict(zip(metal.GAS_STANDARD_SPECIES, gas)), potassium_standard_offset_rt=-.09476121345453237)
    mu = np.asarray(magma.liquid_standard_gibbs(temperature, pressure, include_dry11=True))/(metal.R*temperature)
    properties = dict(T_K=temperature, P_Pa=pressure, basis=dict(common_R_J_mol_K=metal.R),
        component_order=[n.lower() for n in magma.DRY11_COMPONENTS], mu0_RT=mu.tolist(), model_id="native_dry11", provenance={})
    sodium, _ = reference.sodium_standard_rt(temperature, pressure, host[0], properties, **options)
    np.testing.assert_allclose(native(temperature), np.r_[old, sodium], atol=3e-14, rtol=0)
    assert np.all(np.isfinite(jax.jacfwd(native)(temperature)))
