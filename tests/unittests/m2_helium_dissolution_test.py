"""Single-G He uptake, host derivatives and exact finite-host endpoints."""

import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
PATH = Path(__file__).resolve().parents[2] / "examples/m2_material/helium_dissolution.py"
SPEC = importlib.util.spec_from_file_location("helium_dissolution", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def model():
    return MODULE.make_helium_dissolution("guillot2012_olivine", 2173.15,
                                         [.06, .04, 0., 0.], -12.)


def test_analytic_potentials_equal_independent_scalar_ad_and_euler():
    m = model()
    n = np.array([.5, .2, .03, .01, .00002])
    state = m.state(n)
    energy, derivative = m.energy_value_and_grad_rt(n)
    assert energy == pytest.approx(state["gibbs_rt"], abs=1e-16)
    np.testing.assert_allclose(derivative, state["mu_rt"], atol=1e-14)
    assert n @ state["mu_rt"] == pytest.approx(energy, abs=1e-16)
    np.testing.assert_array_equal(state["mu_rt"][[2, 3]], 0.)
    scaled = m.state(1e24 * n)
    assert scaled["gibbs_rt"] / 1e24 == pytest.approx(energy)
    np.testing.assert_allclose(scaled["mu_rt"], state["mu_rt"], atol=1e-14)


def test_measured_unit_henry_limit_uses_dry_mass_and_common_gas_standard():
    m = model()
    capacity = m.receipt["capacity"]["He_mol_per_kg_dry_host_per_bar_fugacity"]
    host = np.array([.5, .2, .03, .01])
    dry_mass = .5*.06 + .2*.04
    p_he = 45.87511296696385
    amount = dry_mass * capacity * p_he
    state = m.state(np.r_[host, amount])
    assert state["mu_rt"][-1] == pytest.approx(-12. + np.log(p_he))
    assert state["mu_rt"][0] == pytest.approx(-capacity*p_he*.06)
    wetter = host.copy()
    wetter[-2:] *= 100.
    np.testing.assert_allclose(m.state(np.r_[wetter, amount])["mu_rt"], state["mu_rt"])


def test_absent_helium_and_absent_host_have_explicit_distinct_limits():
    m = model()
    n = np.array([1., 0., 0., 0., 0.])
    state = m.state(n)
    assert state["gibbs_rt"] == 0.
    np.testing.assert_array_equal(state["mu_rt"][:-1], 0.)
    assert state["mu_rt"][-1] == -np.inf
    energy, gradient = m.energy_value_and_grad_rt(n)
    assert energy == 0.
    np.testing.assert_array_equal(gradient, state["mu_rt"])
    absent = m.state(np.zeros(5))
    assert absent["gibbs_rt"] == 0.
    assert np.all(np.isnan(absent["mu_rt"]))
    for invalid in ([0., 0., 0., 0., 1.], [0., 0., 1., 0., 0.], [-1., 0., 0., 0., 0.]):
        with pytest.raises(ValueError):
            m.state(invalid)


def test_no_hidden_temperature_continuation_or_empirical_acceptance():
    m = model()
    assert not m.receipt["supports_material_admission"]
    assert m.receipt["model"] == "guillot2012_olivine"
    with pytest.raises(ValueError, match="interval"):
        MODULE.make_helium_dissolution("guillot2012_morb", 3000., [.05], 0.)
    with pytest.raises(ValueError, match="Select"):
        MODULE.make_helium_dissolution("BSE", 2173.15, [.05], 0.)
