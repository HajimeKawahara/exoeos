"""Independent native receipts and analytic limits of the published expression."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("mixing", ROOT / "examples/melts_liquid_mixing.py")
mixing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mixing)
REFERENCE = json.loads((ROOT / "tests/reference/melts_silicate_v1.json").read_text())


@pytest.mark.parametrize("state", REFERENCE["states"])
def test_published_mixing_potentials_match_independent_native_references(state):
    common_r = 8.31446261815324
    parameters = mixing.liquid_mixing_parameters(state["T_K"], state["P_Pa"], common_r)
    liquid = state["liquid"]
    expected = (np.asarray(liquid["mu_J_mol"], float)-np.asarray(liquid["mu0_J_mol"], float))/(common_r*state["T_K"])
    actual = mixing.liquid_mixing_state(parameters, liquid["component_moles"])
    present = np.asarray(liquid["component_moles"]) > 0
    np.testing.assert_allclose(np.asarray(actual["mixing_mu_rt"], float)[present], expected[present], atol=2e-8, rtol=0)
    assert actual["mixing_gibbs_rt"] == pytest.approx(np.dot(np.asarray(liquid["component_moles"])[present], expected[present]), abs=2e-8)


@pytest.mark.parametrize("component", [0, 7, 18])
def test_pure_limits_have_zero_mixing_energy_and_present_potential(component):
    parameters = mixing.liquid_mixing_parameters(2173.15, 3e7)
    n = np.zeros(19)
    n[component] = 7.3
    state = mixing.liquid_mixing_state(parameters, n)
    assert state["mixing_gibbs_rt"] == 0
    assert state["mixing_mu_rt"][component] == 0


def test_extensive_gradient_and_amount_scaling_use_the_same_expression():
    parameters = mixing.liquid_mixing_parameters(2173.15, 3e7)
    n = np.asarray(REFERENCE["states"][0]["liquid"]["component_moles"])
    state = mixing.liquid_mixing_state(parameters, n)
    scaled = mixing.liquid_mixing_state(parameters, 23*n)
    assert scaled["mixing_gibbs_rt"] == pytest.approx(23*state["mixing_gibbs_rt"])
    for i in np.flatnonzero(n):
        step = 1e-5*n[i]
        plus, minus = n.copy(), n.copy()
        plus[i] += step
        minus[i] -= step
        derivative = (mixing.liquid_mixing_state(parameters, plus)["mixing_gibbs_rt"]
                      - mixing.liquid_mixing_state(parameters, minus)["mixing_gibbs_rt"])/(2*step)
        assert derivative == pytest.approx(state["mixing_mu_rt"][i], abs=2e-6)


def test_carbon_model_is_not_silently_treated_as_v102():
    parameters = mixing.liquid_mixing_parameters(2173.15, 3e7)
    n = np.eye(19)[14]
    with pytest.raises(ValueError, match="unsupported"):
        mixing.liquid_mixing_state(parameters, n)
