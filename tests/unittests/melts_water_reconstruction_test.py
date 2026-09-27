"""Replacement water G, host derivatives, zero water and gauge covariance."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


water = load("water_reconstruction", "melts_water_reconstruction.py")
native = load("water_native_basis", "melts_liquid_evaluator.py")
mixing = load("water_dry_mixing", "melts_liquid_mixing.py")


@pytest.fixture
def dry():
    calls = []

    def evaluate(t, p, n, **options):
        calls.append(np.asarray(n).copy())
        assert n[native.COMPONENTS.index("h2o")] == 0
        mixed = mixing.liquid_mixing_state(mixing.liquid_mixing_parameters(t, p), n)
        return {"model_id": mixing.PUBLISHED_MODEL_ID, "component_order": native.COMPONENTS,
                "component_moles": list(n), "mu_RT": mixed["mixing_mu_rt"],
                "gibbs_RT": mixed["mixing_gibbs_rt"],
                "basis": {"common_R_J_mol_K": 8.31446261815324},
                "phase_policy": {"oxygen_buffer": "None"}, "provenance": {}}

    def derivative(t, p, n, **options):
        state = evaluate(t, p, n, **options)
        return state["gibbs_RT"], np.asarray(state["mu_RT"], float)

    return SimpleNamespace(MODEL_ID=mixing.PUBLISHED_MODEL_ID, COMPONENTS=native.COMPONENTS,
                           OXIDES=native.OXIDES, NU=native.NU, OXIDE_MASSES=native.OXIDE_MASSES,
                           ELEMENTS=native.ELEMENTS, FORMULA_MATRIX=native.FORMULA_MATRIX,
                           evaluate_liquid=evaluate, energy_value_and_grad_rt=derivative, calls=calls)


def composition():
    n = np.zeros(19)
    for name, value in {"sio2": .21, "fe2o3": .001, "fe2sio4": .03,
                        "mg2sio4": .24, "casio3": .015, "al2o3": .02, "kalsio4": .001,
                        "ca3(po4)2": 1e-8, "h2o": .01}.items():
        n[native.COMPONENTS.index(name)] = value
    return n


def test_full_host_atoms_g_gradient_and_all_dry_predictors(dry):
    provider = water.make_reconstructed_water_evaluator(dry, lambda t, p: -3.2)
    n = composition()
    t, p = 2173.15, 27e6
    state = provider.evaluate_liquid(t, p, n)
    active = n > 0
    mu = np.asarray(state["mu_RT"], float)
    assert state["model_id"] == water.MODEL_ID
    assert state["returned_component_moles"] == n.tolist()
    np.testing.assert_allclose(state["basis"]["element_moles"], native.FORMULA_MATRIX.T @ n)
    assert state["gibbs_RT"] == pytest.approx(n[active] @ mu[active], abs=2e-12)
    # Independent finite differences of the assembled scalar include the
    # Ca/Al/Si host responses and the ferric-iron predictor conversion.
    for i in np.flatnonzero(active & (n > 1e-7)):
        step = n[i]*1e-5
        plus, minus = n.copy(), n.copy()
        plus[i] += step
        minus[i] -= step
        numeric = (provider.evaluate_liquid(t, p, plus)["gibbs_RT"]
                   - provider.evaluate_liquid(t, p, minus)["gibbs_RT"])/(2*step)
        assert numeric == pytest.approx(mu[i], abs=2e-7)
    energy, gradient = provider.energy_value_and_grad_rt(t, p, n)
    assert energy == pytest.approx(state["gibbs_RT"], abs=1e-12)
    np.testing.assert_allclose(gradient[active], mu[active], atol=3e-12, rtol=0)
    scaled = provider.evaluate_liquid(t, p, 1e24*n)
    assert scaled["gibbs_RT"] == pytest.approx(1e24*state["gibbs_RT"], rel=1e-12)
    np.testing.assert_allclose(np.asarray(scaled["mu_RT"], float)[active], mu[active], atol=1e-12)
    assert all(v[native.COMPONENTS.index("h2o")] == 0 for v in dry.calls)
    assert len(provider.water_standard_receipts) == 1
    assert state["water_reconstruction"]["accepted_coupled_material_domain"] is None
    assert "mixing_expression" not in state


def test_zero_water_reduces_to_dry_and_standard_shift_is_linear(dry):
    n = composition()
    provider = water.make_reconstructed_water_evaluator(dry, lambda t, p: 0.)
    shifted = water.make_reconstructed_water_evaluator(dry, lambda t, p: 7.)
    t, p = 2173.15, 27e6
    base = provider.evaluate_liquid(t, p, n)
    changed = shifted.evaluate_liquid(t, p, n)
    assert changed["gibbs_RT"]-base["gibbs_RT"] == pytest.approx(7*n[-1])
    assert changed["mu_RT"][-1]-base["mu_RT"][-1] == pytest.approx(7)
    n[-1] = 0
    assert provider.evaluate_liquid(t, p, n)["gibbs_RT"] == pytest.approx(dry.evaluate_liquid(t, p, n)["gibbs_RT"])
    with pytest.raises(ValueError, match="positive dry host"):
        provider.evaluate_liquid(t, p, np.eye(19)[-1])


def test_saved_expression_reconstructs_scalar_without_provider_constants(dry):
    provider = water.make_reconstructed_water_evaluator(dry, lambda t, p: -3.2)
    n, t, p = composition(), 2173.15, 27e6
    state = provider.evaluate_liquid(t, p, n)
    saved = json.loads(json.dumps(state["water_reconstruction"]))
    expression = saved["expression"]
    i = expression["water_index"]
    h = n[i]
    host = n.copy()
    host[i] = 0
    counts = np.asarray(expression["predictor_oxide_counts"])
    weights = np.asarray(expression["predictor_temperature_weights_K"])
    s = np.asarray(expression["component_masses_kg_mol"]) @ host / expression["water_mass_kg_mol"]
    ln_capacity = expression["log_capacity_prefactor"] + (weights @ host)/(t*(counts @ host))
    water_g = h*(saved["gas_H2O_standard_RT"]-2*ln_capacity)
    water_g += 2*(s*np.log(s/(s+h))+h*np.log(h/(s+h)))
    assert saved["dry_properties"]["component_moles"][i] == 0
    assert saved["dry_properties"]["gibbs_RT"]+water_g == pytest.approx(state["gibbs_RT"], abs=1e-12)
    assert expression == provider.water_expression_parameters
    # A saved result must not alias factory coefficients or dry results.
    state["water_reconstruction"]["expression"]["water_index"] = -1
    assert provider.water_expression_parameters["water_index"] == i
