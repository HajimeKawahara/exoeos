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


def test_selected_published_callback_caches_standards_and_keeps_only_computed_properties(monkeypatch, tmp_path):
    native_spec = importlib.util.spec_from_file_location("native", ROOT / "examples/melts_liquid_evaluator.py")
    native = importlib.util.module_from_spec(native_spec)
    native_spec.loader.exec_module(native)
    calls = []

    def reference_call(t, p, n, *, common_R, **kwargs):
        calls.append((t, p))
        coefficients = mixing.liquid_mixing_parameters(t, p, common_R)
        state = mixing.liquid_mixing_state(coefficients, n)
        standard = np.arange(19, dtype=float) + t + p*1e-5
        mu = standard/(common_R*t) + np.asarray(state["mixing_mu_rt"], dtype=float)
        present = np.asarray(n) > 0
        nullable = lambda values: [float(v) if exists else None for v, exists in zip(values, present)]
        return {"model_id": native.MODEL_ID, "T_K": t, "P_Pa": p,
                "component_order": native.COMPONENTS, "component_moles": list(n),
                "mu_RT": nullable(mu), "mu0_RT": nullable(standard/(common_R*t)),
                "mu0_J_mol": nullable(standard),
                "basis": {"common_R_J_mol_K": common_R, "element_moles": [999.]},
                "phase_policy": {"oxygen_buffer": "None"}, "density": 999., "enthalpy_J": 123.}

    monkeypatch.setattr(native, "evaluate_liquid", reference_call)
    provider = mixing.make_published_liquid_evaluator(native, runtime=tmp_path, python_executable="worker")
    parent = np.eye(19)[0] + 2*np.eye(19)[7]
    first = provider.evaluate_liquid(2173.15, 3e7, parent)
    for index in (0, 7, 18):
        state = provider.evaluate_liquid(2173.15, 3e7, np.eye(19)[index])
        assert state["gibbs_RT"] == state["mu_RT"][index]
        assert "density" not in state and "enthalpy_J" not in state
        assert state["provenance"]["native_composition_evaluated"] is False
    assert calls == [(2173.15, 3e7)]
    assert len(provider.standard_state_receipts) == 1
    other = provider.evaluate_liquid(2173.15, 4e7, parent)
    assert calls == [(2173.15, 3e7), (2173.15, 4e7)]
    assert other["provenance"]["native_standard_state_receipt_sha256"] != first["provenance"]["native_standard_state_receipt_sha256"]
    assert other["model_id"] == mixing.PUBLISHED_MODEL_ID
    present = parent > 0
    assert first["gibbs_RT"] == pytest.approx(parent[present]@np.asarray(first["mu_RT"], float)[present])

    # A trace phosphate amount makes macroscopic total-energy differences
    # unresolved; the independent scalar AD keeps the same intensive limit.
    for scale in (1., 1e24):
        for water in (0., .02, 1.):
            n = np.zeros(19)
            n[0], n[13], n[18] = 1-water, 1e-18 if water < 1 else 0., water
            n *= scale
            state = provider.evaluate_liquid(2173.15, 3e7, n)
            energy, gradient = provider.energy_value_and_grad_rt(2173.15, 3e7, n)
            present = n > 0
            assert energy == pytest.approx(state["gibbs_RT"], rel=2e-14, abs=1e-14)
            np.testing.assert_allclose(gradient[present], np.asarray(state["mu_RT"], float)[present],
                                       rtol=0, atol=5e-12)
            assert np.all(np.isnan(gradient[~present]))

    original = mixing.liquid_mixing_state
    def incorrect_mixture(parameters, n):
        state = original(parameters, n)
        state["mixing_gibbs_rt"] += .01*np.sum(n)
        state["mixing_mu_rt"][0] += .02
        return state
    monkeypatch.setattr(mixing, "liquid_mixing_state", incorrect_mixture)
    wrong = provider.evaluate_liquid(2173.15, 3e7, parent)
    independent, gradient = provider.energy_value_and_grad_rt(2173.15, 3e7, parent)
    assert wrong["gibbs_RT"]-independent == pytest.approx(.01*parent.sum())
    assert wrong["mu_RT"][0]-gradient[0] == pytest.approx(.02)
