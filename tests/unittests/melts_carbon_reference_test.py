"""Offline checks of the carbon fixture produced by actual MELTS workers."""

import json
from pathlib import Path
import runpy

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = runpy.run_path(str(ROOT / "examples/melts_liquid_evaluator.py"))
REFERENCE = json.loads((ROOT / "tests/reference/melts_carbon_v1.json").read_text())
RECORDS = REFERENCE["records"]


@pytest.mark.parametrize("record", RECORDS, ids=lambda r: r["id"])
def test_carbon_fixture_has_consistent_independent_and_species_ledgers(record):
    state = record["result"]
    n = np.asarray(state["component_moles"])
    x = np.asarray(state["x"])
    species_x = np.asarray(state["backend_species"]["x"])
    elements = state["basis"]["element_order"]
    formula = np.asarray(state["basis"]["component_element_matrix"])
    carbonate = np.asarray([{"Ca": 1, "C": 1, "O": 3}.get(e, 0) for e in elements])
    np.testing.assert_allclose(np.vstack([formula, carbonate]).T @ species_x * n.sum(), formula.T @ n, rtol=5e-9, atol=1e-14)
    np.testing.assert_allclose(state["basis"]["element_moles"], formula.T @ n, rtol=5e-9, atol=1e-14)
    np.testing.assert_allclose(state["returned_component_moles"], n, rtol=5e-9, atol=0)
    np.testing.assert_allclose(x, n / n.sum(), rtol=5e-9, atol=0)
    co2 = state["component_order"].index("co2")
    assert (formula.T @ n)[elements.index("C")] == n[co2]
    assert (formula.T @ n)[elements.index("S")] == 0
    assert "N" not in elements
    assert state["backend_species"]["order"] == state["component_order"] + ["caco3"]
    assert state["phase_policy"]["oxygen_buffer"] == "None"
    assert not state["phase_policy"]["equilibrated"]
    assert not state["phase_policy"]["stability_checked"]


@pytest.mark.parametrize("record", RECORDS, ids=lambda r: r["id"])
def test_carbon_fixture_preserves_full_potentials_and_actual_derivatives(record):
    state = record["result"]
    n = np.asarray(state["component_moles"])
    present = n > 0
    mu, mu0, mu_rt, mu0_rt, activity, gamma = [
        np.asarray(state[field], float)[present]
        for field in ("mu_J_mol", "mu0_J_mol", "mu_RT", "mu0_RT", "activity", "ln_gamma_common_R")
    ]
    np.testing.assert_allclose(n[present] @ mu, state["gibbs_J"], rtol=5e-9, atol=1e-5)
    rt = state["basis"]["common_R_J_mol_K"] * state["T_K"]
    np.testing.assert_allclose(mu_rt, mu / rt, rtol=5e-12)
    np.testing.assert_allclose(mu0_rt, mu0 / rt, rtol=5e-12)
    np.testing.assert_allclose(mu_rt - mu0_rt, np.log(np.asarray(state["x"])[present]) + gamma, rtol=5e-9)
    np.testing.assert_allclose(np.log(activity), (mu - mu0) / (EVALUATOR["BACKEND_R"] * state["T_K"]), rtol=5e-9)
    for field in ("mu_J_mol", "mu0_J_mol", "activity"):
        assert [v is not None for v in state[field]] == present.tolist()
    assert [d["component"] for d in record["derivatives"]] == [c for c, p in zip(state["component_order"], present) if p]
    np.testing.assert_allclose([d["dG_dn_J_mol"] for d in record["derivatives"]], mu, rtol=0, atol=0.02)
    assert record["max_scaling_mu_error_J_mol"] < 0.002


def test_carbon_reference_pins_a_separate_model_and_exact_zero_control():
    assert REFERENCE["model_id"] == EVALUATOR["MODEL_IDS"][4]
    assert REFERENCE["status"] == "passed_numerical_property_checks"
    zero = RECORDS[0]["result"]
    co2 = zero["component_order"].index("co2")
    assert zero["component_moles"][co2] == zero["backend_species"]["x"][-1] == 0
    assert zero["mu_J_mol"][co2] is None
    for record in RECORDS:
        backend = record["result"]["provenance"]["backend"]
        assert backend["calculation_mode"] == 4
        assert backend["model"] == "rhyolite-MELTS 1.2.0"
        assert backend["runtime_sha256"] == EVALUATOR["REFERENCE"]["backend"]["runtime_sha256"]
