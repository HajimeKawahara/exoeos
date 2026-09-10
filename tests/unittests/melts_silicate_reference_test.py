"""Offline consistency checks for external data, not native MELTS validation."""

import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest


REFERENCE_DIRECTORY = Path(__file__).resolve().parents[1] / "reference"
REFERENCE = json.loads((REFERENCE_DIRECTORY / "melts_silicate_v1.json").read_text())
GENERATOR = REFERENCE_DIRECTORY / "generate_melts_silicate.py"
STATES = REFERENCE["states"]
OXIDES = REFERENCE["oxide_order"]
COMPONENTS = REFERENCE["component_order"]
NU = np.asarray([
    [REFERENCE["component_oxide_stoichiometry"][component].get(oxide, 0) for oxide in OXIDES]
    for component in COMPONENTS
])
EXPECTED_STATES = {
    "morb_1473K_50MPa": (
        1473.15, 5e7, 0.8191590976360902,
        ["liquid1", "olivine1", "plagioclase1"],
    ),
    "morb_1423K_50MPa": (
        1423.15, 5e7, 0.3832743699736029,
        ["liquid1", "olivine1", "clinopyroxene1", "plagioclase1", "spinel1"],
    ),
    "morb_1473K_500MPa": (
        1473.15, 5e8, 0.3751374496249784,
        ["liquid1", "clinopyroxene1", "clinopyroxene2", "plagioclase1", "spinel1"],
    ),
}


def test_reference_pins_the_backend_basis_and_unmodified_tutorial_input():
    backend = REFERENCE["backend"]
    assert REFERENCE["status"] == "external_reference_only_no_native_silicate_model"
    assert backend["model"] == "rhyolite-MELTS 1.0.2"
    assert backend["calculation_mode"] == 1
    assert backend["release"] == "v2.3.2"
    assert backend["source_commit"] == "9b92b1e0e6d1538a0f498572d6c6651d5d3aca7e"
    assert backend["archive_sha256"] == "97ec2cdb53cae69822a41b5639d8b93edf95b2c361bc15e64e53b192ea9e1425"
    assert REFERENCE["R_J_mol_K"] == 8.3143
    assert OXIDES == [
        "sio2", "tio2", "al2o3", "fe2o3", "cr2o3", "feo", "mno", "mgo",
        "nio", "coo", "cao", "na2o", "k2o", "p2o5", "h2o", "co2", "so3",
        "cl2o-1", "f2o-1",
    ]
    assert COMPONENTS == [
        "sio2", "tio2", "al2o3", "fe2o3", "mgcr2o4", "fe2sio4", "mnsi0.5o2",
        "mg2sio4", "nisi0.5o2", "cosi0.5o2", "casio3", "na2sio3", "kalsio4",
        "ca3(po4)2", "co2", "so3", "cl2o-1", "f2o-1", "h2o",
    ]
    assert REFERENCE["bulk_oxide_mass_g"] == [
        48.68, 1.01, 17.64, 0.89, 0.03, 7.59, 0, 9.10, 0, 0,
        12.45, 2.65, 0.03, 0.08, 0.2, 0, 0, 0, 0,
    ]
    assert REFERENCE["standard_state"]["activity_field"] == "activity (not activity0)"
    policy = REFERENCE["phase_policy"]
    assert (policy["run_mode"], policy["output_flag"], policy["fractionation"]) == (1, 0, False)
    assert policy["explicitly_suppressed_phases"] == []
    assert policy["oxygen_buffer"].startswith("None; closed input oxygen")
    assert {state["id"] for state in STATES} == EXPECTED_STATES.keys()


@pytest.mark.parametrize("state", STATES, ids=lambda state: state["id"])
def test_fixed_assemblage_units_and_full_oxide_mass_closure(state):
    temperature, pressure, liquid_fraction, phases = EXPECTED_STATES[state["id"]]
    assert (state["T_K"], state["P_Pa"]) == (temperature, pressure)
    assert state["backend_T_C"] + 273.15 == temperature
    assert state["backend_P_bar"] * 1e5 == pressure
    assert state["solver_message"] == "Successful run.  No errors."
    assert [phase["name"] for phase in state["phases"]] == phases
    assert state["liquid"]["phase"] == phases[0] == "liquid1"
    masses = np.asarray([phase["mass_g"] for phase in state["phases"]])
    compositions = np.asarray([phase["oxide_wt_percent"] for phase in state["phases"]])
    assert np.all(masses > 0) and np.all(compositions >= 0)
    np.testing.assert_allclose(compositions.sum(axis=1), 100, rtol=0, atol=5e-9)
    np.testing.assert_allclose(masses.sum(), state["bulk_mass_g"], rtol=0, atol=1e-7)
    np.testing.assert_allclose(state["bulk_mass_g"], 100.35, rtol=0, atol=1e-7)
    np.testing.assert_allclose(masses @ compositions / 100, REFERENCE["bulk_oxide_mass_g"], rtol=0, atol=1e-7)
    np.testing.assert_allclose([state["liquid_mass_fraction"], masses[0] / masses.sum()], liquid_fraction, rtol=0, atol=5e-9)


@pytest.mark.parametrize("state", STATES, ids=lambda state: state["id"])
def test_liquid_amounts_follow_component_stoichiometry(state):
    phase, liquid = state["phases"][0], state["liquid"]
    oxide_moles = phase["mass_g"] * np.asarray(phase["oxide_wt_percent"]) / 100
    oxide_moles /= REFERENCE["oxide_molar_masses_g_mol"]
    amounts = np.asarray(liquid["component_moles"])
    assert np.all(amounts >= 0)
    np.testing.assert_allclose(NU.T @ amounts, oxide_moles, rtol=0, atol=5e-9)
    np.testing.assert_allclose(np.linalg.solve(NU.T, oxide_moles), amounts, rtol=0, atol=5e-9)
    np.testing.assert_allclose(amounts / amounts.sum(), liquid["x"], rtol=0, atol=1e-10)
    n, ox = dict(zip(COMPONENTS, amounts)), dict(zip(OXIDES, oxide_moles))
    # These explicit formula checks do not reuse the stored conversion matrix.
    np.testing.assert_allclose(n["fe2sio4"], ox["feo"] / 2, rtol=0, atol=5e-9)
    np.testing.assert_allclose(n["mgcr2o4"], ox["cr2o3"], rtol=0, atol=5e-9)
    np.testing.assert_allclose(n["mg2sio4"], (ox["mgo"] - ox["cr2o3"]) / 2, rtol=0, atol=5e-9)


@pytest.mark.parametrize("state", STATES, ids=lambda state: state["id"])
def test_activities_missing_components_and_gibbs_energy_are_consistent(state):
    liquid = state["liquid"]
    x, amounts = np.asarray(liquid["x"]), np.asarray(liquid["component_moles"])
    present = x > 0
    assert present.sum() == 12
    assert np.all(x[~present] == 0) and np.all(amounts[~present] == 0)
    np.testing.assert_allclose(x.sum(), 1, rtol=0, atol=5e-9)
    for field in ("mu_J_mol", "mu0_J_mol", "activity", "ln_activity", "ln_gamma"):
        assert [value is not None for value in liquid[field]] == present.tolist()
        assert np.all(np.isfinite(np.asarray(liquid[field], dtype=float)[present]))
    mu, mu0, activity, log_a, log_gamma = [
        np.asarray(liquid[field], dtype=float)[present]
        for field in ("mu_J_mol", "mu0_J_mol", "activity", "ln_activity", "ln_gamma")
    ]
    assert np.all(activity > 0)
    rt = REFERENCE["R_J_mol_K"] * state["T_K"]
    np.testing.assert_allclose(mu, mu0 + rt * np.log(activity), rtol=0, atol=1e-4)
    np.testing.assert_allclose(log_a, np.log(activity), rtol=0, atol=5e-9)
    np.testing.assert_allclose(log_a, np.log(x[present]) + log_gamma, rtol=0, atol=5e-9)
    np.testing.assert_allclose(amounts[present] @ mu, state["phases"][0]["gibbs_J"], rtol=0, atol=1e-4)
    np.testing.assert_allclose(x[present] @ log_gamma, liquid["gex_RT"], rtol=0, atol=5e-9)


@pytest.mark.parametrize("state", STATES, ids=lambda state: state["id"])
def test_oxide_potentials_preserve_free_energy_and_balanced_reactions(state):
    liquid = state["liquid"]
    present = np.asarray(liquid["x"]) > 0
    oxide_moles = NU.T @ liquid["component_moles"]
    active_oxides = oxide_moles > 0
    assert [value is not None for value in liquid["oxide_mu_J_mol"]] == active_oxides.tolist()
    mu = np.asarray(liquid["mu_J_mol"], dtype=float)[present]
    oxide_mu = np.asarray(liquid["oxide_mu_J_mol"], dtype=float)[active_oxides]
    np.testing.assert_allclose(NU[np.ix_(present, active_oxides)] @ oxide_mu, mu, rtol=0, atol=1e-4)
    np.testing.assert_allclose(oxide_moles[active_oxides] @ oxide_mu, state["phases"][0]["gibbs_J"], rtol=0, atol=1e-4)
    component = dict(zip(COMPONENTS, liquid["mu_J_mol"]))
    oxide = dict(zip(OXIDES, liquid["oxide_mu_J_mol"]))
    np.testing.assert_allclose(oxide["feo"], (component["fe2sio4"] - component["sio2"]) / 2, rtol=0, atol=1e-4)
    np.testing.assert_allclose(oxide["mgo"], (component["mg2sio4"] - component["sio2"]) / 2, rtol=0, atol=1e-4)
    # Synthetic metal potentials only check basis invariance, not phase equilibrium.
    fe, si, mg = -2e5, -3.5e5, -2.5e5
    oxide_reactions = [
        oxide["sio2"] + 2 * fe - 2 * oxide["feo"] - si,  # 2 FeO + Si -> SiO2 + 2 Fe
        oxide["mgo"] + fe - oxide["feo"] - mg,  # FeO + Mg -> MgO + Fe
    ]
    component_reactions = [
        2 * component["sio2"] - component["fe2sio4"] + 2 * fe - si,
        (component["mg2sio4"] - component["fe2sio4"]) / 2 + fe - mg,
    ]
    np.testing.assert_allclose(component_reactions, oxide_reactions, rtol=0, atol=1e-4)


def test_generator_rejects_modified_runtime_before_importing_melts(tmp_path):
    generator = runpy.run_path(str(GENERATOR))
    assert generator["RUNTIME_HASHES"] == REFERENCE["backend"]["runtime_sha256"]
    first_runtime_file = next(iter(generator["RUNTIME_HASHES"]))
    (tmp_path / first_runtime_file).write_text("raise RuntimeError('must not be imported')\n")
    with pytest.raises(ValueError, match="Runtime hash mismatch"):
        generator["check_runtime"](tmp_path)


def test_generator_help_needs_no_external_runtime():
    result = subprocess.run([sys.executable, str(GENERATOR), "--help"], check=True, capture_output=True, text=True)
    assert "--runtime" in result.stdout and "--check" in result.stdout
