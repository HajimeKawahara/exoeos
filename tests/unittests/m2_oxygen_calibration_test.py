"""Verify the Henry conversion and measured O standard independently."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "examples/m2_material/oxygen_calibration.py"
SPEC = importlib.util.spec_from_file_location("m2_oxygen_calibration", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
scenario = MODULE.oxygen_standard_scenario


def standards():
    return dict(oxygen_standard_rt=-8., oxygen_lngamma_infinite_dilution=2.,
                hydrogen_gas_standard_rt=-3., water_gas_standard_rt=-10.)


@pytest.mark.parametrize("reference,a,b", [("sakao1959", 7040., -3.224),
                                          ("matoba1965", 7480., -3.421)])
def test_target_standard_reproduces_equilibrium_in_the_dilute_mass_basis(reference, a, b):
    t = 1873.15
    supplied = standards()
    result = scenario(t, 101325., **supplied, reference=reference)
    # Work directly in grams, moles, and gas fugacity ratio; not the helper's
    # conversion constant. Finite 1e-10 g O makes the limiting error negligible.
    n_o, n_fe = 1e-10/15.999, (100.-1e-10)/55.845
    x_o = n_o/(n_o+n_fe)
    mu_o = (supplied["oxygen_standard_rt"]+result["standard_offsets_rt"]["O_metal"]
            +supplied["oxygen_lngamma_infinite_dilution"]+np.log(x_o))
    ratio = 10.**(a/t+b)*1e-10
    reaction = supplied["water_gas_standard_rt"]+np.log(ratio)-supplied["hydrogen_gas_standard_rt"]-mu_o
    assert reaction == pytest.approx(0., abs=3e-12)
    assert result["accepted_coupled_material_domain"] is None
    assert not result["baseline_replaced"]


def test_offset_is_invariant_to_consistent_element_gauge_and_alloy_convention():
    original = standards()
    baseline = scenario(1873.15, 101325., **original)
    changed = dict(original)
    # Common elemental O shift; H2O includes one O, metal O includes one O.
    changed["oxygen_standard_rt"] += 17.
    changed["water_gas_standard_rt"] += 17.
    # Separate H gauge cancels the two gas standards (two H atoms each).
    changed["hydrogen_gas_standard_rt"] -= 13.
    changed["water_gas_standard_rt"] -= 13.
    # Moving an arbitrary constant from gamma into mu0 leaves total mu fixed.
    changed["oxygen_standard_rt"] += 5.
    changed["oxygen_lngamma_infinite_dilution"] -= 5.
    result = scenario(1873.15, 101325., **changed)
    assert result["standard_offsets_rt"] == pytest.approx(baseline["standard_offsets_rt"])


def test_temperature_and_pressure_transfer_are_distinct_and_never_admitted():
    with pytest.raises(ValueError, match="explicit opt-in"):
        scenario(2173.15, 267.2e5, **standards())
    result = scenario(2173.15, 267.2e5, **standards(), allow_temperature_extrapolation=True)
    assert result["temperature_distance_K"] == pytest.approx(249.)
    assert result["pressure_distance_Pa"] == 267.2e5-101325.
    assert result["pressure_transfer_error_bound"] is None
    assert result["FeSiOH_composition_transfer_error_bound"] is None
    assert result["accepted_coupled_material_domain"] is None


@pytest.mark.parametrize("key,value", [("oxygen_standard_rt", True), ("oxygen_standard_rt", [1.]),
                                       ("oxygen_lngamma_infinite_dilution", np.inf),
                                       ("water_gas_standard_rt", "0")])
def test_invalid_scalar_contracts_are_rejected(key, value):
    inputs = standards()
    inputs[key] = value
    with pytest.raises(ValueError, match="finite real scalar"):
        scenario(1873.15, 101325., **inputs)


def test_raw_table_and_source_standard_replay_with_independent_binary_ma_formula():
    report = MODULE.calibration_report()
    meta = json.loads(MODULE.PROVENANCE_PATH.read_text())["inherited_standard"]
    assert report["observed_sample_count"] == 23
    assert report["refitted_parameter_count"] == 0
    for row in report["observations"]:
        t = row["T_K"]
        # Independent g=H-TS evaluation, using polynomial sums instead of
        # the runner's expression. No provider activity helper is used here.
        mu = {}
        u = t/1000.
        for name, (dh, a, b, c, d, e, f, g, h) in meta["shomate"].items():
            enthalpy = dh+f-h-e/u+sum(co*u**(i+1)/(i+1) for i, co in enumerate((a,b,c,d)))
            entropy = g+a*np.log(u)-e/(2*u*u)+sum(co*u**i/i for i, co in enumerate((b,c,d), 1))
            mu[name] = (1000*enthalpy-t*entropy)/(meta["source"]["gas_constant_J_mol_K"]*t)
        x = row["O_mole_fraction"]
        # Binary Ma derivative is ln gamma_O = -q(1-x), q=-16500/T.
        # Combined source-standard/source-gamma is therefore the following.
        source_o = -meta["source"]["log10_to_ln"]*(2.736-11439/t)+mu["FeO_silicate"]-mu["Fe_metal"]
        logratio = (source_o+4.29-16500/t-16500*x/t+np.log(x)+mu["H2_gas"]-mu["H2O_gas"])/np.log(10.)
        assert row["predicted_log10_gas_ratio"]["adopted_Ma"] == pytest.approx(logratio, abs=2e-13)
    summary = report["summary"]
    assert summary["adopted_Ma"]["rmse_log10_gas_ratio"] > .9
    assert summary["sakao1959_standard_only_Ma"]["rmse_log10_gas_ratio"] < .008
    assert summary["sakao1959_standard_only_Ma"]["max_absolute_residual_log10_gas_ratio"] < .019
    assert report["material_admission"] == "adopted_oxygen_reaction_reference_failed"


def test_alternative_actual_standards_require_provenance_and_replay_independently():
    with pytest.raises(ValueError, match="nonempty provenance"):
        MODULE.calibration_report(standards_provider=lambda t: standards())
    def alternate(t):
        return {**standards(), "oxygen_lngamma_infinite_dilution": 16500./t}
    report = MODULE.calibration_report(standards_provider=alternate,
                                       standard_provenance={"kind": "test_independent_standard_set"})
    assert report["standard_provenance"]["kind"] == "test_independent_standard_set"
    # Changing the baseline cancels in the alternative-standard replay.
    assert report["summary"]["sakao1959_standard_only_Ma"]["rmse_log10_gas_ratio"] < .008
