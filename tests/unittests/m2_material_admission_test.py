"""Prevent cross-host, rectangular-domain and concentration-basis admission."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "examples/m2_material/material_admission.py"
SPEC = importlib.util.spec_from_file_location("m2_material_admission", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
assess = MODULE.assess_material_state


def bse():
    ledger = json.loads((PATH.parent / "bse_inventory.json").read_text())
    return dict(zip(ledger["oxide_order"], ledger["oxide_mass_fractions"]))


def host(name):
    data = json.loads((PATH.parent / "hydrogen_reference.json").read_text())
    values = data["chaudhari_2025"]["hosts"][name]["table_1_oxide_weight_percent"]
    total = sum(value for value in values.values() if value is not None)
    return {key: value / total for key, value in values.items() if value is not None}


def test_bse_control_has_quantified_extrapolation_and_no_common_admission():
    result = assess(2173.15, 1e5, silicate_oxide_mass_fractions=bse(),
                    molecular_h2_mass_ppm=0.02, water_mass_percent=0.,
                    hydrogen_partial_pressure_Pa=5e4)
    assert result["nominal_melts"]["inside_nominal_limits"]
    assert not result["nominal_melts"]["establishes_phase_stability"]
    basalt = result["silicate_hydrogen_references"]["basalt"]
    assert basalt["temperature_distance_K"] == 500.
    assert basalt["pressure_envelope_distance_Pa"] == 499900000.
    assert not basalt["reference_conditions_supported"]
    assert basalt["host_comparison"]["candidate_minus_reported_mass_percent"]["MgO"] > 27.
    assert basalt["host_comparison"]["candidate_minus_reported_mass_percent"]["FeO"] > 8.
    assert "Cr2O3" in basalt["host_comparison"]["candidate_nonzero_oxides_not_reported_by_source"]
    overlap = result["implemented_hydrogen_reference_overlap"]
    assert overlap["temperature_overlap_K"] is None
    assert overlap["temperature_gap_K"] == 170.
    assert overlap["total_pressure_gap_Pa"] == 499898675.
    assert result["published_h2_extrapolation"]["predicted_molecular_h2_mass_ppm"] == pytest.approx(.02575)
    assert result["accepted_coupled_material_domain"] is None
    assert result["material_admission"] == "not_established"


def test_matching_a_measured_host_does_not_establish_a_coupled_domain():
    result = assess(1673.15, 1.2e9, silicate_oxide_mass_fractions=host("basalt"),
                    buffer="Fe-FeO-H2O")
    assert result["silicate_hydrogen_references"]["basalt"]["reference_conditions_supported"]
    assert not result["silicate_hydrogen_references"]["andesite"]["reference_conditions_supported"]
    assert result["material_admission"] == "not_established"
    unknown_buffer = assess(1673.15, 1.2e9, silicate_oxide_mass_fractions=host("basalt"))
    assert not unknown_buffer["silicate_hydrogen_references"]["basalt"]["reference_conditions_supported"]


def test_haplogranite_envelope_does_not_create_unmeasured_temperature_pressure_pairs():
    result = assess(1473.15, 3e9, silicate_oxide_mass_fractions=host("haplogranite"),
                    buffer="Fe-FeO-H2O")
    reference = result["silicate_hydrogen_references"]["haplogranite"]
    assert reference["temperature_distance_K"] == 0
    assert reference["pressure_envelope_distance_Pa"] == 0
    assert not reference["reference_temperature_pressure_supported"]
    assert not reference["reference_conditions_supported"]


def test_native_water_is_removed_only_for_dry_comparison_and_h2_uses_total_mass():
    oxides = {name.lower(): value * .98 for name, value in host("basalt").items()}
    oxides["h2o"] = .02
    result = assess(1673.15, 1.2e9, silicate_oxide_mass_fractions=oxides,
                    molecular_h2_mass_ppm=100., water_mass_percent=1.9998,
                    buffer="Fe-FeO-H2O")
    assert result["silicate_hydrogen_references"]["basalt"]["host_comparison"]["same_reported_host"]
    assert result["input"]["silicate_oxide_mass_fractions"]["H2O"] == .02
    with pytest.raises(ValueError, match="denominator"):
        assess(1673.15, 1.2e9, silicate_oxide_mass_fractions=oxides,
               molecular_h2_mass_ppm=100., water_mass_percent=2.)


def test_atomic_alloy_composition_is_compared_on_the_experimental_mass_basis():
    # Four atomic percent Si is only about two mass percent, below Kato's limit.
    result = assess(1873.15, 101325., silicate_oxide_mass_fractions=bse(),
                    alloy_atomic_fractions={"H": .001, "O": 0., "Si": .04, "Fe": .959},
                    hydrogen_partial_pressure_Pa=101325.)
    alloy = result["alloy_hydrogen_reference"]
    assert alloy["reference_conditions_supported"]
    assert 2. < alloy["mass_percent"]["Si"] < 2.5
    expected = 1e6 * .001 * .00100794 / (.959 * .055845 + .04 * .0280855 + .001 * .00100794)
    assert alloy["atomic_h_mass_ppm"] == pytest.approx(expected)
    with_oxygen = assess(1873.15, 101325., silicate_oxide_mass_fractions=bse(),
                         alloy_atomic_fractions=[.95, .04, .009, .001],
                         hydrogen_partial_pressure_Pa=101325.)["alloy_hydrogen_reference"]
    assert not with_oxygen["reference_conditions_supported"]
    assert not with_oxygen["condition_checks"]["oxygen_free"]


def test_one_bar_is_not_the_one_atmosphere_hydrogen_reference():
    alloy = assess(1873.15, 1e5, silicate_oxide_mass_fractions=bse(),
                    alloy_atomic_fractions=[.999, 0., 0., .001],
                    hydrogen_partial_pressure_Pa=1e5)["alloy_hydrogen_reference"]
    assert alloy["total_pressure_distance_Pa"] == 1325.
    assert not alloy["reference_conditions_supported"]


def test_sossi_comparison_uses_feo_total_without_changing_the_input_state():
    measured = {"SiO2": 46.53, "Al2O3": 4.37, "FeO": 8.44, "MgO": 38.05, "CaO": 2.06}
    # Re-express half of the same Fe atoms as Fe2O3, then normalize physical mass.
    measured["Fe2O3"] = measured["FeO"] / 2. * (2 * .055845 + 3 * .0159994) / (2 * (.055845 + .0159994))
    measured["FeO"] /= 2.
    fractions = {name: value / sum(measured.values()) for name, value in measured.items()}
    result = assess(2173.15, 1e5, silicate_oxide_mass_fractions=fractions)
    assert result["sossi_liquid_comparison"]["host_comparison"]["same_reported_host"]
    assert result["input"]["silicate_oxide_mass_fractions"] == fractions
    assert result["accepted_coupled_material_domain"] is None


def test_published_extrapolation_is_not_a_calibration_or_error_bound():
    report = MODULE.published_basalt_h2_extrapolation(1e5, molecular_h2_mass_ppm=.103)
    assert report["predicted_molecular_h2_mass_ppm"] == pytest.approx(.0515)
    assert report["actual_over_extrapolated_concentration"] == pytest.approx(2.)
    assert not report["supports_material_admission"]
    assert not report["is_error_bound"]
    zero = MODULE.published_basalt_h2_extrapolation(0., molecular_h2_mass_ppm=1.)
    assert zero["actual_over_extrapolated_concentration"] is None
    assert zero["zero_reference_concentration"]


@pytest.mark.parametrize("changes", [
    {"temperature_K": np.nan}, {"pressure_Pa": 0.},
    {"silicate_oxide_mass_fractions": {"SiO2": .5}},
    {"silicate_oxide_mass_fractions": {"SiO2": .5, "sio2": .5}},
    {"silicate_oxide_mass_fractions": {"unknown": 1.}},
    {"silicate_oxide_mass_fractions": {"SiO2": .9, "CO2": .1}},
    {"silicate_oxide_mass_fractions": {"H2O": 1.}},
    {"molecular_h2_mass_ppm": 1e6}, {"hydrogen_partial_pressure_Pa": 2e5},
    {"alloy_atomic_fractions": [1., 0., 0., np.nan]},
    {"alloy_atomic_fractions": {"Fe": 1.}},
])
def test_inconsistent_input_cannot_produce_a_material_receipt(changes):
    kwargs = {"temperature_K": 2173.15, "pressure_Pa": 1e5,
              "silicate_oxide_mass_fractions": bse(), **changes}
    with pytest.raises(ValueError):
        assess(**kwargs)
