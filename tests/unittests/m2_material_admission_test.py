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


def test_constitutive_choices_do_not_misuse_legacy_reference_checks_as_acceptance():
    inputs = dict(silicate_oxide_mass_fractions=bse(),
                  alloy_atomic_fractions=[.97, .001, .004, .024, .001, 0., 0., 0., 0., 0., 0.])
    legacy = assess(2173.15, 2.7e7, **inputs)
    selected = assess(2173.15, 2.7e7, **inputs, liquid_model="published_water",
                      metal_model="associated_k", hydrogen_oxygen_model="schenck1961_abstract")
    for key in ("silicate_hydrogen_references", "alloy_hydrogen_reference", "input",
                "implemented_hydrogen_reference_overlap"):
        assert selected[key] == legacy[key]
    assert not selected["reference_assessment_scope"]["reference_conditions_are_constitutive_acceptance"]
    assert selected["accepted_coupled_material_domain"] is None
    evidence = selected["constitutive_evidence"]
    assert evidence["selected_models"]["liquid_model"] == "published_water"
    assert evidence["actual_comparison_coordinates"]["alloy_mass_percent"] == selected["alloy_hydrogen_reference"]["mass_percent"]
    assert legacy["constitutive_evidence"]["selected_models"]["metal_model"] is None
    categories = set(evidence["categories"])
    assert len(evidence["systems"]) == 6
    assert all(categories <= set(system) for system in evidence["systems"].values())
    omission = evidence["additional_omitted_transfer_paths"]
    assert set(omission["paths"]) == {"Na_to_metal", "He_to_silicate", "He_to_metal"}
    assert not omission["all_13_element_transfer_coverage_established"]
    for invalid in ({"liquid_model": "unknown"},
                    {"metal_model": "ma", "hydrogen_oxygen_model": "schenck1961_abstract"},
                    {"metal_model": "associated"}):
        with pytest.raises(ValueError):
            assess(2173.15, 2.7e7, **inputs, **invalid)


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


def test_phosphorus_uses_the_complete_mass_basis_without_four_component_admission():
    fractions = {"Fe": .95, "Si": .04, "O": 0., "H": .001, "P": .009}
    alloy = assess(1873.15, 101325., silicate_oxide_mass_fractions=bse(),
                   alloy_atomic_fractions=fractions,
                   hydrogen_partial_pressure_Pa=101325.)["alloy_hydrogen_reference"]
    denominator = (.95 * .055845 + .04 * .0280855 + .001 * .00100794
                   + .009 * .030973761998)
    assert alloy["atomic_h_mass_ppm"] == pytest.approx(1e6 * .001 * .00100794 / denominator)
    assert alloy["mass_percent"]["P"] == pytest.approx(100 * .009 * .030973761998 / denominator)
    assert not alloy["condition_checks"]["phosphorus_free"]
    assert not alloy["reference_conditions_supported"]
    zero = assess(1873.15, 101325., silicate_oxide_mass_fractions=bse(),
                  alloy_atomic_fractions=[.959, .04, 0., .001, 0.],
                  hydrogen_partial_pressure_Pa=101325.)["alloy_hydrogen_reference"]
    assert zero["reference_conditions_supported"]
    assert zero["condition_checks"]["phosphorus_free"]


def test_associated_metal_atomic_recount_keeps_all_ten_elements_in_mass_basis():
    fractions = dict(zip(MODULE.EXTENDED_ALLOY_COMPONENTS,
                         [.918, .001, .01, .03, .001, .001, .001, .001, .036, .001]))
    x = np.array(list(fractions.values()))
    assert x.sum() == pytest.approx(1.)
    alloy = assess(1873.15, 101325., silicate_oxide_mass_fractions=bse(),
                   alloy_atomic_fractions=fractions,
                   hydrogen_partial_pressure_Pa=101325.)["alloy_hydrogen_reference"]
    assert alloy["mass_percent"]["Cr"] == pytest.approx(
        100*.036*.0519961/(x@MODULE.EXTENDED_ALLOY_MOLAR_MASSES))
    assert alloy["atomic_h_mass_ppm"] == pytest.approx(
        1e6*.03*.00100794/(x@MODULE.EXTENDED_ALLOY_MOLAR_MASSES))
    assert not alloy["condition_checks"]["other_metal_solutes_free"]
    assert not alloy["reference_conditions_supported"]


def test_potassium_sensitivity_uses_the_full_eleven_element_mass_denominator():
    x = np.array([.96, .001, .002, .02, .001, .001, .001, .001, .002, .001, .01])
    fractions = dict(zip(MODULE.POTASSIUM_ALLOY_COMPONENTS, x))
    assert x.sum() == pytest.approx(1.)
    alloy = assess(1873.15, 101325., silicate_oxide_mass_fractions=bse(),
                   alloy_atomic_fractions=fractions,
                   hydrogen_partial_pressure_Pa=101325.)["alloy_hydrogen_reference"]
    denominator = x @ MODULE.POTASSIUM_ALLOY_MOLAR_MASSES
    assert alloy["mass_percent"]["K"] == pytest.approx(100*.01*.0390983/denominator)
    assert alloy["atomic_h_mass_ppm"] == pytest.approx(1e6*.02*.00100794/denominator)
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


def test_sodium_counts_in_complete_alloy_mass_without_automatic_admission():
    x=np.array([.958,.001,.004,.024,.001,0.,0.,0.,.001,0.,.001,.01])
    assert x.sum()==pytest.approx(1.)
    inputs=dict(silicate_oxide_mass_fractions=bse(),metal_model='associated_k_na',
                hydrogen_oxygen_model='schenck1961_abstract')
    result=assess(2173.15,2.7e7,alloy_atomic_fractions=x,**inputs)
    mapped=assess(2173.15,2.7e7,alloy_atomic_fractions=dict(zip(MODULE.SODIUM_ALLOY_COMPONENTS,x)),**inputs)
    assert result['alloy_hydrogen_reference']==mapped['alloy_hydrogen_reference']
    alloy=result['alloy_hydrogen_reference']
    expected=100*x*MODULE.SODIUM_ALLOY_MOLAR_MASSES/(x@MODULE.SODIUM_ALLOY_MOLAR_MASSES)
    assert alloy['mass_percent']['Na']==pytest.approx(expected[-1])
    assert alloy['atomic_h_mass_ppm']==pytest.approx(expected[3]*1e4)
    assert not alloy['condition_checks']['other_metal_solutes_free']
    assert result['material_admission']=='not_established'
    assert result['accepted_coupled_material_domain'] is None
    path=result['constitutive_evidence']['additional_omitted_transfer_paths']['paths']['Na_to_metal']
    assert path['conditional_model']['model_selector']=='associated_k_na'
    assert not path['conditional_model']['empirical_BSE_calibration']
    with pytest.raises(ValueError,match='every alloy element'):
        assess(2173.15,2.7e7,alloy_atomic_fractions=np.r_[1.,np.zeros(10)],**inputs)
