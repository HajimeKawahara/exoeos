"""Shared experimental background, grouped prediction and actual-state scope."""

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
s = importlib.import_module("examples.m2_material.sossi_water_calibration")


def state_arguments():
    mean = {"SiO2": 46.53, "Al2O3": 4.37, "FeO": 8.44, "MgO": 38.05, "CaO": 2.06}
    host = {name: value / sum(mean.values()) * .9999 for name, value in mean.items()}
    host["H2O"] = .0001
    return dict(temperature_K=2173., pressure_Pa=1e5,
                silicate_oxide_mass_fractions=host,
                water_partial_pressure_Pa=2700., hydrogen_partial_pressure_Pa=1200.,
                molecular_h2_mass_ppm=20., water_mass_percent=.01*(1-20e-6))


def test_primary_table_reconstruction_keeps_missing_background_and_rounding():
    report = s.calibration_report()
    assert report["observed_sample_count"] == 14
    for name, expected in (("basalt_epsilon_6.3", [526.411312, 182.991425]),
                           ("peridotite_epsilon_5.1", [649.955945, 226.290858])):
        branch = report["branches"][name]
        assert branch["coefficients_ppmw_per_sqrt_bar"] == pytest.approx(expected)
        assert branch["fitted_sample_count"] == 11
        assert branch["baseline_sample"] == "Per-5"
        assert "Per-4" in branch["raw_sample_order"]
    assert report["branches"]["peridotite_epsilon_5.1"]["max_table_rounding_difference_ppm"] == pytest.approx(.1)
    assert report["accepted_coupled_material_domain"] is None
    assert report["branches_are_independent_observations"] is False


def test_operator_propagates_one_shared_background_not_eleven_independent_copies():
    rows = s._observations()
    branch = s.calibration_report()["branches"]["basalt_epsilon_6.3"]
    included = [i for i, row in enumerate(rows) if row["sample"] not in s.EXCLUDED]
    matrix = np.array(branch["raw_measurement_coefficient_operator"])
    baseline = branch["raw_sample_order"].index("Per-5")
    # A constant shift to all raw measurements disappears on subtraction.
    np.testing.assert_allclose(matrix @ np.ones(14), 0., atol=1e-14)
    raw = np.array([row["water_raw_63_ppm"] for row in rows])
    # Published corrected/raw columns have separately rounded values.
    np.testing.assert_allclose(matrix @ raw + branch["coefficient_table_rounding_offset_ppmw_per_sqrt_bar"],
                               branch["coefficients_ppmw_per_sqrt_bar"])
    delta = .01
    changed = raw.copy()
    changed[baseline] += delta
    direct = np.linalg.lstsq(s._design([rows[i] for i in included]),
                            changed[included]-changed[baseline], rcond=None)[0]
    np.testing.assert_allclose(direct-matrix@raw, matrix[:, baseline]*delta, atol=1e-12)
    # The corrected observations share positive covariance sigma_blank^2.
    sigma = np.array([rows[i]["water_raw_63_sigma_ppm"] for i in included])
    covariance = np.diag(sigma**2) + rows[baseline]["water_raw_63_sigma_ppm"]**2
    inverse = np.linalg.pinv(s._design([rows[i] for i in included]))
    np.testing.assert_allclose(inverse @ covariance @ inverse.T,
                               branch["coefficient_covariance_if_raw_errors_independent"])


def test_sd_bound_is_attainable_for_permitted_correlations():
    rows = s._observations()
    branch = s.calibration_report()["branches"]["peridotite_epsilon_5.1"]
    sigma = np.array([row["water_raw_51_sigma_ppm"] for row in rows])
    for operator, upper in zip(branch["raw_measurement_coefficient_operator"],
                               branch["coefficient_sd_upper_ppmw_per_sqrt_bar"]):
        operator = np.asarray(operator)
        direction = np.where(operator >= 0, 1., -1.) * sigma
        covariance = np.outer(direction, direction)
        np.testing.assert_allclose(np.diag(covariance), sigma**2)
        assert np.sqrt(operator @ covariance @ operator) == pytest.approx(upper)


def test_grouped_cross_validation_keeps_time_series_together_and_out_of_fit():
    report = s.calibration_report()["branches"]["basalt_epsilon_6.3"]["grouped_cross_validation"]
    assert report["group_count"] == 9
    assert report["sample_count"] == 11
    time_series = {"Per-7", "Per-TS1", "Per-TS2"}
    all_held = []
    for fold in report["folds"]:
        held, training = set(fold["held_out_samples"]), set(fold["training_samples"])
        assert held.isdisjoint(training)
        if held & time_series:
            assert held == time_series
            assert training.isdisjoint(time_series)
        all_held.extend(held)
    assert len(set(all_held)) == 11
    assert report["rmse_ppm"] == pytest.approx(4.23419489)


def test_actual_state_separates_mass_conversion_from_calibration():
    kwargs = state_arguments()
    result = s.assess_sossi_water_state(**kwargs)
    assert result["predictor_support"]["inside_fitted_predictor_hull"] is True
    assert result["evaluation_scope"] == "reported_reference_coordinates"
    assert result["supports_material_admission"] is False
    for branch in result["branches"].values():
        native = branch["water_equivalent_mass_ppm_native_host"]
        assert branch["water_equivalent_mass_ppm_complete_liquid"] == pytest.approx(native*(1-20e-6))
        assert branch["actual_water_mass_ppm_complete_liquid"] == pytest.approx(100*(1-20e-6))
        assert branch["prediction_error_bound"] is None
    kwargs["pressure_Pa"] = 267e5
    kwargs["temperature_K"] = 2173.15
    result = s.assess_sossi_water_state(**kwargs)
    assert "total_pressure_differs_from_measured_1_bar" in result["coordinate_extrapolation_reasons"]
    assert result["evaluation_scope"] == "explicit_extrapolation"
    # A matching predictor pair cannot conceal a different total pressure.
    assert result["predictor_support"]["inside_fitted_predictor_hull"] is True


def test_joint_predictor_hull_is_stricter_than_two_independent_ranges():
    kwargs = state_arguments()
    # Each coordinate is experimentally attained, but their joint maximum is not.
    kwargs["water_partial_pressure_Pa"] = 2700
    kwargs["hydrogen_partial_pressure_Pa"] = 6400
    result = s.assess_sossi_water_state(**kwargs)
    assert result["predictor_support"]["inside_fitted_predictor_hull"] is False
    assert "outside_joint_sqrt_fugacity_predictor_hull" in result["coordinate_extrapolation_reasons"]


@pytest.mark.parametrize("key,value", [("water_partial_pressure_Pa", np.nan),
                                        ("hydrogen_partial_pressure_Pa", -1.),
                                        ("water_partial_pressure_Pa", True),
                                        ("hydrogen_partial_pressure_Pa", 1e5),
                                        ("water_mass_percent", .01)])
def test_inconsistent_state_inputs_fail(key, value):
    kwargs = state_arguments()
    kwargs[key] = value
    with pytest.raises(ValueError):
        s.assess_sossi_water_state(**kwargs)


def test_data_tampering_fails_before_fit(tmp_path, monkeypatch):
    changed = tmp_path / "table.csv"
    changed.write_bytes(s.DATA_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(s, "DATA_PATH", changed)
    with pytest.raises(ValueError, match="hash mismatch"):
        s.calibration_report()


def test_missing_mass_inputs_are_not_silently_zero():
    kwargs = state_arguments()
    kwargs.pop("water_mass_percent")
    kwargs.pop("molecular_h2_mass_ppm")
    result = s.assess_sossi_water_state(**kwargs)
    for branch in result["branches"].values():
        assert branch["water_equivalent_mass_ppm_complete_liquid"] is None
        assert branch["actual_minus_reference_ppm_complete_liquid"] is None
    json.dumps(result, allow_nan=False)


def test_sossi_complete_liquid_conversion_includes_independent_helium_mass():
    kwargs = state_arguments()
    # Hold 100 kg native host (including .01 kg water), then add .1/.2 kg H2/He.
    total = 100.3
    kwargs.update(molecular_h2_mass_ppm=.1 / total * 1e6,
                  dissolved_helium_mass_ppm=.2 / total * 1e6,
                  water_mass_percent=.01 / total * 100.)
    result = s.assess_sossi_water_state(**kwargs)
    assert result["native_host_mass_fraction_of_complete_liquid"] == pytest.approx(100. / total)
    for branch in result["branches"].values():
        assert branch["water_equivalent_mass_ppm_complete_liquid"] == pytest.approx(
            branch["water_equivalent_mass_ppm_native_host"] * 100. / total)
        assert branch["actual_water_mass_ppm_complete_liquid"] == pytest.approx(.01 / total * 1e6)
        assert branch["prediction_error_bound"] is None
    assert result["supports_material_admission"] is False
    assert result["accepted_coupled_material_domain"] is None
    kwargs.pop("dissolved_helium_mass_ppm")
    with pytest.raises(ValueError, match="complete-liquid"):
        s.assess_sossi_water_state(**kwargs)


def test_known_helium_does_not_fill_an_unsupplied_hydrogen_mass():
    kwargs = state_arguments()
    kwargs.pop("molecular_h2_mass_ppm")
    kwargs["dissolved_helium_mass_ppm"] = 100.
    result = s.assess_sossi_water_state(**kwargs)
    assert result["native_host_mass_fraction_of_complete_liquid"] is None
    assert all(row["water_equivalent_mass_ppm_complete_liquid"] is None
               for row in result["branches"].values())


def test_declared_fugacities_preserve_partial_pressures_and_control_the_predictors():
    kwargs = state_arguments()
    ideal = s.assess_sossi_water_state(**kwargs)
    assert ideal == s.assess_sossi_water_state(**kwargs, water_fugacity_coefficient=1.,
                                              hydrogen_fugacity_coefficient=1.)
    coefficients = {"water_fugacity_coefficient": 1.25, "hydrogen_fugacity_coefficient": .8}
    nonideal = s.assess_sossi_water_state(**kwargs, **coefficients)
    same_fugacity = {**kwargs, "water_partial_pressure_Pa": 1.25*kwargs["water_partial_pressure_Pa"],
                     "hydrogen_partial_pressure_Pa": .8*kwargs["hydrogen_partial_pressure_Pa"]}
    equivalent = s.assess_sossi_water_state(**same_fugacity)
    assert nonideal["gas_partial_pressures_Pa"] == ideal["gas_partial_pressures_Pa"]
    assert nonideal["gas_fugacities_Pa"] == equivalent["gas_partial_pressures_Pa"]
    assert nonideal["branches"] == equivalent["branches"]
    assert nonideal["predictor_support"] == equivalent["predictor_support"]
    assert nonideal["gas_fugacity_coefficients"] == {"H2O": 1.25, "H2": .8}
    assert nonideal["accepted_coupled_material_domain"] is None
    # Fugacity need not be bounded by total pressure; only partial pressures are.
    large = s.assess_sossi_water_state(**kwargs, water_fugacity_coefficient=100.)
    assert large["gas_fugacities_Pa"]["H2O"] > kwargs["pressure_Pa"]


@pytest.mark.parametrize("coefficient", [0., -1., np.nan, np.inf, True, "1", None])
@pytest.mark.parametrize("name", ["water_fugacity_coefficient", "hydrogen_fugacity_coefficient"])
def test_invalid_fugacity_coefficients_do_not_produce_sossi_comparisons(name, coefficient):
    with pytest.raises(ValueError):
        s.assess_sossi_water_state(**state_arguments(), **{name: coefficient})
