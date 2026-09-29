"""Reconstruct Sossi's water reference with its shared background uncertainty.

This example uses H2O-equivalent mass ppm directly, never pooled OH columns.
It does not replace MELTS, infer a redox Gibbs scalar from Eq.11, or calibrate
molecular H2 dissolution. Run with ``python -m examples.m2_material.
sossi_water_calibration --output NEW_FILE``.
"""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .material_admission import assess_material_state, _fugacity_coefficient


DATA_DIR = Path(__file__).with_name("water_data")
DATA_PATH = DATA_DIR / "Sossi2023_Table1.csv"
PROVENANCE_PATH = DATA_DIR / "sossi2023_provenance.json"
EXCLUDED = ("Per-1", "Per-2", "Per-5")
BASELINE = "Per-5"
BRANCHES = {"basalt_epsilon_6.3": "63", "peridotite_epsilon_5.1": "51"}


def _observations():
    metadata = json.loads(PROVENANCE_PATH.read_text())
    if hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() != metadata["table_sha256"]:
        raise ValueError("Sossi Table 1 data hash mismatch.")
    with DATA_PATH.open(newline="") as stream:
        return [{key: value if key == "sample" else float(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]


def _design(rows):
    return np.sqrt([[row["f_H2O_bar"], row["f_H2_bar"]] for row in rows])


def _fit_operator(rows, indices):
    """Linear response of coefficients to raw data, including the shared blank.

    The design is fixed at the paper's calculated fugacities. Covariances
    for those predictors are unavailable. Equal-weight least squares uses
    the published corrected column for the point estimate; this operator
    is its linear response to unrounded raw concentration measurements.
    """
    design = _design([rows[i] for i in indices])
    inverse = np.linalg.pinv(design)
    if np.linalg.matrix_rank(design) != 2:
        raise ValueError("Both fugacity coefficients must be identifiable.")
    correction = np.zeros((len(indices), len(rows)))
    correction[np.arange(len(indices)), indices] = 1.
    correction[:, next(i for i, row in enumerate(rows) if row["sample"] == BASELINE)] -= 1.
    return inverse, inverse @ correction


def _branch_report(rows, suffix):
    indices = [i for i, row in enumerate(rows) if row["sample"] not in EXCLUDED]
    y = np.array([row[f"water_corrected_{suffix}_ppm"] for row in rows])
    sigma = np.array([row[f"water_raw_{suffix}_sigma_ppm"] for row in rows])
    inverse, operator = _fit_operator(rows, indices)
    coefficients = inverse @ y[indices]
    design = _design(rows)
    residual = design @ coefficients - y
    groups = {}
    for i in indices:
        key = (rows[i]["f_H2O_bar"], rows[i]["f_H2_bar"])
        groups.setdefault(key, []).append(i)
    folds = []
    for held in groups.values():
        training = [i for i in indices if i not in held]
        fold_inverse, _ = _fit_operator(rows, training)
        fitted = fold_inverse @ y[training]
        predictions = design[held] @ fitted
        folds.append({"held_out_samples": [rows[i]["sample"] for i in held],
                      "training_samples": [rows[i]["sample"] for i in training],
                      "coefficients_ppmw_per_sqrt_bar": fitted.tolist(),
                      "predicted_water_mass_ppm": predictions.tolist(),
                      "residual_ppm": (predictions - y[held]).tolist()})
    held_residuals = np.concatenate([fold["residual_ppm"] for fold in folds])
    blank = next(i for i, row in enumerate(rows) if row["sample"] == BASELINE)
    raw = np.array([row[f"water_raw_{suffix}_ppm"] for row in rows])
    return {
        "coefficient_order": ["sqrt_f_H2O_bar", "sqrt_f_H2_bar"],
        "coefficients_ppmw_per_sqrt_bar": coefficients.tolist(),
        "coefficient_table_rounding_offset_ppmw_per_sqrt_bar": (coefficients-operator@raw).tolist(),
        "raw_measurement_coefficient_operator": operator.tolist(),
        "raw_sample_order": [row["sample"] for row in rows],
        "coefficient_sd_upper_ppmw_per_sqrt_bar": (np.abs(operator) @ sigma).tolist(),
        "coefficient_covariance_if_raw_errors_independent": ((operator * sigma**2) @ operator.T).tolist(),
        "independence_is_established": False,
        "baseline_sample": BASELINE,
        "baseline_water_mass_ppm": float(raw[blank]),
        "baseline_reported_sigma_ppm": float(sigma[blank]),
        "max_table_rounding_difference_ppm": float(np.max(np.abs(y - (raw-raw[blank])))),
        "fitted_sample_count": len(indices),
        "fit_rmse_ppm": float(np.sqrt(np.mean(residual[indices]**2))),
        "fit_max_absolute_residual_ppm": float(np.max(np.abs(residual[indices]))),
        "grouped_cross_validation": {
            "method": "Leave one distinct calculated gas-fugacity pair out; all three time-series replicates stay together.",
            "conditional_on_shared_baseline": BASELINE,
            "scope": "Internal cross-validation of the fixed Eq.11 functional form, conditional on the common baseline and spectroscopy; not independent external validation.",
            "group_count": len(folds), "sample_count": len(indices),
            "rmse_ppm": float(np.sqrt(np.mean(held_residuals**2))),
            "max_absolute_residual_ppm": float(np.max(np.abs(held_residuals))),
            "folds": folds},
        "all_observation_residuals": [{"sample": row["sample"],
                                       "included_in_fit": i in indices,
                                       "predicted_water_mass_ppm": float(design[i] @ coefficients),
                                       "observed_corrected_water_mass_ppm": float(y[i]),
                                       "residual_ppm": float(residual[i])}
                                      for i, row in enumerate(rows)],
    }


def calibration_report():
    """Re-fit the 11 paper-selected samples and audit both IR alternatives.

    Reported raw SDs include shared spectroscopy errors with unknown joint
    covariance. For every linear statistic L y, sd(L y) <= sum |L_i| sd(y_i)
    for any covariance with these marginals. This bound includes the shared
    subtraction; it is not a confidence interval or extrapolation bound.
    """
    rows = _observations()
    return {"report_kind": "sossi2023_shared_baseline_calibration_v1",
            "observed_sample_count": len(rows), "fit_excluded_samples": list(EXCLUDED),
            "source_doi": "10.1016/j.epsl.2022.117894",
            "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            "provenance_sha256": hashlib.sha256(PROVENANCE_PATH.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "coefficient_fit": "Equal-weight linear least squares on published rounded corrected Table 1 concentrations; a fresh reconstruction, not the authors' unrounded regression.",
            "uncertainty_scope": "Concentration measurement marginals with fixed predictors and selected Per-5 background. Unknown correlations are not set to zero in the SD upper bound. Fugacity, temperature, composition, baseline-selection bias and transfer errors remain unbounded.",
            "concentration_basis": "H2O-equivalent mass / glass mass; H2-induced OH is included, dissolved molecular H2 is not measured by this band.",
            "branches_are_independent_observations": False,
            "branches": {name: _branch_report(rows, suffix) for name, suffix in BRANCHES.items()},
            "accepted_coupled_material_domain": None,
            "constitutive_model_changed": False}


def _predictor_support(rows, target):
    """Find a convex combination in the two fitted sqrt-fugacity coordinates.

    Caratheodory's theorem needs at most three points in 2-D. The tiny
    experimental table does not require a new optimization dependency.
    Geometric inclusion does not establish accuracy between experiments.
    """
    fitted = [row for row in rows if row["sample"] not in EXCLUDED]
    design = _design(fitted)
    for indices in itertools.combinations(range(len(fitted)), 3):
        matrix = np.vstack((design[list(indices)].T, np.ones(3)))
        if np.linalg.matrix_rank(matrix) < 3:
            continue
        weights = np.linalg.solve(matrix, np.r_[target, 1.])
        if np.all(weights >= -1e-12):
            return {"inside_fitted_predictor_hull": True,
                    "samples": [fitted[i]["sample"] for i in indices],
                    "weights": weights.tolist()}
    return {"inside_fitted_predictor_hull": False, "samples": [], "weights": []}


def assess_sossi_water_state(temperature_K, pressure_Pa, *, silicate_oxide_mass_fractions,
                             water_partial_pressure_Pa, hydrogen_partial_pressure_Pa,
                             molecular_h2_mass_ppm=None, water_mass_percent=None,
                             dissolved_helium_mass_ppm=0.0,
                             water_fugacity_coefficient=1.0, hydrogen_fugacity_coefficient=1.0):
    """Compare a supplied native host with the measured 1-bar water reference.

    Partial pressures use the complete gas denominator. Fugacity equals the
    partial pressure times its supplied coefficient (one by default).
    Native oxides include H2O but exclude added H2/He.
    H2/He ppm and water percent, when supplied, use complete-liquid mass.
    Predictions retain the experiment's H2O-equivalent/glass basis. Their
    optional complete-liquid conversion adds the supplied H2/He masses to
    the denominator, not fitted solubility effects. No host/T/P extrapolation is
    accepted, including evaluations inside the two-predictor convex hull.
    """
    values = (water_partial_pressure_Pa, hydrogen_partial_pressure_Pa)
    if any(isinstance(v, (bool, str)) or not np.isscalar(v) or not np.isreal(v)
           or not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("Supply finite nonnegative gas partial pressures in Pa.")
    coefficients = (_fugacity_coefficient(water_fugacity_coefficient, "phi_H2O"),
                    _fugacity_coefficient(hydrogen_fugacity_coefficient, "phi_H2"))
    fugacities = np.asarray(values, dtype=float) * np.asarray(coefficients)
    if not np.all(np.isfinite(fugacities)):
        raise ValueError("The supplied partial pressures and coefficients require finite fugacities.")
    reference = assess_material_state(
        temperature_K, pressure_Pa, silicate_oxide_mass_fractions=silicate_oxide_mass_fractions,
        molecular_h2_mass_ppm=molecular_h2_mass_ppm, water_mass_percent=water_mass_percent,
        dissolved_helium_mass_ppm=dissolved_helium_mass_ppm,
        hydrogen_partial_pressure_Pa=hydrogen_partial_pressure_Pa,
        hydrogen_fugacity_coefficient=coefficients[1])
    pressure, temperature = reference["input"]["pressure_Pa"], reference["input"]["temperature_K"]
    if sum(values) > pressure * (1 + 1e-12):
        raise ValueError("H2 and H2O partial pressures cannot exceed total pressure.")
    query = np.sqrt(fugacities/1e5)
    rows, report = _observations(), calibration_report()
    support = _predictor_support(rows, query)
    host = reference["sossi_liquid_comparison"]["host_comparison"]
    reasons = []
    if pressure != 1e5:
        reasons.append("total_pressure_differs_from_measured_1_bar")
    if temperature != 2173.:
        reasons.append("temperature_differs_from_nominal_2173_K_no_temperature_coefficient_fitted")
    if not host["same_reported_host"]:
        reasons.append("host_differs_from_reported_glass_mean_no_composition_radius_fitted")
    if not support["inside_fitted_predictor_hull"]:
        reasons.append("outside_joint_sqrt_fugacity_predictor_hull")
    helium = reference["input"]["dissolved_helium_mass_ppm"]
    dilution = None if molecular_h2_mass_ppm is None else 1 - (molecular_h2_mass_ppm + helium) * 1e-6
    actual = None if water_mass_percent is None else water_mass_percent * 1e4
    predictions = {}
    for name, branch in report["branches"].items():
        suffix = BRANCHES[name]
        sigma = np.array([row[f"water_raw_{suffix}_sigma_ppm"] for row in rows])
        value = float(query @ branch["coefficients_ppmw_per_sqrt_bar"])
        linear = query @ np.asarray(branch["raw_measurement_coefficient_operator"])
        sd_upper = float(np.abs(linear) @ sigma)
        predictions[name] = {
            "water_equivalent_mass_ppm_native_host": value,
            "concentration_parameter_sd_upper_ppm_native_host": sd_upper,
            "water_equivalent_mass_ppm_complete_liquid": None if dilution is None else value * dilution,
            "actual_water_mass_ppm_complete_liquid": actual,
            "actual_minus_reference_ppm_complete_liquid": None if dilution is None or actual is None else actual-value*dilution,
            "internal_grouped_cv_rmse_ppm": branch["grouped_cross_validation"]["rmse_ppm"],
            "internal_grouped_cv_max_absolute_residual_ppm": branch["grouped_cross_validation"]["max_absolute_residual_ppm"],
            "prediction_error_bound": None}
    result = {"report_kind": "actual_state_sossi_water_comparison_v1",
            "temperature_K": temperature, "pressure_Pa": pressure,
            "gas_partial_pressures_Pa": {"H2O": float(values[0]), "H2": float(values[1])},
            "gas_fugacity_assumption": "Ideal gas: supplied partial pressures equal fugacities.",
            "predictor_support": support, "host_comparison": host,
            "coordinate_extrapolation_reasons": reasons,
            "evaluation_scope": "explicit_extrapolation" if reasons else "reported_reference_coordinates",
            "nominal_reference_temperature_K": 2173., "reference_total_pressure_Pa": 1e5,
            "composition_transfer_error_bound": None, "pressure_transfer_error_bound": None,
            "temperature_transfer_error_bound": None, "supports_material_admission": False,
            "accepted_coupled_material_domain": None,
            "concentration_basis": report["concentration_basis"],
            "complete_liquid_conversion": "Native-host ppm times (1 - supplied molecular-H2 mass fraction - supplied dissolved-He mass fraction); arithmetic dilution only, not a constitutive correction.",
            "dissolved_helium_mass_ppm": helium,
            "native_host_mass_fraction_of_complete_liquid": dilution,
            "uncertainty_scope": report["uncertainty_scope"],
            "branches": predictions,
            "data_sha256": report["data_sha256"],
            "provenance_sha256": report["provenance_sha256"],
            "runner_sha256": report["runner_sha256"],
            "material_reference_provenance": reference["provenance"]}
    if coefficients != (1.0, 1.0):
        result.update(
            gas_fugacity_coefficients=dict(zip(("H2O", "H2"), coefficients)),
            gas_fugacities_Pa=dict(zip(("H2O", "H2"), fugacities.tolist())),
            gas_fugacity_assumption="Supplied fugacity coefficients multiply the unchanged full-gas partial pressures.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        stream.write(json.dumps(calibration_report(), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
