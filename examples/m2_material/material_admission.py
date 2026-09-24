"""Assess an actual M2 material state against source-specific evidence.

This diagnostic does not change any constitutive law or accept a coupled
physical domain. Experimental conditions, model extrapolations, and nominal
software limits remain separate. Concentrations use the complete phase mass;
the dry silicate comparison removes H2 and water before normalization.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DATA = Path(__file__).resolve().parent
ATM_PA = 101325.0
ALLOY_COMPONENTS = ("Fe", "Si", "O", "H")
ALLOY_MOLAR_MASSES = np.array([0.055845, 0.0280855, 0.0159994, 0.00100794])
OXIDES = ("SiO2", "TiO2", "Al2O3", "Cr2O3", "FeO", "Fe2O3", "MgO", "CaO",
          "Na2O", "K2O", "P2O5", "MnO", "NiO", "CoO", "H2O", "CO2")


def _nonnegative(value, name):
    if not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite, real and nonnegative.")
    return float(value)


def _distance(value, bounds):
    """Return an absolute coordinate distance, never an uncertainty score."""
    return float(max(bounds[0] - value, value - bounds[1], 0.0))


def _composition_comparison(candidate, measured, *, explicit_zero=()):
    """Compare reported dry oxides without inventing detection limits.

    Reported means are normalized by their sum, not the rounded Table total.
    Unreported oxides remain unreported; Fe-free synthesis separately supports
    the named explicit zeros. Numerical identity is not an empirical tolerance.
    """
    reported = {name: value for name, value in measured.items() if value is not None}
    scale = sum(reported.values())
    reference = {name: value / scale for name, value in reported.items()}
    reference.update({name: 0.0 for name in explicit_zero})
    difference = {name: 100.0 * (candidate.get(name, 0.0) - value)
                  for name, value in reference.items()}
    unknown = {name: value for name, value in candidate.items()
               if name not in reference and value > 0}
    identical = (not unknown and all(abs(value) <= 1e-10 for value in difference.values()))
    return {"reported_dry_oxide_mass_fraction": reference,
            "candidate_minus_reported_mass_percent": difference,
            "candidate_nonzero_oxides_not_reported_by_source": unknown,
            "same_reported_host": bool(identical),
            "interpretation": "Descriptive differences; no compositional validity radius or unreported detection limit is assumed."}


def published_basalt_h2_extrapolation(hydrogen_partial_pressure_Pa, *,
                                      molecular_h2_mass_ppm=None):
    """Reproduce the authors' 515 ppm/GPa pure-H2 illustration at ideal p_H2.

    Chaudhari et al. (2025), pp. 14--16, convert the buffered basalt fit to
    a pure-H2 illustrative law. Applying it to p_H2 in a mixture additionally
    assumes ideal gas fugacity and Henry proportionality. It has no measured
    low-pressure/BSE calibration or quantified extrapolation error. It is
    neither an upper bound nor a replacement standard for the current model.
    """
    partial = _nonnegative(hydrogen_partial_pressure_Pa, "p_H2")
    predicted = 515.0 * partial / 1e9
    actual = None if molecular_h2_mass_ppm is None else _nonnegative(molecular_h2_mass_ppm, "H2 ppm")
    return {
        "source_doi": "10.1007/s00410-025-02272-y",
        "source_locator": "Printed pages 14--16, low-pressure discussion and illustrative magma-ocean calculation",
        "kind": "published_extrapolation_with_ideal_gas_extension",
        "pure_h2_slope_mass_ppm_per_GPa": 515.0,
        "hydrogen_partial_pressure_Pa": partial,
        "predicted_molecular_h2_mass_ppm": predicted,
        "actual_over_extrapolated_concentration": None if actual is None or predicted == 0 else actual / predicted,
        "zero_reference_concentration": predicted == 0,
        "is_error_bound": False,
        "supports_material_admission": False,
        "assumptions": ["Ideal H2 fugacity equals its partial pressure.",
                        "The authors' basalt-based illustrative coefficient is transferred without a temperature or host correction."],
        "unresolved": ["BSE host dependence", "Temperature dependence", "Low-pressure calibration", "Extrapolation uncertainty"],
    }


def assess_material_state(temperature_K, pressure_Pa, *, silicate_oxide_mass_fractions,
                          molecular_h2_mass_ppm=None, water_mass_percent=None,
                          alloy_atomic_fractions=None,
                          hydrogen_partial_pressure_Pa=None, buffer=None):
    """Return material evidence for a supplied state, without mutating it.

    Supply native silicate oxides as a normalized name-to-mass-fraction mapping
    (including native water, excluding the added molecular-H2 contribution),
    H2 mass ppm and native-water mass percent relative to the complete liquid,
    and optional alloy atomic fractions as a mapping or in Fe, Si, O, H order.
    Oxide names accept chemical case or lowercase native MELTS labels. ``None`` denotes
    an absent/unsupplied alloy, not a zero-composition calibration. This example
    owns no equilibrium or planetary inventory calculation.
    """
    temperature = _nonnegative(temperature_K, "T")
    pressure = _nonnegative(pressure_Pa, "P")
    if temperature == 0 or pressure == 0:
        raise ValueError("T and P must be strictly positive.")
    aliases = {name.lower(): name for name in OXIDES}
    native_oxides = {}
    for name, value in silicate_oxide_mass_fractions.items():
        if not isinstance(name, str) or name.lower() not in aliases:
            raise ValueError(f"Unknown oxide label: {name}.")
        canonical = aliases[name.lower()]
        if canonical in native_oxides:
            raise ValueError(f"Duplicate oxide alias: {name}.")
        native_oxides[canonical] = _nonnegative(value, name)
    if not native_oxides or not np.isclose(sum(native_oxides.values()), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("Native oxide mass fractions must sum to one; no input is renormalized.")
    if native_oxides.get("CO2", 0.0) > 0:
        raise ValueError("This M2 assessment requires the declared carbon-free host.")
    dry_fraction = sum(value for name, value in native_oxides.items() if name not in ("H2O", "CO2"))
    if dry_fraction <= 0:
        raise ValueError("A positive dry silicate host mass is required.")
    oxides = {name: value / dry_fraction for name, value in native_oxides.items() if name not in ("H2O", "CO2")}
    h2 = None if molecular_h2_mass_ppm is None else _nonnegative(molecular_h2_mass_ppm, "H2 ppm")
    water = None if water_mass_percent is None else _nonnegative(water_mass_percent, "H2O mass percent")
    if (0.0 if h2 is None else h2 * 1e-6) + (0.0 if water is None else water / 100.0) >= 1.0:
        raise ValueError("A positive dry host mass must remain after H2 and water.")
    if h2 is not None and water is not None:
        expected_water = 100.0 * native_oxides.get("H2O", 0.0) * (1.0 - h2 * 1e-6)
        if not np.isclose(water, expected_water, rtol=1e-10, atol=1e-10):
            raise ValueError("Native-water and H2 concentrations must use the same complete-liquid mass denominator.")
    partial = None if hydrogen_partial_pressure_Pa is None else _nonnegative(hydrogen_partial_pressure_Pa, "p_H2")
    if partial is not None and partial > pressure:
        raise ValueError("H2 partial pressure cannot exceed total pressure.")
    evidence = json.loads((DATA / "domain_evidence.json").read_text())["evidence"]
    references = json.loads((DATA / "hydrogen_reference.json").read_text())
    chaudhari = references["chaudhari_2025"]
    observations = [row for row in chaudhari["observations"]
                    if not row["excluded_from_further_analysis_by_authors"]]
    silicate = {}
    for host, model in chaudhari["hosts"].items():
        rows = [row for row in observations if row["host"] == host]
        temperatures = sorted(set(row["temperature_K"] for row in rows))
        pressures = sorted(set(row["pressure_Pa"] for row in rows))
        pairs = sorted(set((row["temperature_K"], row["pressure_Pa"]) for row in rows))
        # Retain the existing evaluator's source-specific operational domain.
        supported_tp = ((temperature, pressure) in pairs if host == "haplogranite" else
                        temperature in temperatures and pressures[0] <= pressure <= pressures[-1])
        composition = _composition_comparison(oxides, model["table_1_oxide_weight_percent"],
                                              explicit_zero=("FeO", "Fe2O3"))
        h2_range = [min(row["H2_mass_ppm"] for row in rows), max(row["H2_mass_ppm"] for row in rows)]
        water_range = [min(row["H2O_total_weight_percent"] for row in rows),
                       max(row["H2O_total_weight_percent"] for row in rows)]
        silicate[host] = {
            "source_doi": chaudhari["doi"],
            "measured_temperature_pressure_pairs_K_Pa": pairs,
            "temperature_distance_K": min(abs(temperature - value) for value in temperatures),
            "pressure_envelope_distance_Pa": _distance(pressure, [pressures[0], pressures[-1]]),
            "reference_temperature_pressure_supported": bool(supported_tp),
            "required_buffer_matches": buffer == model["buffer"],
            "host_comparison": composition,
            "observed_molecular_h2_mass_ppm_range": h2_range,
            "molecular_h2_distance_from_observed_range_ppm": None if h2 is None else _distance(h2, h2_range),
            "observed_native_water_mass_percent_range": water_range,
            "native_water_distance_from_observed_range_mass_percent": None if water is None else _distance(water, water_range),
            "reference_conditions_supported": bool(supported_tp and buffer == model["buffer"]
                                                   and composition["same_reported_host"]),
            "concentration_range_meaning": "Observed envelopes across the named host's runs; not an independent concentration calibration rectangle.",
        }

    kato = references["kato_1970"]
    alloy = {"status": "not_supplied", "reference_conditions_supported": False,
             "interpretation": "Metal absence does not calibrate an incipient alloy or its activity model."}
    if alloy_atomic_fractions is not None:
        if isinstance(alloy_atomic_fractions, dict):
            if set(alloy_atomic_fractions) != set(ALLOY_COMPONENTS):
                raise ValueError("Supply exactly Fe, Si, O and H alloy atomic fractions.")
            alloy_atomic_fractions = [alloy_atomic_fractions[name] for name in ALLOY_COMPONENTS]
        x = np.asarray(alloy_atomic_fractions)
        if (x.shape != (4,) or not np.isrealobj(x) or not np.all(np.isfinite(x))
                or np.any(x < 0) or not np.isclose(x.sum(), 1.0, rtol=0.0, atol=1e-10)):
            raise ValueError("Alloy atomic fractions must be finite, nonnegative and sum to one in Fe, Si, O, H order.")
        mass = x * ALLOY_MOLAR_MASSES
        wt_percent = mass / mass.sum() * 100.0
        fe_si = x[1] > 0
        temperature_bounds = (kato["silicon_interaction"]["combined_pure_iron_regression_temperature_K"] if fe_si
                              else kato["pure_iron"]["converted_temperature_K"])
        conditions = {
            "temperature": _distance(temperature, temperature_bounds) == 0,
            "pure_h2_reference_total_pressure": pressure == ATM_PA,
            "pure_h2_reference_partial_pressure": partial == ATM_PA,
            "oxygen_free": x[2] == 0,
            "silicon_below_2_5_mass_percent": wt_percent[1] < 2.5,
            "positive_iron_host": x[0] > 0,
        }
        alloy = {"status": "evaluated", "source_doi": kato["doi"],
                 "component_order": list(ALLOY_COMPONENTS), "atomic_fractions": x.tolist(),
                 "mass_percent": dict(zip(ALLOY_COMPONENTS, wt_percent.tolist())),
                 "atomic_h_mass_ppm": float(wt_percent[3] * 1e4),
                 "reference_temperature_K": temperature_bounds,
                 "temperature_distance_K": _distance(temperature, temperature_bounds),
                 "total_pressure_distance_Pa": abs(pressure - ATM_PA),
                 "hydrogen_partial_pressure_distance_Pa": None if partial is None else abs(partial - ATM_PA),
                 "condition_checks": {name: bool(value) for name, value in conditions.items()},
                 "reference_conditions_supported": bool(all(conditions.values())),
                 "finite_concentration_calibration": "Not established: the reference is a dilute-H law, with no measured O-H or finite-H interaction domain."}
        dilute_values = [1e4 * 10**(-1874.0 / value - 1.601 - .033 * wt_percent[1])
                         for value in temperature_bounds]
        alloy["dilute_regression_atomic_h_mass_ppm_range"] = dilute_values
        alloy["atomic_h_distance_from_dilute_regression_range_ppm"] = _distance(
            alloy["atomic_h_mass_ppm"], dilute_values)
        alloy["concentration_range_meaning"] = (
            "Range of the one-atmosphere reference fit over its reported temperature interval, "
            "not a measured validity limit for finite-H interactions. Values are not used for admission.")
        alloy["reference_equilibrium_atomic_h_mass_ppm"] = (
            float(1e4 * 10**(-1874.0 / temperature - 1.601 - .033 * wt_percent[1]))
            if all(conditions.values()) else None)
    melts = evidence["melts_guidance"]
    sossi = evidence["sossi_2020"]
    sossi_oxides = dict(oxides)
    # Report all Fe as FeO, matching the experiment's FeO-total convention.
    fe2o3_to_feo = 2 * (ALLOY_MOLAR_MASSES[0] + ALLOY_MOLAR_MASSES[2]) / (
        2 * ALLOY_MOLAR_MASSES[0] + 3 * ALLOY_MOLAR_MASSES[2])
    sossi_oxides["FeO"] = sossi_oxides.get("FeO", 0.0) + fe2o3_to_feo * sossi_oxides.pop("Fe2O3", 0.0)
    sossi_oxides = {name: value / sum(sossi_oxides.values()) for name, value in sossi_oxides.items()}
    sossi_reference = {("FeO" if name == "FeO_total" else name): value
                       for name, value in sossi["average_glass_oxide_weight_percent"].items()}
    max_silicate_temperature = max(row["temperature_K"] for row in observations)
    min_alloy_temperature = kato["pure_iron"]["converted_temperature_K"][0]
    overlap = [max(min(row["temperature_K"] for row in observations), min_alloy_temperature),
               min(max_silicate_temperature, kato["pure_iron"]["converted_temperature_K"][1])]
    input_state = {"temperature_K": temperature, "pressure_Pa": pressure,
                   "silicate_oxide_mass_fractions": native_oxides,
                   "dry_silicate_oxide_mass_fractions": oxides,
                   "molecular_h2_mass_ppm": h2, "water_mass_percent": water,
                   "hydrogen_partial_pressure_Pa": partial, "buffer": buffer}
    return {
        "report_kind": "actual_state_material_evidence",
        "input": input_state,
        "accepted_coupled_material_domain": None,
        "material_admission": "not_established",
        "nominal_melts": {
            "inside_nominal_limits": (_distance(temperature, melts["temperature_K"]) == 0
                                      and _distance(pressure, melts["pressure_Pa"]) == 0),
            "temperature_distance_K": _distance(temperature, melts["temperature_K"]),
            "pressure_distance_Pa": _distance(pressure, melts["pressure_Pa"]),
            "establishes_phase_stability": False},
        "sossi_liquid_comparison": {
            "source_doi": sossi["doi"],
            "temperature_distance_from_reported_band_K": _distance(temperature, sossi["temperature_K"]),
            "pressure_distance_Pa": _distance(pressure, sossi["pressure_Pa"]),
            "host_comparison": _composition_comparison(sossi_oxides, sossi_reference),
            "iron_reporting": "All candidate Fe is expressed as FeO-equivalent before dry renormalization for this comparison only; the supplied physical state is unchanged.",
            "interpretation": "The reported temperature uncertainty is not a continuous calibration domain; exact BSE stability needs its own phase assessment."},
        "silicate_hydrogen_references": silicate,
        "alloy_hydrogen_reference": alloy,
        "implemented_hydrogen_reference_overlap": {
            "sources": [chaudhari["doi"], kato["doi"]],
            "temperature_overlap_K": overlap if overlap[0] <= overlap[1] else None,
            "temperature_gap_K": max(0.0, min_alloy_temperature - max_silicate_temperature),
            "total_pressure_gap_Pa": min(row["pressure_Pa"] for row in observations) - ATM_PA,
            "interpretation": "Necessary-condition intersection of these two implemented reference families only; not proof that other physical domains cannot exist."},
        "published_h2_extrapolation": None if partial is None else published_basalt_h2_extrapolation(
            partial, molecular_h2_mass_ppm=h2),
        "unresolved": ["No common calibrated BSE/silicate-H2/Fe-Si-O-H domain is established.",
                       "Composition differences and source extrapolations have no quantified validity radius or error bound.",
                       "Phase stability, reaction standards and omitted-transfer errors require independent assessments."],
        "provenance": {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
                       for name in ("domain_evidence.json", "hydrogen_reference.json", "material_admission.py")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON object with assess_material_state keyword arguments.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = assess_material_state(**json.loads(args.input.read_text()))
    report["input_file_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
