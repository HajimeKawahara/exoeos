"""Replay Fe-O gas equilibria and construct explicit alternative O standards.

The measured Henry mass-percent reference constrains a reaction standard,
not a coupled BSE material domain. The helper never changes a provider model.
Run ``python -m examples.m2_material.oxygen_calibration --output NEW_FILE``.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).with_name("steel_data")
PROVENANCE_PATH = DATA_DIR / "oxygen_provenance.json"
DATA_PATH = DATA_DIR / "Sakao1959_Table1.csv"
LN10 = np.log(10.)
# This converts x_O to wt% O at infinite dilution in pure liquid Fe.
MASS_PERCENT_PER_MOLE_FRACTION = 100. * 15.999 / 55.845
REFERENCES = {
    "sakao1959": (7040., -3.224, -1750., .76, (1824.15, 1873.15, 1924.15)),
    "matoba1965": (7480., -3.421, -10130., 4.94, (1823.15, 1880.15, 1936.15)),
}


def _finite_scalar(value, name, positive=False):
    if (isinstance(value, (bool, str)) or not np.isscalar(value)
            or not np.isreal(value) or not np.isfinite(value)
            or (positive and value <= 0)):
        raise ValueError(f"{name} must be a finite real scalar" + (" > 0." if positive else "."))
    return float(value)


def oxygen_reference(temperature_K, *, reference="sakao1959", allow_temperature_extrapolation=False):
    """Return the published dilute-Fe reaction fit, with explicit T support.

    Interpolation within the measured temperature envelope is permitted.
    Unmeasured high pressure and FeSiOH compositions are never certified.
    """
    t = _finite_scalar(temperature_K, "T_K", positive=True)
    if reference not in REFERENCES:
        raise ValueError("Unknown oxygen reference.")
    if not isinstance(allow_temperature_extrapolation, bool):
        raise ValueError("allow_temperature_extrapolation must be bool.")
    a, b, c, d, measured = REFERENCES[reference]
    inside = min(measured) <= t <= max(measured)
    if not inside and not allow_temperature_extrapolation:
        raise ValueError("Temperature extrapolation requires explicit opt-in.")
    return {"reference": reference, "T_K": t, "log10_K": a/t+b,
            "log10_f_O_per_mass_percent": c/t+d,
            "measured_temperatures_K": list(measured),
            "inside_measured_temperature_envelope": inside,
            "temperature_distance_K": max(min(measured)-t, t-max(measured), 0.),
            "temperature_transfer_error_bound": None}


def oxygen_standard_scenario(temperature_K, pressure_Pa, *, oxygen_standard_rt,
                            oxygen_lngamma_infinite_dilution,
                            hydrogen_gas_standard_rt, water_gas_standard_rt,
                            reference="sakao1959", allow_temperature_extrapolation=False):
    """Return an additive O-only standard scenario in the supplied alloy basis.

    Supply mu0/(RT) and ln(gamma_O) at x_Fe=1 in the *same* mole-fraction
    convention. Gas standards must share R,T and a common standard pressure.
    At infinite dilution, wt%O/xO = 100 M_O/M_Fe, hence
    mu_O,wt% = mu_O,x + ln(gamma_O,infinity) - ln(100 M_O/M_Fe).
    The measured K=(f_H2O/f_H2)/a_O fixes this Henry standard exactly.

    Linear O standard offsets preserve the supplied excess Gibbs scalar.
    Their use at the requested P or in finite FeSiOH is a conditional
    scenario, not an empirical high-pressure/composition correction. A
    temperature-dependent callback must recompute the offset at each T;
    a fixed-temperature consumer can use standard_offsets_rt directly.
    """
    ref = oxygen_reference(temperature_K, reference=reference,
                           allow_temperature_extrapolation=allow_temperature_extrapolation)
    p = _finite_scalar(pressure_Pa, "P_Pa", positive=True)
    names = ("oxygen_standard_rt", "oxygen_lngamma_infinite_dilution",
             "hydrogen_gas_standard_rt", "water_gas_standard_rt")
    o, gamma, h2, water = [_finite_scalar(v, name) for v, name in zip(
        (oxygen_standard_rt, oxygen_lngamma_infinite_dilution,
         hydrogen_gas_standard_rt, water_gas_standard_rt), names)]
    adopted_henry = o + gamma - np.log(MASS_PERCENT_PER_MOLE_FRACTION)
    target_henry = water - h2 + LN10*ref["log10_K"]
    offset = target_henry-adopted_henry
    return {"role": "explicit_measured_O_standard_scenario", **ref, "P_Pa": p,
            "reaction": "O(metal) + H2(g) = H2O(g)",
            "concentration_basis": "Henry mass percent O in pure liquid Fe at infinite dilution",
            "input_standards_rt": dict(zip(names, (o, gamma, h2, water))),
            "mass_percent_per_O_mole_fraction_at_infinite_dilution": MASS_PERCENT_PER_MOLE_FRACTION,
            "adopted_Henry_O_standard_rt": float(adopted_henry),
            "target_Henry_O_standard_rt": float(target_henry),
            "adopted_log10_K": float((adopted_henry+h2-water)/LN10),
            "adopted_minus_reference_log10_K": float((adopted_henry+h2-water)/LN10-ref["log10_K"]),
            "standard_offsets_rt": {"O_metal": float(offset)},
            "target_O_standard_rt": float(o+offset),
            "reference_total_pressure_Pa": 101325.,
            "pressure_distance_Pa": abs(p-101325.),
            "pressure_transfer_error_bound": None,
            "FeSiOH_composition_transfer_error_bound": None,
            "published_fit_covariance": None,
            "excess_Gibbs_model_changed": False,
            "baseline_replaced": False,
            "accepted_coupled_material_domain": None}


def _inherited_standards(temperature_K):
    """Reconstruct four pinned source rows for an offline reference replay."""
    from exoeos.ma_fe_si_o import MaFeSiOLiquid
    from exoeos.gibbs_excess import solution_state

    meta = json.loads(PROVENANCE_PATH.read_text())["inherited_standard"]
    t = temperature_K/1000.
    rt = meta["source"]["gas_constant_J_mol_K"]*temperature_K
    mu = {}
    for name, (dh, a, b, c, d, e, f, g, h) in meta["shomate"].items():
        enthalpy = dh+a*t+b*t*t/2+c*t**3/3+d*t**4/4-e/t+f-h
        entropy = a*np.log(t)+b*t+c*t*t/2+d*t**3/3-e/(2*t*t)+g
        mu[name] = (1000.*enthalpy-temperature_K*entropy)/rt
    model = MaFeSiOLiquid()
    source_o = (-meta["source"]["log10_to_ln"]*(2.736-11439./temperature_K)
                +mu["FeO_silicate"]-mu["Fe_metal"])
    return {"oxygen_standard_rt": float(source_o+model.standard_state_shift_RT(temperature_K)[2]),
            "oxygen_lngamma_infinite_dilution": float(solution_state(
                model, temperature_K, 101325., np.array([1., 0., 0.])).log_activity_coefficients[2]),
            "hydrogen_gas_standard_rt": mu["H2_gas"], "water_gas_standard_rt": mu["H2O_gas"]}


def calibration_report(*, standards_provider=None, standard_provenance=None):
    """Replay 23 observations using pinned or explicitly supplied standards.

    An optional callback returns the four oxygen_standard_scenario standard
    arguments at each T. A nonempty provenance mapping must accompany it.
    This permits an offline comparison of the caller's actual gas standards
    without importing that consumer or changing the measured reference.
    """
    import jax
    import inspect
    from exoeos.ma_fe_si_o import MaFeSiOLiquid
    from exoeos.gibbs_excess import solution_state

    if not jax.config.x64_enabled:
        raise ValueError("Set JAX_ENABLE_X64=1 for the archived standard replay.")
    if standards_provider is None:
        standards_provider = _inherited_standards
        standard_provenance = {"kind": "pinned_Young_source_Shomate_and_adopted_Ma"}
    elif not callable(standards_provider) or not isinstance(standard_provenance, dict) or not standard_provenance:
        raise ValueError("A supplied standard callback requires nonempty provenance.")
    meta = json.loads(PROVENANCE_PATH.read_text())
    if hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() != meta["sakao1959"]["data_sha256"]:
        raise ValueError("Sakao Table 1 data hash mismatch.")
    with DATA_PATH.open(newline="") as stream:
        observations = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    rows = []
    model = MaFeSiOLiquid()
    for observation in observations:
        t = observation["temperature_C"]+273.15
        w = observation["O_mass_percent"]
        amounts = np.array([(100.-w)/55.845, 0., w/15.999])
        x = amounts/amounts.sum()
        standards = standards_provider(t)
        gamma = float(solution_state(model, t, 101325., x).log_activity_coefficients[2])
        baseline = (standards["oxygen_standard_rt"]+gamma+np.log(x[2])
                    +standards["hydrogen_gas_standard_rt"]-standards["water_gas_standard_rt"])/LN10
        observed = np.log10(observation["corrected_H2O_H2_ratio"])
        predictions = {"adopted_Ma": baseline}
        for reference in REFERENCES:
            scenario = oxygen_standard_scenario(t, 101325., **standards, reference=reference)
            fit = oxygen_reference(t, reference=reference)
            predictions[reference+"_published_finite_O"] = fit["log10_K"]+np.log10(w)+fit["log10_f_O_per_mass_percent"]*w
            predictions[reference+"_standard_only_Ma"] = baseline+scenario["standard_offsets_rt"]["O_metal"]/LN10
        rows.append({"observation": observation, "T_K": t, "O_mole_fraction": float(x[2]),
                     "observed_log10_gas_ratio": float(observed),
                     "predicted_log10_gas_ratio": predictions,
                     "residual_log10_gas_ratio": {k: float(v-observed) for k, v in predictions.items()}})
    names = rows[0]["residual_log10_gas_ratio"]
    residuals = {name: np.array([row["residual_log10_gas_ratio"][name] for row in rows]) for name in names}
    scenarios = [oxygen_standard_scenario(t, 101325., **standards_provider(t),
                                          reference=reference, allow_temperature_extrapolation=True)
                 for t in (1873.15, 2173.15) for reference in REFERENCES]
    return {"report_kind": "measured_oxygen_standard_replay_v1",
            "standard_provenance": standard_provenance,
            "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            "provenance_sha256": hashlib.sha256(PROVENANCE_PATH.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "activity_provider_sha256": hashlib.sha256(Path(inspect.getfile(MaFeSiOLiquid)).read_bytes()).hexdigest(),
            "execution": {"numpy": np.__version__, "jax": jax.__version__, "jax_enable_x64": True},
            "observed_sample_count": len(rows), "refitted_parameter_count": 0,
            "summary": {name: {"rmse_log10_gas_ratio": float(np.sqrt(np.mean(values**2))),
                               "max_absolute_residual_log10_gas_ratio": float(np.max(np.abs(values)))}
                        for name, values in residuals.items()},
            "observations": rows, "standard_scenarios": scenarios,
            "finding": "The inherited O standard plus adopted Ma activities does not reproduce the measured low-pressure Fe-O/H2/H2O equilibrium. Replacing only its O standard substantially reduces the measured residual; finite-composition and transfer errors remain.",
            "uncertainty_scope": "Residuals and alternative published fits are descriptive, not statistical or extrapolation bounds. Matoba supplies an independent experiment's fit, not digitized independent raw observations.",
            "material_admission": "adopted_oxygen_reaction_reference_failed",
            "accepted_coupled_material_domain": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = calibration_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
