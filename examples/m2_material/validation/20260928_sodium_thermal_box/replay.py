"""Separate fixed-activity thermal continuation of primary Na limits."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


MATERIAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MATERIAL))
from sodium_fixed_box import partition_sodium

baseline_path = MATERIAL / "validation/20260928_sodium_fixed_box/replay.py"
spec = importlib.util.spec_from_file_location("fixed_na_box", baseline_path)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def assess(raw):
    fixed = baseline.assess(raw)
    target_temperature = fixed["source"]["temperature_K"]
    inverse_temperature_coefficient_K = -14340.
    rows = []
    for primary in fixed["rows"]:
        observation = primary["primary_observation"]
        initial_temperature = observation["temperature_K"]
        log_factor = math.log(10.) * inverse_temperature_coefficient_K * (
            1. / target_temperature - 1. / initial_temperature)
        factor = math.exp(log_factor)
        trials = {}
        for name, old in primary["fixed_box_trials"].items():
            d = old["inputs"]["metal_silicate_mass_fraction_D"] * factor
            result = partition_sodium(**fixed["fixed_box_inputs"],
                                       metal_silicate_mass_fraction_D=d)
            result["fraction_of_global_Na_in_metal"] = result["sodium_mol"]["metal"] / fixed["global_Na_inventory_mol"]
            result["D_critical_over_declared_D"] = fixed["target"]["D_critical"] / d
            result["below_one_percent_global_Na_target"] = result["fraction_of_global_Na_in_metal"] < .01
            trials[name] = result
        rows.append(dict(primary_run=observation["run"], reference_temperature_K=initial_temperature,
                         reference_pressure_GPa=observation["pressure_GPa"],
                         metal_Si_wt_percent=observation["metal_Si_wt_percent"],
                         carbon_mole_fraction_calculated=observation["carbon_mole_fraction_calculated"],
                         log_K_multiplier=log_factor, K_multiplier=factor,
                         D_multiplier_assumed_equal_to_K_multiplier=factor,
                         thermal_fixed_box_trials=trials))
    files = [Path(__file__), baseline_path, MATERIAL / "sodium_fixed_box.py",
             MATERIAL / "sodium_reference.py", MATERIAL / "sodium_reference_sources.json"]
    return dict(assessment_id="fixed_activity_thermal_Na_partition_diagnostic_v1",
                original_source=fixed["source"], global_Na_inventory_mol=fixed["global_Na_inventory_mol"],
                fixed_other_phase_Na_mol=fixed["fixed_other_phase_Na_mol"],
                fixed_box_inputs=fixed["fixed_box_inputs"], engineering_target=fixed["target"],
                temperature_target_K=target_temperature,
                adopted_log10_K_inverse_T_coefficient_K=inverse_temperature_coefficient_K,
                rows=rows,
                continuation_assumptions=[
                    "Transfer the FeS/basalt central inverse-T slope to each S-free reference; this transfer is a declared scenario, not a measured S-free slope.",
                    "Freeze the concentration-to-D molar conversion, oxide Fe/Na activities and metal activity ratio; only K changes with temperature.",
                    "A change of the infinite-dilution activity convention must change K consistently. No independent measured gammaNa_infinite(T) is inferred.",
                    "S=0 removes the published Na-S analogy term identically, even with epsilon(T) proportional to 1/T.",
                    "Na-Si and Na-C thermal effects are unmeasured here and are held fixed, not measured to be zero; do not insert the K-Si coefficient for Na.",
                    "Pressure and phase compositions are held invariant in the partition-law continuation; finite redistribution still uses the exact Na-inclusive mass denominators.",
                    "The fit intercept cancels in K(T)/K(Tref); parameter errors and covariance do not provide a joint extrapolation bound.",
                ],
                supports_material_admission=False, is_BSE_error_bound=False,
                pressure_closure_performed=False, coupled_source_equilibrium_performed=False,
                interpretation="Compare the engineering screen under two explicit continuations. Exceeding the target here is not a prediction of the full BSE equilibrium.",
                input_and_code_sha256={str(p.relative_to(MATERIAL)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.source.read_bytes())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
