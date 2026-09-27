"""Reconstruct a published H2 absorption calibration as an explicit scenario.

This does not change the adopted H2 standard or fit a new BSE solubility law.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


SOURCE = Path(__file__).with_name("water_data") / "Chaudhari2025_absorption_sources.json"


def absorption_scenario_report(base_scenario):
    """Return a finite dilute-Henry amplitude scenario, preserving other keys.

    Beer-Lambert concentrations at fixed absorption and matrix scale as
    1/epsilon. The same factor scales the *dilute* mole-fraction Henry
    coefficient. Applying that constant at finite concentrations is an
    explicit model continuation, not an exact remapping of a finite glass.
    """
    source = json.loads(SOURCE.read_text())
    old, new = source["old_extinction_L_mol_cm"], source["new_extinction_L_mol_cm"]
    error = source["reported_new_extinction_plus_minus"]
    shift = math.log(new/old)
    scenario = deepcopy(base_scenario)
    offsets = scenario.setdefault("standard_offsets_rt", {})
    if offsets.get("H2_dissolved", 0.) != 0.:
        raise ValueError("Base must have zero H2_dissolved offset; do not double-apply calibration.")
    offsets["H2_dissolved"] = shift
    ratio = old/new
    concentration_receipts = []
    for method in ("table3_oxidation", "table4_KFT"):
        for row in source[method]:
            # A mole of oxidized H2 gives one mole of H2O in both methods.
            delta = row["published_difference_water_wt_percent"]
            concentration_receipts.append({"method": method, "sample": row["sample"],
                "published_water_equivalent_difference_wt_percent": delta,
                "stoichiometric_H2_mass_ppm": delta*1e4*2.01588/18.01528,
                "difference_minus_rounded_columns_wt_percent": delta-(
                    row.get("oxidized_water_wt_percent", row.get("KFT_water_wt_percent"))
                    -row["initial_water_wt_percent"])})
    return {"assessment_id": "H2_absorption_dilute_amplitude_scenario_v1",
        "scenario": scenario, "standard_shift_key": "H2_dissolved",
        "standard_shift_RT": shift, "dilute_Henry_coefficient_ratio_new_over_old": ratio,
        "reported_extinction_only_standard_shift_interval_RT": [
            math.log((new-error)/old), math.log((new+error)/old)],
        "equations": {"Beer_Lambert": "c_new/c_old = epsilon_old/epsilon_new",
            "dilute_standard": "mu0_new/RT = mu0_old/RT + ln(epsilon_new/epsilon_old)",
            "finite_model_G": "G_new/RT = G_old/RT + n_H2_dissolved * standard_shift_RT",
            "adopted_lnK_before": "-11.403 - 0.76 P_melt/GPa",
            "adopted_lnK_after": "-11.403 - 0.76 P_melt/GPa - standard_shift_RT"},
        "concentration_receipts": concentration_receipts,
        "fixed_dry_host_moles_remapping": [
            {"old_H2_mole_fraction": x, "constant_Henry_scenario_x": ratio*x,
             "rescaling_solute_amount_at_fixed_host_x": ratio*x/(1-x+ratio*x)}
            for x in (1e-6, .001, .01, .04)],
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adopted_by_default": False, "accepted_coupled_material_domain": None,
        "scope": [
            "A measured spectroscopic recalibration motivates this amplitude scenario; unlike the water-capacity width, it is not chosen to cover residual spread.",
            "The published coefficient is retained, not independently refitted; raw calibration-table observations and their rounding remain separate.",
            "Absorption matrix, original experimental mole denominator, temperature dependence and peridotite transfer remain unverified by this conversion.",
            "The quoted coefficient interval varies only the new absorption coefficient, not all measurement/systematic uncertainty or BSE extrapolation.",
            "No gas standard, pressure slope, host mixing term, or original archive is changed.",
            "Finite-composition continuation uses the existing molecular mole-fraction scalar; no claim of exact finite-glass mass-to-mole rescaling is made."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-scenario", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.base_scenario.read_bytes()
    result = absorption_scenario_report(json.loads(raw))
    result["base_scenario_sha256"] = hashlib.sha256(raw).hexdigest()
    result["base_scenario_path"] = str(args.base_scenario)
    args.output_dir.mkdir(exist_ok=False)
    for name, record in (("receipt.json", result), ("scenario.json", result["scenario"])):
        with (args.output_dir/name).open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
