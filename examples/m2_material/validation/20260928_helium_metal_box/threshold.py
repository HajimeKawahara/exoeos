"""Retain the unchanged 15 scenarios and add their fixed-box 1% threshold."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
MATERIAL = HERE.parents[1]
sys.path.insert(0, str(MATERIAL))
from helium_metal_threshold import metal_partition_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original_path = HERE / "assessment.json"
    raw = original_path.read_bytes()
    original = json.loads(raw)
    for name, expected in original["input_and_code_sha256"].items():
        if hashlib.sha256((MATERIAL / name).read_bytes()).hexdigest() != expected:
            raise ValueError("The original box recipe changed: " + name)
    maximum = max(row["reported_D"] for row in original["primary_partition_reference"]["S2_same_run_Fe80Ni20"])
    rows = []
    for row in original["rows"]:
        inputs = {key: value for key, value in row["finite_He_D_zero_control"]["inputs"].items()
                  if key != "metal_silicate_mass_fraction_D"}
        result = metal_partition_threshold(inputs, .01)
        result["host"] = row["henry_reference"]["host"]
        result["critical_D_over_maximum_TableS2_central_D"] = result["critical_mass_fraction_partition_D"] / maximum
        rows.append(result)
    files = [Path(__file__), MATERIAL / "helium_metal_threshold.py", MATERIAL / "helium_metal_box.py"]
    result = {
        "assessment_id": "fixed_host_He_metal_partition_break_even_v1",
        "original_box_assessment_sha256": hashlib.sha256(raw).hexdigest(),
        "original_source_sha256": original["source"]["sha256"],
        "target": {"partitioned_He_amount_relative_to_total": .01,
                   "kind": "Engineering partition criterion applied only within the fixed box; not a probabilistic uncertainty."},
        "maximum_same_run_TableS2_central_D": maximum,
        "rows": rows,
        "interpretation": "Dcrit is the imposed mass-fraction ratio needed for 1% of finite He to enter metal with all other box assumptions fixed. Its ratio to the largest Table S2 central value measures a conditional departure, not allowed or excluded material uncertainty.",
        "empirical_low_pressure_D_bound": None,
        "supports_material_admission": False, "is_BSE_error_bound": False,
        "atmospheric_mass_target_assessed": False, "bottom_pressure_target_assessed": False,
        "metal_appearance_boundary_target_assessed": False,
        "input_and_code_sha256": {str(path.relative_to(MATERIAL)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
