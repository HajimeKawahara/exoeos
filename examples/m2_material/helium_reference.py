"""Dry-host He Henry coefficients; no equilibrium or empirical admission."""

import argparse
import hashlib
import json
import math
from pathlib import Path


DATA = Path(__file__).with_name("na_helium_transfer_sources.json")
R_J_MOL_K = 8.31446261815324


def helium_henry_coefficient(host, temperature_K):
    """Interpolate Appendix C simulations, retaining their dry-host basis.

    The STP conversion explicitly uses 273.15 K and 101325 Pa. This does not
    assign any high-pressure or wet-BSE empirical validity to the coefficient.
    """
    temperature = float(temperature_K)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature_K must be positive and finite")
    evidence = json.loads(DATA.read_text())
    rows = sorted((row for row in evidence["references"]["guillot_sator_2012"]["He_rows"]
                   if row["host"] == host), key=lambda row: row["T_K"])
    if not rows:
        raise ValueError("unknown simulated dry host")
    if not rows[0]["T_K"] <= temperature <= rows[-1]["T_K"]:
        raise ValueError("temperature outside this host's simulated interval")
    for lower, upper in zip(rows[:-1], rows[1:]):
        if lower["T_K"] <= temperature <= upper["T_K"]:
            weight = ((1 / temperature - 1 / lower["T_K"])
                      / (1 / upper["T_K"] - 1 / lower["T_K"]))
            solubility = math.exp(math.log(lower["S"])
                                  + weight * math.log(upper["S"] / lower["S"]))
            break
    standard_volume = 1e6 * R_J_MOL_K * 273.15 / 101325
    return {
        "host": host,
        "temperature_K": temperature,
        "S_cm3_STP_per_g_bar": solubility,
        "He_mol_per_kg_dry_host_per_bar_fugacity": 1000 * solubility / standard_volume,
        "standard_volume_cm3_per_mol": standard_volume,
        "STP_assumption": {"T_K": 273.15, "P_Pa": 101325},
        "interpolation": "ln(S) linear in reciprocal absolute temperature",
        "bracketing_simulated_rows": [lower, upper],
        "supports_material_admission": False,
        "is_BSE_error_bound": False,
        "source_data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    supplied = json.loads(args.input.read_text())
    result = helium_henry_coefficient(**supplied)
    result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    result["evaluator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
