"""Validate and save opt-in MELTS carbon properties using a pinned runtime.

The saved compositions are numerical controls, not calibrated equilibria.
No download, installation, or modification of the original MORB archive occurs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "examples/melts_liquid_evaluator.py"


def generate(runtime, python_executable=None):
    provider = runpy.run_path(str(EVALUATOR))
    components = provider["COMPONENTS"]
    reference = provider["REFERENCE"]
    carbon = components.index("co2")
    base = np.asarray(reference["states"][0]["liquid"]["component_moles"])
    records = []
    count = 0

    def evaluate(temperature, pressure, amounts):
        nonlocal count
        count += 1
        return provider["evaluate_liquid"](
            temperature, pressure, amounts, runtime=runtime,
            python_executable=python_executable, calculation_mode=4,
        )

    for name, temperature, pressure, carbon_moles in (
        ("zero_c_1473K_50MPa", 1473.15, 5e7, 0.0),
        ("carbon_1473K_50MPa", 1473.15, 5e7, 0.001),
        ("carbon_1473K_500MPa", 1473.15, 5e8, 0.001),
        ("carbon_1673K_50MPa", 1673.15, 5e7, 0.003),
    ):
        n = base.copy()
        n[carbon] = carbon_moles
        result = evaluate(temperature, pressure, n)
        present = n > 0
        mu = np.asarray(result["mu_J_mol"], float)
        scale_errors = []
        for scale in (0.1, 2.5):
            scaled = evaluate(temperature, pressure, scale * n)
            provider["require_match"](scaled["gibbs_J"], scale * result["gibbs_J"], "extensive carbon Gibbs scaling")
            scaled_mu = np.asarray(scaled["mu_J_mol"], float)
            provider["require_match"](scaled_mu[present], mu[present], "intensive carbon potential scaling")
            scale_errors.append(float(np.max(np.abs(scaled_mu[present] - mu[present]))))
        derivatives = []
        for index in np.flatnonzero(present):
            step = n[index] * 1e-3
            plus, minus = n.copy(), n.copy()
            plus[index] += step
            minus[index] -= step
            fd = (evaluate(temperature, pressure, plus)["gibbs_J"]
                  - evaluate(temperature, pressure, minus)["gibbs_J"]) / (2 * step)
            provider["require_match"](fd, mu[index], "carbon dG/dn = mu", rtol=0, atol=0.02)
            derivatives.append({"component": components[index], "dG_dn_J_mol": fd,
                                "error_J_mol": float(fd - mu[index])})
        records.append({"id": name, "result": result,
                        "derivatives": derivatives,
                        "max_scaling_mu_error_J_mol": max(scale_errors)})

    return {
        "status": "passed_numerical_property_checks",
        "evidence_level": "Supplied-liquid equation consistency; not empirical calibration or phase equilibrium.",
        "model_id": records[0]["result"]["model_id"],
        "host_source": "melts_silicate_v1.json, states[0].liquid.component_moles; only co2 is changed",
        "carbon_input_basis": "mol CO2 independent component; adds one C and two O atoms per formula unit",
        "excluded": ["metal C/N/S", "reduced dissolved CO/CH4", "dissolved N/S", "graphite/carbides", "cross-phase standard alignment"],
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluation_count": count,
        "tolerances": {"property_rtol": 5e-9, "fd_atol_J_mol": 0.02, "fd_relative_step": 1e-3},
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--python", help="External Python with numpy and tinynumpy.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(generate(args.runtime, args.python), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
