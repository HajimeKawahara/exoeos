"""Audit the supplied dry BSE liquid equations; no equilibrium certification."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import melts_liquid_evaluator as melts


DATA = Path(__file__).resolve().parent
TEMPERATURE_K = 2173.15
PRESSURE_PA = 1e5


def component_amounts(ledger):
    """Map the exported absolute oxide ledger to 100 g of dry rock.

    The source's conventional atomic masses define the physical amount
    scale. Native MELTS rounded masses are used only for its input conversion.
    No H/He inventory is inserted into this dry property evaluation.
    """
    scale = 0.1 / ledger["dry_rock_mass_kg"]
    oxides = dict(zip(ledger["oxide_order"], ledger["oxide_amounts_mol"]))
    # MELTS uses lower-case labels, whereas the inventory uses chemical case.
    by_lowercase = {name.lower(): amount for name, amount in oxides.items()}
    oxide_moles = scale * np.array([by_lowercase.get(name, 0.) for name in melts.OXIDES])
    n = np.linalg.solve(melts.NU.T, oxide_moles)
    if np.any(n < 0):
        raise ValueError("BSE oxide ledger lies outside the nonnegative MELTS component cone.")
    actual = dict(zip(melts.ELEMENTS, melts.FORMULA_MATRIX.T @ n))
    expected = np.asarray(ledger["rock_element_amounts_mol"]) * scale
    np.testing.assert_allclose([actual.get(name, 0.) for name in ledger["elements"]],
                               expected, rtol=1e-14, atol=0)
    return n, scale


def validate(runtime, python_executable=None):
    ledger = json.loads((DATA / "bse_inventory.json").read_text())
    source = json.loads((DATA / "bse_source.json").read_text())
    source_hash = hashlib.sha256((DATA / "bse_source.json").read_bytes()).hexdigest()
    if source_hash != ledger["provenance"]["source_sha256"]:
        raise ValueError("Pinned BSE source differs from the exported inventory receipt.")
    n, amount_scale = component_amounts(ledger)

    def evaluate(values):
        return melts.evaluate_liquid(TEMPERATURE_K, PRESSURE_PA, values, runtime=runtime,
                                     python_executable=python_executable)

    reference = evaluate(n)
    scaled = evaluate(2.5 * n)
    present = n > 0
    mu = np.array(reference["mu_J_mol"], dtype=float)
    np.testing.assert_allclose(scaled["gibbs_J"], 2.5 * reference["gibbs_J"], rtol=5e-9)
    np.testing.assert_allclose(np.array(scaled["mu_J_mol"], dtype=float)[present],
                               mu[present], rtol=5e-9)
    derivatives = []
    for index in np.flatnonzero(present):
        step = np.zeros_like(n)
        step[index] = 1e-3 * n[index]
        plus, minus = evaluate(n + step), evaluate(n - step)
        derivative = (plus["gibbs_J"] - minus["gibbs_J"]) / (2 * step[index])
        derivatives.append({"component": melts.COMPONENTS[index], "step_mol": step[index],
                            "derivative_J_mol": derivative, "mu_J_mol": mu[index],
                            "error_J_mol": derivative - mu[index]})
    maximum_error = max(abs(item["error_J_mol"]) for item in derivatives)
    if maximum_error > 0.02:
        raise ValueError(f"BSE extensive energy derivative error {maximum_error} exceeds 0.02 J/mol.")
    return {
        "status": "passed_local_equation_checks_only",
        "scientific_acceptance": {"M2_A": "pending", "M2_B": "pending",
                                  "liquid_stability": "not_evaluated", "cross_phase_standards": "not_aligned"},
        "source": source["source"],
        "input": {"ledger": ledger, "physical_dry_mass_g": 100.,
                  "absolute_amount_scale": amount_scale,
                  "inventory_commit": "0c604bcb7ccd850e9e747fb9fb16448e33d22e25",
                  "inventory_repository": "https://github.com/HajimeKawahara/exoinventory",
                  "hydrogen_and_helium": "Retained in exported total inventory; absent from this dry-liquid-only audit.",
                  "ledger_sha256": hashlib.sha256((DATA / "bse_inventory.json").read_bytes()).hexdigest(),
                  "source_sha256": source_hash},
        "checks": {"scaling_factor": 2.5, "finite_difference_relative_step": 1e-3,
                   "derivative_absolute_tolerance_J_mol": 0.02,
                   "maximum_derivative_error_J_mol": maximum_error,
                   "derivatives": derivatives, "native_evaluations": 2 + 2 * len(derivatives)},
        "state": reference,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", dest="python_executable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.runtime, args.python_executable)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
