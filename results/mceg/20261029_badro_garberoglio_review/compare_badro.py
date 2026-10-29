"""Compare inspected Badro SI Eqs. 3-4 with an existing parameter option.

Run with PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1. This is a
source-algebra comparison at numerical fixture states, not an empirical fit.
The committed default and historical reference fixture are never modified.
"""

from decimal import Decimal, localcontext
import importlib.util
import json
from pathlib import Path

import jax
import numpy as np

from exoeos import MaFeSiOLiquid, solution_state


ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "tests/reference/fe_si_o_ma2001.json"
GENERATOR = ROOT / "tests/reference/generate_fe_si_o_ma2001.py"


def main():
    if not jax.config.x64_enabled:
        raise RuntimeError("The source comparison requires JAX_ENABLE_X64=1.")
    spec = importlib.util.spec_from_file_location("ma_reference", GENERATOR)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    fixture = json.loads(REFERENCE.read_text())
    parameters = dict(fixture["parameters"])
    # Table S1, p.3: epsilon_O^O(T0)=-1; p.2: T0=1873 K scaling.
    # The separate infinite-dilution term remains 4.29-16500/T.
    parameters["epsilon_OO_times_temperature_K"] = "-1873"
    model = MaFeSiOLiquid(interaction_K=(12.41 * 1873, -1873, -5 * 1873))
    default = MaFeSiOLiquid()
    rows = []
    with localcontext() as context:
        context.prec = 60
        for state in fixture["states"]:
            t, p, x = state["T_K"], state["P_Pa"], state["x"]
            # The generator's expanded solvent and solute expressions were
            # visually matched to Badro Eqs.3-4 before this calculation.
            expected = reference.activities(parameters, Decimal(str(t)),
                                            [Decimal(str(v)) for v in x])
            target = np.array([float(v) for v in expected["ln_gamma_source"]])
            actual_state = solution_state(model, t, p, x)
            actual = np.asarray(actual_state.log_activity_coefficients
                                + model.standard_state_shift_RT(t))
            default_source = np.asarray(solution_state(default, t, p, x).log_activity_coefficients
                                        + default.standard_state_shift_RT(t))
            error = float(np.max(np.abs(actual - target)))
            g_error = abs(float(actual_state.reduced_excess_gibbs)
                          - float(expected["gex_formal_RT"]))
            # After Fe completion, changes in epsilon_OO affect every species.
            oxygen = x[2]
            delta_b = (-16500 + 1873) / t
            delta_fe = delta_b * (oxygen + np.log1p(-oxygen))
            predicted_delta = np.array([delta_fe, delta_fe, delta_b * oxygen])
            delta_error = float(np.max(np.abs(default_source - actual - predicted_delta)))
            assert max(error, g_error, delta_error) < 5e-12, state["id"]
            rows.append({"id": state["id"], "T_K": t, "P_Pa": p, "x": x,
                         "badro_ln_gamma_source": target.tolist(),
                         "custom_provider_max_abs_ln_gamma_error": error,
                         "custom_provider_abs_gex_RT_error": g_error,
                         "default_minus_badro_ln_gamma_source": (default_source - actual).tolist(),
                         "default_difference_formula_max_abs_error": delta_error})
    result = {
        "scope": "Badro SI Eqs.3-4 and Table S1 parameter option; numerical fixture states are not calibration observations.",
        "components": ["Fe", "Si", "O"],
        "parameters": parameters,
        "default_changed": False,
        "absolute_tolerance": 5e-12,
        "state_count": len(rows),
        "maximum_absolute_ln_gamma_error": max(r["custom_provider_max_abs_ln_gamma_error"] for r in rows),
        "maximum_absolute_gex_RT_error": max(r["custom_provider_abs_gex_RT_error"] for r in rows),
        "maximum_default_difference_formula_error": max(r["default_difference_formula_max_abs_error"] for r in rows),
        "states": rows,
    }
    output = Path(__file__).with_name("badro_parameter_comparison.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Badro parameter option: {len(rows)} states agree within {result['absolute_tolerance']:g}.")


if __name__ == "__main__":
    main()
