"""Primary-source second-virial diagnostics for the M2 major gas components.

These correlations describe dilute-gas pair interactions. They do not bound
higher virials, trace-species interactions, or high-temperature extrapolation.
The untranscribed H2--He coefficient must be supplied explicitly for a mixture
state; it is never silently set to zero or inferred from pure-fluid values.
"""

import math

import numpy as np

from exoeos.helmholtz import state_tp
from exoeos.second_virial import SecondVirialEOS


COMPONENTS = ("H2", "He", "H2O")

# B = sum(c * (T / 100 K)**d), in cm^3/mol. See gas_virial_sources.json.
_POWERS = {
    "H2-H2": ([42.0803, -143.982, 146.918, -47.5601],
              [-0.33, -1.4, -1.8, -2.2]),
    "H2-H2O": ([33.047, -250.41, 285.42, -186.78],
               [-0.21, -1.50, -2.26, -3.21]),
    "He-H2O": ([55.57, -59.25, 13.32, -4.767],
               [-0.347, -0.85, -1.45, -2.1]),
    "H2O-H2O": ([344.04, -758.26, -24219.0, -3978200.0],
                [-0.5, -0.8, -3.35, -8.3]),
}

# Hurly and Mehl (2007), Table 5: T, B, T*dB/dT.
_HELIUM = np.array([
    [1000, 9.5505, -2.25660], [1200, 9.1354, -2.29376],
    [1400, 8.7804, -2.31003], [1600, 8.4715, -2.31430],
    [1800, 8.1990, -2.31131], [2000, 7.9559, -2.30378],
    [2500, 7.4448, -2.27457], [3000, 7.03314, -2.23916],
])


def pair_coefficients_cm3_mol(temperature_k):
    """Return five supported correlations and an explicit missing sixth pair.

    The example limits evaluation to 1000--3000 K. Above 2000 K the water
    cross correlations are extrapolations beyond their recommended range.
    The pure H2 correlation has no quantified high-temperature error here.
    """
    temperature = float(temperature_k)
    if not math.isfinite(temperature) or not 1000 <= temperature <= 3000:
        raise ValueError("This diagnostic requires 1000 <= T <= 3000 K.")
    result = {
        pair: sum(c * (temperature / 100.0)**d for c, d in zip(cs, ds))
        for pair, (cs, ds) in _POWERS.items()
    }
    index = min(np.searchsorted(_HELIUM[:, 0], temperature, side="right") - 1,
                len(_HELIUM) - 2)
    low, high = _HELIUM[index:index + 2]
    width = high[0] - low[0]
    s = (temperature - low[0]) / width
    result["He-He"] = float(
        (2*s**3 - 3*s**2 + 1)*low[1]
        + (s**3 - 2*s**2 + s)*width*low[2]/low[0]
        + (-2*s**3 + 3*s**2)*high[1]
        + (s**3 - s**2)*width*high[2]/high[0]
    )
    result["H2-He"] = None
    return result


def major_gas_diagnostic(temperature_k, pressure_pa, mole_fractions,
                         *, h2_he_cm3_mol=None):
    """Evaluate a conditioned major-gas state without concealing missing pairs.

    ``mole_fractions`` gives H2, He, H2O fractions in the *original* gas.
    The state, if requested, normalizes these three components; the discarded
    fraction and the missing-pair weight remain in the report. The input
    H2--He coefficient defines a sensitivity scenario, not a measured value.
    """
    pressure = float(pressure_pa)
    fractions = np.asarray(mole_fractions, dtype=float)
    if (fractions.shape != (3,) or not np.all(np.isfinite(fractions))
            or np.any(fractions < 0) or not 0 < fractions.sum() <= 1 + 1e-12):
        raise ValueError("Supply three nonnegative original gas mole fractions.")
    if not math.isfinite(pressure) or pressure <= 0:
        raise ValueError("Pressure must be finite and positive.")
    included = float(fractions.sum())
    x = fractions / included
    pairs = pair_coefficients_cm3_mol(temperature_k)
    known_part = (x[0]**2*pairs["H2-H2"] + x[1]**2*pairs["He-He"]
                  + x[2]**2*pairs["H2O-H2O"]
                  + 2*x[0]*x[2]*pairs["H2-H2O"]
                  + 2*x[1]*x[2]*pairs["He-H2O"])
    report = {
        "temperature_k": float(temperature_k), "pressure_pa": pressure,
        "components": list(COMPONENTS),
        "original_mole_fractions": fractions.tolist(),
        "conditioned_mole_fractions": x.tolist(),
        "omitted_gas_mole_fraction": max(0.0, 1.0 - included),
        "pair_coefficients_cm3_mol": pairs,
        "known_part_of_conditioned_B_cm3_mol": float(known_part),
        "H2_He_weight_in_conditioned_B": float(2*x[0]*x[1]),
        "uncovered_pair_weight_in_original_B": max(0.0, 1.0 - included**2),
        "water_cross_pairs_extrapolated": float(temperature_k) > 2000,
        "full_mixture_physical_error_bound": None,
        "state_scope": "conditioned_three_component_second_virial_scenario",
        "state": None,
    }
    if h2_he_cm3_mol is None:
        return report
    cross = float(h2_he_cm3_mol)
    if not math.isfinite(cross):
        raise ValueError("The H2--He scenario coefficient must be finite.")
    matrix = 1e-6 * np.array([
        [pairs["H2-H2"], cross, pairs["H2-H2O"]],
        [cross, pairs["He-He"], pairs["He-H2O"]],
        [pairs["H2-H2O"], pairs["He-H2O"], pairs["H2O-H2O"]],
    ])
    eos = SecondVirialEOS(matrix)
    state = state_tp(eos, temperature_k, pressure, x)
    if not np.isfinite(float(state.molar_density)):
        raise ValueError("The scenario has no stable second-virial density root.")
    report["supplied_H2_He_scenario_cm3_mol"] = cross
    report["state"] = {
        "B_cm3_mol": float(x @ matrix @ x * 1e6),
        "molar_density_mol_m3": float(state.molar_density),
        "compressibility_factor": float(state.compressibility_factor),
        "log_fugacity_coefficients": np.asarray(
            state.log_fugacity_coefficients).tolist(),
        "reduced_residual_gibbs": float(state.reduced_residual_gibbs),
    }
    return report


if __name__ == "__main__":
    import argparse
    import hashlib
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--root-index", type=int, default=0)
    parser.add_argument("--h2-he-cm3-mol", type=float, nargs="*", default=[])
    args = parser.parse_args()
    raw = args.closure.read_bytes()
    parcel = json.loads(raw)["runs"][args.run_index]["roots"][
        args.root_index]["basal_parcel"]
    fractions = dict(zip(parcel["gas_species"], parcel["gas_mole_fractions"]))
    selected = [fractions[name] for name in ("H2", "He1", "H2O1")]
    report = {
        "schema": "m2_major_gas_virial_diagnostic_v1",
        "input_closure_sha256": hashlib.sha256(raw).hexdigest(),
        "input_run_index": args.run_index, "input_root_index": args.root_index,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources_sha256": hashlib.sha256(Path(__file__).with_name(
            "gas_virial_sources.json").read_bytes()).hexdigest(),
        "known_pairs": major_gas_diagnostic(parcel["T_K"],
            parcel["P_bar"]*1e5, selected),
        "explicit_coefficient_scenarios": [major_gas_diagnostic(parcel["T_K"],
            parcel["P_bar"]*1e5, selected, h2_he_cm3_mol=cross)
            for cross in args.h2_he_cm3_mol],
        "scenario_values_are_empirical_bounds": False,
        "finite_equilibrium_recomputed": False,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
