"""Published MELTS v1.0.2 liquid mixing expression, including its boundary limits.

The property provider owns this expression; consumers own phase stability.
No standard state, empirical domain or native build identity is inferred.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


PARAMETER_PATH = Path(__file__).parent / "m2_liquid_mixing/parameters.json"
PARAMETERS = json.loads(PARAMETER_PATH.read_text())
COMPONENTS = PARAMETERS["component_order"]


def liquid_mixing_parameters(temperature, pressure, common_R=8.31446261815324):
    """Return the regular-solution coefficients in the consumer's RT units."""
    if any(not np.isfinite(v) or v <= 0 for v in (temperature, pressure, common_R)):
        raise ValueError("Temperature, pressure and common R must be positive and finite.")
    matrix = np.zeros((len(COMPONENTS), len(COMPONENTS)))
    entries = iter(PARAMETERS["parameters"])
    for i in range(len(COMPONENTS)):
        for j in range(i + 1, len(COMPONENTS)):
            row = next(entries)
            value = (float(row["H_J_mol"]) - temperature * float(row["S_J_mol_K"])
                     + (pressure / 1e5 - 1.) * float(row["V_J_mol_bar"])) / (common_R * temperature)
            matrix[i, j] = matrix[j, i] = value
    return {"model_id": PARAMETERS["model_id"], "component_order": COMPONENTS,
            "T_K": float(temperature), "P_Pa": float(pressure),
            "common_R_J_mol_K": float(common_R),
            "entropy_coefficient": PARAMETERS["backend_R_J_mol_K"] / common_R,
            "quadratic_matrix_rt": matrix.tolist(), "water_index": PARAMETERS["water_index"],
            "unsupported_positive_indices": PARAMETERS["unsupported_positive_indices"],
            "element_order": PARAMETERS["element_order"],
            "component_element_matrix": PARAMETERS["component_element_matrix"],
            "domain": "Nonnegative component simplex, excluding the four unsupported components; closed x log x limits.",
            "source": PARAMETERS["source"],
            "parameter_sha256": hashlib.sha256(PARAMETER_PATH.read_bytes()).hexdigest(),
            "native_build_identity_established": False}


def liquid_mixing_state(parameters, component_moles):
    """Evaluate extensive mixing G/RT and present-component mixing potentials.

    The extra water term is N [w log w + (1-w) log(1-w)].
    Exact zero amounts use the continuous energy limit, never log(0).
    """
    amounts = np.asarray(component_moles, dtype=float)
    if (amounts.shape != (len(parameters["component_order"]),)
            or not np.all(np.isfinite(amounts)) or np.any(amounts < 0)
            or not np.isfinite(amounts.sum()) or amounts.sum() <= 0):
        raise ValueError("Supply finite nonnegative component amounts with a positive total.")
    if np.any(amounts[parameters["unsupported_positive_indices"]] > 0):
        raise ValueError("Positive unsupported MELTS v1.0.2 component.")
    total = float(amounts.sum())
    x = amounts / total
    water = parameters["water_index"]
    w = float(x[water])
    alpha = parameters["entropy_coefficient"]
    matrix = np.asarray(parameters["quadratic_matrix_rt"])

    def xlogx(v):
        return v * np.log(v) if v > 0 else 0.

    excess = float(.5 * x @ matrix @ x)
    energy = total * (excess + alpha * (sum(xlogx(v) for v in x) + xlogx(w) + xlogx(1-w)))
    potentials = matrix @ x - excess
    present = amounts > 0
    potentials[present] += alpha * np.log(x[present])
    for i in np.flatnonzero(present):
        potentials[i] += alpha * np.log(w if i == water else 1-w)
    return {"mixing_gibbs_rt": float(energy),
            "mixing_mu_rt": [float(v) if present[i] else None for i, v in enumerate(potentials)],
            "component_moles": amounts.tolist(), "model_id": parameters["model_id"]}


def compare_native_mixing(properties, *, tolerance_rt=1e-8):
    """Check this source expression against an independent native receipt.

    This finite comparison is evidence of compatibility, not a global error
    bound between a release binary and the published mathematical model.
    """
    if (properties["model_id"] != "alphamelts_2_3_2_rhyolite_melts_1_0_2_supplied_liquid_v1"
            or properties["component_order"] != COMPONENTS):
        raise ValueError("A supplied rhyolite-MELTS v1.0.2 receipt in the declared basis is required.")
    common_R = properties["basis"]["common_R_J_mol_K"]
    parameters = liquid_mixing_parameters(properties["T_K"], properties["P_Pa"], common_R)
    state = liquid_mixing_state(parameters, properties["component_moles"])
    present = np.asarray(properties["component_moles"]) > 0
    native = np.asarray(properties["mu_RT"], dtype=float) - np.asarray(properties["mu0_RT"], dtype=float)
    difference = np.asarray(state["mixing_mu_rt"], dtype=float)[present] - native[present]
    if not np.all(np.isfinite(difference)):
        raise ValueError("Native present-component mixing potentials are unavailable.")
    maximum = float(np.max(np.abs(difference)))
    return {"status": "compatible_at_supplied_state" if maximum <= tolerance_rt else "mismatch",
            "maximum_potential_difference_rt": maximum, "tolerance_rt": tolerance_rt,
            "global_native_error_bound_certified": False, "parameters": parameters}
