"""Declared site-mixing expressions for independent mineral stability bounds.

These property expressions do not select phases. Their closed physical site
domains include signed endmember coordinates where the published models do.
Native pure standards, finite binary compatibility and empirical calibration
are separate inputs and evidence.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np


PARAMETER_PATH = Path(__file__).with_name("m2_solid_mixing") / "parameters.json"
COMMON_R = 8.31446261815324


def polynomial_value(terms, coordinates):
    """Evaluate a sparse polynomial without evaluating a source string."""
    x = np.asarray(coordinates, dtype=float)
    return sum(float(coefficient) * np.prod(x ** np.asarray(powers, dtype=int))
               for coefficient, powers in terms)


def solid_mixing_parameters(phase: str, temperature_k: float, pressure_pa: float,
                            *, common_R: float = COMMON_R) -> dict:
    """Supply the complete declared site domain and dimensionless expression.

    Olivine currently exposes the complete Fe/Mg/Ca face only; the required
    absence of Mn, Ni and Co must be established by the chemical consumer.
    Other phases include their full declared site domains, ordering included.
    """
    if (not all(np.isfinite(v) and v > 0 for v in (temperature_k, pressure_pa, common_R))
            or not isinstance(phase, str)):
        raise ValueError("A phase and positive finite T/P/R are required.")
    data = json.loads(PARAMETER_PATH.read_text())
    if phase not in data["models"]:
        raise ValueError("No declared site expression for phase: " + phase)
    model = deepcopy(data["models"][phase])
    rt = common_R * temperature_k
    terms = {}
    for field, multiplier in (("H_polynomial", 1.), ("S_polynomial", -temperature_k),
                              ("V_polynomial", pressure_pa / 1e5 - 1.)):
        for value, powers in model.pop(field):
            key = tuple(powers)
            terms[key] = terms.get(key, 0.) + multiplier * value
    model["polynomial_rt"] = [[value / rt, list(powers)] for powers, value in sorted(terms.items()) if value]
    for site in model["entropy_sites"]:
        site["coefficient_rt"] = site.pop("multiplicity") * data["native_entropy_R_J_mol_K"] / common_R
    if model["barrier"] is not None:
        barrier = model["barrier"]
        barrier["numerator_rt"] = barrier.pop("numerator_J_mol") / rt
        if "coordinate_index" in barrier:
            powers = [0] * len(model["coordinate_order"])
            powers[barrier.pop("coordinate_index")] = 1
            barrier["polynomial"] = [[1., powers]]
    return {**model, "T_K": float(temperature_k), "P_Pa": float(pressure_pa),
            "native_entropy_R_J_mol_K": data["native_entropy_R_J_mol_K"],
            "common_R_J_mol_K": float(common_R), "oxide_order": data["oxide_order"],
            "schema": data["schema"], "parameter_sha256": hashlib.sha256(PARAMETER_PATH.read_bytes()).hexdigest(),
            "source_commit": data["upstream_commit"], "source_url": data["upstream_url"],
            "native_binary_uniform_error_bound": None,
            "empirical_calibration_domain_established": False}


def solid_mixing_state(parameters: dict, coordinates) -> dict:
    """Return mixing G on an admissible site state, including boundary limits."""
    x = np.asarray(coordinates, dtype=float)
    bounds = np.asarray(parameters["coordinate_bounds"], dtype=float)
    if (x.shape != (len(bounds),) or not np.all(np.isfinite(x))
            or np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1])):
        raise ValueError("Coordinates are outside the declared site box.")
    if any(polynomial_value(row, x) < 0 for row in parameters["nonnegative_polynomials"]):
        raise ValueError("Coordinates violate a physical site constraint.")
    energy = polynomial_value(parameters["polynomial_rt"], x)
    for site in parameters["entropy_sites"]:
        amount = polynomial_value(site["polynomial"], x)
        if amount < 0 or amount > 1:
            raise ValueError("An occupation lies outside [0,1].")
        if amount:
            energy += site["coefficient_rt"] * amount * np.log(amount)
    barrier = parameters["barrier"]
    if barrier is not None:
        amount = polynomial_value(barrier["polynomial"], x)
        if amount < 0:
            raise ValueError("The barrier coordinate is negative.")
        if amount == 0:
            return {"status": "positive_infinite_boundary", "mixing_gibbs_rt": None,
                    "boundary_limit": "+infinity"}
        energy += barrier["numerator_rt"] / amount
    if parameters.get("pure_reference_bounds"):
        return {"status": "ok_unreferenced_site_properties", "unreferenced_gibbs_rt": float(energy),
                "pure_reference_bounds": parameters["pure_reference_bounds"],
                "native_endmember_fractions": [polynomial_value(row, x) for row in parameters["endmember_polynomials"]]}
    return {"status": "ok_declared_site_properties", "mixing_gibbs_rt": float(energy),
            "native_endmember_fractions": [polynomial_value(row, x) for row in parameters["endmember_polynomials"]]}
