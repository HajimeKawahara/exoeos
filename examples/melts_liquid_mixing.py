"""Published MELTS v1.0.2 liquid mixing expression, including its boundary limits.

The property provider owns this expression; consumers own phase stability.
No standard state, empirical domain or native build identity is inferred.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PARAMETER_PATH = Path(__file__).parent / "m2_liquid_mixing/parameters.json"
PARAMETERS = json.loads(PARAMETER_PATH.read_text())
COMPONENTS = PARAMETERS["component_order"]
PUBLISHED_MODEL_ID = "melts_v102_published_mixing_native_standard_states_v1"


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


def make_published_liquid_evaluator(native_evaluator, *, runtime, python_executable,
                                    common_R=8.31446261815324):
    """Select published mixing with native standards cached once per T/P/R.

    The resulting model is explicit and distinct from native composition
    evaluation. Every cache entry retains the actual native standard receipt
    and a mixing comparison. No interpolated pressure or standard offsets
    are introduced. This callback supplies liquid properties only.
    """
    runtime = Path(runtime).resolve()
    cache, receipts = {}, []
    provider_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def evaluate_liquid(temperature, pressure, component_moles, *, runtime=runtime,
                        python_executable=python_executable, common_R=common_R,
                        calculation_mode=1, **options):
        if options or calculation_mode != 1:
            raise ValueError("The published callback supplies mode-1 liquid properties only.")
        n, grams = native_evaluator.validate_request(temperature, pressure, component_moles,
                                                      common_R, calculation_mode)
        key = (float(temperature), float(pressure), float(common_R), str(Path(runtime).resolve()),
               str(python_executable))
        if key not in cache:
            probe = np.ones(len(COMPONENTS))
            probe[PARAMETERS["unsupported_positive_indices"]] = 0
            native = native_evaluator.evaluate_liquid(temperature, pressure, probe,
                        runtime=Path(runtime), python_executable=python_executable, common_R=common_R)
            comparison = compare_native_mixing(native)
            if comparison["status"] != "compatible_at_supplied_state":
                raise ValueError("The native standard probe disagrees with the published mixing expression.")
            standard = np.asarray(native["mu0_J_mol"], dtype=float)
            if not np.all(np.isfinite(standard[probe > 0])):
                raise ValueError("A supported pure-component standard is unavailable.")
            receipt = {"T_K": temperature, "P_Pa": pressure, "common_R_J_mol_K": common_R,
                       "native_properties": native, "comparison": comparison,
                       "standard_state_policy": "Native pure-liquid mu0 at this exact T/P; no fitted offset or pressure interpolation."}
            payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            identifier = hashlib.sha256(payload).hexdigest()
            receipt["sha256_without_this_field"] = identifier
            receipts.append(receipt)
            cache[key] = (standard, comparison["parameters"], native, identifier)
        standard, parameters, anchor, identifier = cache[key]
        mixed = liquid_mixing_state(parameters, n)
        present = n > 0
        rt = common_R*temperature
        mixing_mu = np.asarray(mixed["mixing_mu_rt"], dtype=float)
        mu = standard/rt + mixing_mu
        energy = float(n[present] @ standard[present]/rt + mixed["mixing_gibbs_rt"])
        if not np.isfinite(energy) or not np.all(np.isfinite(mu[present])):
            raise ValueError("The published expression is not finite on the requested active support.")
        x = n/n.sum()
        ln_activity = mixing_mu/parameters["entropy_coefficient"]
        nullable = lambda values: [float(value) if exists else None for value, exists in zip(values, present)]
        oxide_moles = native_evaluator.NU.T @ n
        active_oxides = oxide_moles > 0
        subspace = native_evaluator.NU[np.ix_(present, active_oxides)]
        oxide_mu = [None]*len(grams)
        oxide_status = "unavailable: present component/oxide subspace is not square and full rank"
        if subspace.shape[0] == subspace.shape[1] and np.linalg.matrix_rank(subspace) == subspace.shape[0]:
            values = np.full(grams.shape, np.nan)
            values[active_oxides] = np.linalg.solve(subspace, rt*mu[present])
            oxide_mu = [float(value) if exists else None for value, exists in zip(values, active_oxides)]
            oxide_status = "full potentials on the present square subspace; no oxide standards or activities"
        return {
            "model_id": PUBLISHED_MODEL_ID, "status": "ok_supplied_liquid_properties",
            "T_K": float(temperature), "P_Pa": float(pressure),
            "component_order": COMPONENTS, "component_moles": n.tolist(),
            "returned_component_moles": n.tolist(), "x": x.tolist(),
            "oxide_order": native_evaluator.OXIDES, "oxide_mass_g": grams.tolist(),
            "returned_oxide_mass_g": grams.tolist(),
            "oxide_molar_masses_g_mol": native_evaluator.OXIDE_MASSES.tolist(),
            "mass_g": float(grams.sum()), "gibbs_J": energy*rt, "gibbs_RT": energy,
            "mu_J_mol": nullable(rt*mu), "mu0_J_mol": nullable(standard),
            "mu_RT": nullable(mu), "mu0_RT": nullable(standard/rt),
            "activity": nullable(np.exp(ln_activity)), "ln_activity": nullable(ln_activity),
            "ln_gamma": nullable(ln_activity-np.log(np.where(present, x, 1.))),
            "ln_activity_common_R": nullable(mixing_mu),
            "ln_gamma_common_R": nullable(mixing_mu-np.log(np.where(present, x, 1.))),
            "oxide_mu_J_mol": oxide_mu, "oxide_potential_status": oxide_status,
            "basis": {**anchor["basis"], "element_moles": (native_evaluator.FORMULA_MATRIX.T@n).tolist(),
                      "mixing": "Published regular-liquid expression; native pure-liquid standards held at this exact T/P."},
            "phase_policy": {**anchor["phase_policy"], "stability_checked": False},
            "capabilities": {"carbon": "unsupported; published v1.0.2 mixing only"},
            "mixing_expression": parameters,
            "provenance": {"model_selection": PUBLISHED_MODEL_ID,
                           "native_standard_state_receipt_sha256": identifier,
                           "mixing_provider_sha256": provider_hash,
                           "native_composition_evaluated": False},
        }

    return SimpleNamespace(
        MODEL_ID=PUBLISHED_MODEL_ID, COMPONENTS=COMPONENTS, ELEMENTS=native_evaluator.ELEMENTS,
        OXIDES=native_evaluator.OXIDES, NU=native_evaluator.NU, OXIDE_MASSES=native_evaluator.OXIDE_MASSES,
        FORMULA_MATRIX=native_evaluator.FORMULA_MATRIX, REFERENCE=native_evaluator.REFERENCE,
        REFERENCE_PATH=native_evaluator.REFERENCE_PATH, __file__=__file__,
        evaluate_liquid=evaluate_liquid, standard_state_receipts=receipts,
        mixing_model_id=PARAMETERS["model_id"],
        mixing_parameter_sha256=hashlib.sha256(PARAMETER_PATH.read_bytes()).hexdigest())
