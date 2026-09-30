"""Dry published MELTS plus an integrable, absorptivity-based water law.

This opt-in model replaces the entire native water contribution. It does
not add a solubility correction on top of hydrated MELTS. Model and
calibration scopes are recorded separately from numerical validity.
"""

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from exoeos import mass_fraction_solute_state


MODEL_ID = "dry_melts_thompson2025_water_equivalent_v1"
DRY_MODEL_ID = "melts_v102_published_mixing_native_standard_states_v1"
CALIBRATION_PATH = Path(__file__).parent / "m2_material/water_calibration.py"
SOURCE_PATH = Path(__file__).parent / "m2_material/water_data/water_basis_pressure_sources.json"
_spec = importlib.util.spec_from_file_location("_exoeos_water_capacity", CALIBRATION_PATH)
_capacity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_capacity)


def make_reconstructed_water_evaluator(dry_evaluator, gas_water_standard_rt):
    """Combine dry-host G with water G in the existing endmember basis.

    The gas callback receives K/Pa and returns the caller's common-R
    H2O standard at1bar. Native standards are retained for dry components.
    Pressure enters dry MELTS; the water law has no fitted pressure-volume
    term. All dry oxides enter its predictor denominator, with ferric iron
    expressed as2FeO to match the empirical FeO-total convention. Actual
    component masses and conserved formulas are never changed by this
    predictor conversion. Host composition derivatives are retained.
    """
    if dry_evaluator.MODEL_ID != DRY_MODEL_ID:
        raise ValueError("The water reconstruction requires published dry MELTS.")
    names = list(dry_evaluator.COMPONENTS)
    water_index = names.index("h2o")
    oxides = list(dry_evaluator.OXIDES)
    nu = np.asarray(dry_evaluator.NU, dtype=float)
    masses = nu @ np.asarray(dry_evaluator.OXIDE_MASSES, dtype=float) / 1000
    water_mass = masses[water_index]
    predictor = nu.copy()
    predictor[:, oxides.index("feo")] += 2 * predictor[:, oxides.index("fe2o3")]
    predictor[:, oxides.index("fe2o3")] = 0
    predictor[:, oxides.index("h2o")] = 0
    b = np.zeros(len(oxides))
    for name, coefficient in zip(("cao", "sio2", "mgo", "al2o3", "feo"), _capacity.B_K):
        b[oxides.index(name)] = coefficient
    counts = jnp.asarray(predictor.sum(axis=1))
    weights = jnp.asarray(predictor @ b)
    oxygen = jnp.asarray(dry_evaluator.FORMULA_MATRIX[:, list(dry_evaluator.ELEMENTS).index("O")])
    masses_jax = jnp.asarray(masses)
    log_a = np.log(_capacity.A_PER_SQRT_BAR * _capacity.WATER_KG_MOL / _capacity.OH_KG_MOL)
    standard_receipts = []
    expression = {
        "schema": "dry_melts_water_equivalent_expression_v1",
        "component_order": names, "water_index": water_index,
        "component_masses_kg_mol": masses.tolist(), "water_mass_kg_mol": float(water_mass),
        "predictor_oxide_counts": np.asarray(counts).tolist(),
        "predictor_temperature_weights_K": np.asarray(weights).tolist(),
        "component_oxygen_counts": np.asarray(oxygen).tolist(), "log_capacity_prefactor": float(log_a),
        "capacity_formula": "log_Cw = log_prefactor + (weights @ dry)/(T_K*(counts @ dry))",
        "gibbs_formula": "G_RT = G_dry_RT + h*(gas_standard_RT - 2*log_Cw) + 2*G_mass_fraction_RT",
        "water_mixing_factor": 2., "water_standard_pressure_Pa": 1e5,
        "domain": "nonnegative amounts; positive dry host; h <= oxygen @ dry",
        "rounding": "JSON numbers are the exact binary64 values used by this provider.",
    }

    @jax.jit
    def water_state(t, n, gas_standard):
        h = n[water_index]
        dry = n.at[water_index].set(0.)
        oxide_total = counts @ dry
        composition_term = weights @ dry / oxide_total
        log_c = log_a + composition_term / t
        dilute = mass_fraction_solute_state(0., jnp.zeros_like(dry), dry, masses_jax,
                                            h, water_mass, 0.)
        standard = gas_standard - 2 * log_c
        dlogc = (weights - composition_term * counts) / (t * oxide_total)
        host_mu = 2 * dilute.host_mu_RT - 2 * h * dlogc
        mu = host_mu.at[water_index].set(standard + 2 * dilute.solute_mu_RT)
        energy = h * standard + 2 * dilute.gibbs_RT
        valid = h <= oxygen @ dry
        return jnp.where(valid, energy, jnp.nan), jnp.where(valid, mu, jnp.nan)

    # A separate scalar derivative is retained for the consumer's fresh audit.
    water_gradient = jax.jit(jax.value_and_grad(lambda n, t, gas: water_state(t, n, gas)[0]))

    def evaluate_liquid(temperature, pressure, component_moles, **options):
        n = np.asarray(component_moles, dtype=float)
        if (n.shape != (len(names),) or np.any(~np.isfinite(n)) or np.any(n < 0)
                or n.sum()-n[water_index] <= 0):
            raise ValueError("Require a nonnegative supplied liquid with positive dry host.")
        dry = n.copy()
        dry[water_index] = 0
        result = dict(dry_evaluator.evaluate_liquid(temperature, pressure, dry, **options))
        dry_properties = copy.deepcopy(result)
        gas = float(gas_water_standard_rt(temperature, pressure))
        added_g, added_mu = water_state(temperature, jnp.asarray(n), gas)
        present = n > 0
        dry_mu = np.asarray(result["mu_RT"], dtype=float)
        dry_mu[water_index] = 0
        mu = dry_mu + np.asarray(added_mu)
        energy = float(result["gibbs_RT"] + added_g)
        if not np.isfinite(energy) or not np.all(np.isfinite(mu[present])):
            raise ValueError("Reconstructed water is outside its mathematical domain.")
        common_r = result["basis"]["common_R_J_mol_K"]
        receipt = {"T_K": float(temperature), "P_Pa": float(pressure),
                   "common_R_J_mol_K": common_r, "H2O_gas_standard_RT": gas,
                   "gas_standard_pressure_Pa": 1e5}
        if receipt not in standard_receipts:
            standard_receipts.append(receipt)
        rt = common_r * temperature
        grams = (nu.T @ n) * np.asarray(dry_evaluator.OXIDE_MASSES)
        nullable = lambda a: [float(v) if yes else None for v, yes in zip(a, present)]
        result.update(model_id=MODEL_ID, component_moles=n.tolist(), returned_component_moles=n.tolist(),
                      x=(n/n.sum()).tolist(), oxide_mass_g=grams.tolist(), returned_oxide_mass_g=grams.tolist(),
                      mass_g=float(grams.sum()), gibbs_J=rt*energy, gibbs_RT=energy,
                      mu_RT=nullable(mu), mu_J_mol=nullable(rt*mu), oxide_mu_J_mol=[None]*len(oxides),
                      oxide_potential_status="Not converted; use complete component potentials.")
        for key in ("mu0_J_mol", "mu0_RT", "activity", "ln_activity", "ln_gamma",
                    "ln_activity_common_R", "ln_gamma_common_R"):
            result[key] = [None]*len(names)
        result.pop("mixing_expression", None)  # This is no longer the hydrated regular-solution model.
        result["basis"] = {**result["basis"],
                           "element_moles": (dry_evaluator.FORMULA_MATRIX.T @ n).tolist(),
                           "mixing": "Dry published MELTS plus integrated H2O-equivalent mass law; one scalar G."}
        result["provenance"] = {**result["provenance"], "model_selection": MODEL_ID,
                               "water_provider_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "capacity_provider_sha256": hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest(),
                               "water_sources_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()}
        result["water_reconstruction"] = {
            "expression": copy.deepcopy(expression), "dry_properties": dry_properties,
            "gas_H2O_standard_RT": gas, "gas_standard_pressure_Pa": 1e5,
            "water_added_gibbs_RT": float(added_g), "native_water_amount_used_mol": 0.,
            "water_mass_fraction": float(n[water_index]*water_mass/(masses @ n)),
            "predictor": "All dry oxide moles, Fe2O3 recast as2FeO only for FeO-total regression.",
            "water_pressure_volume_term": "Not fitted; held zero as the published solubility-law assumption.",
            "accepted_coupled_material_domain": None,
            "scope": "Explicit extrapolative water model; independent pressure replay is not a BSE error bound."}
        return result

    def energy_value_and_grad_rt(temperature, pressure, component_moles, **options):
        n = np.asarray(component_moles, dtype=float)
        evaluate_liquid(temperature, pressure, n, **options)  # Same checked domain.
        dry = n.copy()
        dry[water_index] = 0
        energy, gradient = dry_evaluator.energy_value_and_grad_rt(temperature, pressure, dry, **options)
        gradient = np.asarray(gradient).copy()
        gradient[water_index] = 0
        extra, extra_gradient = water_gradient(jnp.asarray(n), temperature,
                                              float(gas_water_standard_rt(temperature, pressure)))
        gradient += np.asarray(extra_gradient)
        gradient[n == 0] = np.nan
        return float(energy + extra), gradient

    return SimpleNamespace(**{**vars(dry_evaluator), "MODEL_ID": MODEL_ID, "__file__": __file__,
                              "evaluate_liquid": evaluate_liquid,
                              "energy_value_and_grad_rt": energy_value_and_grad_rt,
                              "water_source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
                              "water_standard_receipts": standard_receipts,
                              "water_expression_parameters": expression,
                              "dry_provider_model_id": DRY_MODEL_ID})
