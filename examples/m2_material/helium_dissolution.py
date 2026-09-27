"""One extensive trace-He scalar on a declared dry-host Henry capacity."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import xlogy


MODELS = {"guillot2012_olivine": "olivine", "guillot2012_morb": "MORB",
          "guillot2012_rhyolite": "rhyolite"}


def helium_dissolution_gibbs_rt(amounts, dry_molar_masses, capacity, gas_standard_rt):
    """Extensive G/RT, with He last and dry mass excluding host water/H2.

    The trace-solute model uses n_He/(kg dry host), not a He-inclusive mole
    fraction. It adds no separate finite-concentration solvent dilution.
    """
    n = jnp.asarray(amounts)
    mass = jnp.dot(n[:-1], jnp.asarray(dry_molar_masses))
    he = n[-1]
    safe_mass = jnp.where(mass > 0, mass, 1.)
    value = (he * (gas_standard_rt - 1. - jnp.log(safe_mass * capacity))
             + xlogy(he, jnp.where(he > 0, he, 1.)))
    return jnp.where((mass == 0) & (he > 0), jnp.inf, value)


def make_helium_dissolution(model, temperature_K, dry_host_molar_masses_kg,
                            gas_standard_rt):
    """Freeze provider coefficients; return analytic potentials and scalar AD.

    No pressure-volume correction or wet-host calibration is inferred.
    A zero-He endpoint has zero G/host shifts and insertion mu=-infinity.
    An entirely absent phase has zero G and undefined potentials; positive He
    without positive dry-host mass is outside this model's domain.
    """
    if model not in MODELS:
        raise ValueError("Select a declared Guillot 2012 dry-host He model")
    masses = np.asarray(dry_host_molar_masses_kg, dtype=float)
    if (masses.ndim != 1 or not masses.size or np.any(~np.isfinite(masses))
            or np.any(masses < 0) or not np.any(masses > 0)):
        raise ValueError("Require finite nonnegative dry masses with a positive component")
    standard = float(gas_standard_rt)
    if not np.isfinite(standard):
        raise ValueError("The He gas standard must be finite")
    path = Path(__file__).with_name("helium_reference.py")
    spec = importlib.util.spec_from_file_location("_helium_capacity_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    coefficient = reference.helium_henry_coefficient(MODELS[model], temperature_K)
    capacity = coefficient["He_mol_per_kg_dry_host_per_bar_fugacity"]

    def validate(amounts):
        n = np.asarray(amounts, dtype=float)
        if n.shape != (len(masses) + 1,) or np.any(~np.isfinite(n)) or np.any(n < 0):
            raise ValueError("Supply finite nonnegative host amounts followed by He")
        mass = float(n[:-1] @ masses)
        if mass <= 0 and n.sum() > 0:
            raise ValueError("A present He-bearing phase requires positive dry-host mass")
        return n, mass

    def state(amounts):
        n, mass = validate(amounts)
        if n.sum() == 0:
            return {"gibbs_rt": 0., "mu_rt": np.full_like(n, np.nan)}
        he = n[-1]
        if he == 0:
            return {"gibbs_rt": 0., "mu_rt": np.r_[np.zeros_like(masses), -np.inf]}
        chemical_potential = standard + np.log(he / (mass * capacity))
        return {"gibbs_rt": float(he * (chemical_potential - 1.)),
                "mu_rt": np.r_[-he * masses / mass, chemical_potential]}

    scalar = lambda n: helium_dissolution_gibbs_rt(n, masses, capacity, standard)
    derivative = jax.jit(jax.value_and_grad(scalar))

    def energy_value_and_grad_rt(amounts):
        n, mass = validate(amounts)
        if n.sum() == 0 or n[-1] == 0:
            endpoint = state(n)
            return endpoint["gibbs_rt"], endpoint["mu_rt"]
        energy, gradient = derivative(n)
        return float(energy), np.asarray(gradient)

    files = [Path(__file__), path, reference.DATA]
    receipt = {"model": model, "capacity": coefficient,
               "gas_standard_rt": standard,
               "dry_host_molar_masses_kg": masses.tolist(),
               "amount_basis": "Host component mol followed by atomic He mol; dry host excludes water and molecular H2.",
               "scalar": "nHe*(muHe_gas0_RT + ln(nHe/(M_dry*a)) - 1)",
               "pressure_standard_bar": 1.,
               "pressure_policy": "No condensed partial-molar-volume correction; gas pressure enters through equilibrium fugacity.",
               "host_policy": "Declared pure dry simulated-host capacity applied to the supplied dry-host mass; not an empirical BSE uncertainty bound.",
               "temperature_policy": "Reciprocal-temperature interpolation within the selected host's simulated interval; extrapolation rejected.",
               "experimental_scope": "Interpolation of simulations is not interpolation of measured BSE solubility. The retained low-pressure basalt benchmark is at 1673.15 K; no wet-BSE empirical transfer error is assigned.",
               "provider_recipe_file_sha256": {
                   "examples/m2_material/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
               "supports_material_admission": False}
    return SimpleNamespace(state=state, energy_value_and_grad_rt=energy_value_and_grad_rt,
                           scalar=scalar, receipt=receipt)
