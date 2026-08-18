"""Mass-density closures for equation-of-state results."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from exoeos.contracts import TPHelmholtzEOS


Array = jax.Array


def _scalar_array(value: ArrayLike, name: str) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar; use jax.vmap for batches.")
    return array


def _component_vector(value: ArrayLike, name: str) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; use jax.vmap for batches.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one component.")
    return array


def mass_density_tp(
    eos: TPHelmholtzEOS,
    temperature: ArrayLike,
    pressure: ArrayLike,
    mole_fractions: ArrayLike,
    molar_masses: ArrayLike,
    *,
    phase: str = "vapor",
) -> Array:
    """Return mass density in kg m^-3 at temperature and pressure.

    ``molar_masses`` must be ordered like ``mole_fractions`` and use
    kg mol^-1. Numerical domain validation and composition normalization are
    left to the caller.
    """

    temperature_array = _scalar_array(temperature, "temperature")
    pressure_array = _scalar_array(pressure, "pressure")
    composition = _component_vector(mole_fractions, "mole_fractions")
    masses = _component_vector(molar_masses, "molar_masses")
    if composition.shape != masses.shape:
        raise ValueError(
            "molar_masses must have the same shape as mole_fractions; "
            f"received {masses.shape} and {composition.shape}."
        )

    dtype = jnp.result_type(
        temperature_array,
        pressure_array,
        composition,
        masses,
        jnp.float32,
    )
    temperature_array = temperature_array.astype(dtype)
    pressure_array = pressure_array.astype(dtype)
    composition = composition.astype(dtype)
    masses = masses.astype(dtype)

    molar_density = jnp.asarray(
        eos.molar_density(
            temperature_array,
            pressure_array,
            composition,
            phase=phase,
        )
    )
    if molar_density.ndim != 0:
        raise ValueError("eos.molar_density must return a scalar for a single state.")

    dtype = jnp.result_type(molar_density, composition, masses)
    molar_density = molar_density.astype(dtype)
    composition = composition.astype(dtype)
    masses = masses.astype(dtype)
    return molar_density * jnp.sum(composition * masses)


def additive_volume_mass_density(
    mass_fractions: ArrayLike,
    component_mass_densities: ArrayLike,
) -> Array:
    """Return mixture density from ``1 / rho = sum_i(w_i / rho_i)``.

    Component mass densities must use kg m^-3. Numerical domain validation and
    mass-fraction normalization are left to the caller.
    """

    fractions = _component_vector(mass_fractions, "mass_fractions")
    densities = _component_vector(
        component_mass_densities,
        "component_mass_densities",
    )
    if fractions.shape != densities.shape:
        raise ValueError(
            "component_mass_densities must have the same shape as mass_fractions; "
            f"received {densities.shape} and {fractions.shape}."
        )

    dtype = jnp.result_type(fractions, densities, jnp.float32)
    fractions = fractions.astype(dtype)
    densities = densities.astype(dtype)
    return jnp.reciprocal(jnp.sum(fractions / densities))
