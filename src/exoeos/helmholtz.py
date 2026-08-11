"""Thermodynamic derivatives from residual Helmholtz free energy."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from exoeos.constants import MOLAR_GAS_CONSTANT
from exoeos.contracts import HelmholtzEOS
from exoeos.state import TRhoState


Array = jax.Array


def _common_dtype(eos: HelmholtzEOS, *values: ArrayLike):
    model_dtypes = []
    for leaf in jax.tree_util.tree_leaves(eos):
        leaf_dtype = getattr(leaf, "dtype", None)
        if leaf_dtype is not None and jnp.issubdtype(leaf_dtype, jnp.inexact):
            model_dtypes.append(leaf_dtype)
    return jnp.result_type(*values, *model_dtypes, jnp.float32)


def _scalar_array(value: ArrayLike, name: str) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar; use jax.vmap for batches.")
    return array


def _vector_array(value: ArrayLike, name: str) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; use jax.vmap for batches.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one component.")
    return array


def psir(
    eos: HelmholtzEOS,
    T: ArrayLike,
    rho_vec: ArrayLike,
) -> Array:
    """Return reduced residual Helmholtz energy per volume.

    Args:
        eos: Residual Helmholtz model.
        T: Temperature in K.
        rho_vec: Partial molar densities in mol m^-3.

    Returns:
        ``A^r / (R T V)`` in mol m^-3.
    """

    temperature = _scalar_array(T, "T")
    partial_densities = _vector_array(rho_vec, "rho_vec")
    dtype = _common_dtype(eos, temperature, partial_densities)
    temperature = temperature.astype(dtype)
    partial_densities = partial_densities.astype(dtype)

    molar_density = jnp.sum(partial_densities)
    mole_fractions = partial_densities / molar_density
    alphar = jnp.asarray(eos.alphar(temperature, molar_density, mole_fractions))
    if alphar.ndim != 0:
        raise ValueError("eos.alphar must return a scalar for a single state.")
    return molar_density * alphar


def state_trho(
    eos: HelmholtzEOS,
    T: ArrayLike,
    rho: ArrayLike,
    x: ArrayLike,
) -> TRhoState:
    """Evaluate a residual state at temperature and molar density.

    Args:
        eos: Residual Helmholtz model.
        T: Temperature in K.
        rho: Total molar density in mol m^-3.
        x: Mole fractions with shape ``(K,)``.

    Returns:
        State derived from the model's residual Helmholtz energy.
    """

    temperature = _scalar_array(T, "T")
    molar_density = _scalar_array(rho, "rho")
    mole_fractions = _vector_array(x, "x")
    dtype = _common_dtype(
        eos,
        temperature,
        molar_density,
        mole_fractions,
    )
    temperature = temperature.astype(dtype)
    molar_density = molar_density.astype(dtype)
    mole_fractions = mole_fractions.astype(dtype)
    partial_densities = molar_density * mole_fractions

    energy_density, chemical_potentials = jax.value_and_grad(
        lambda values: psir(eos, temperature, values)
    )(partial_densities)
    pressure_residual_over_rt = (
        jnp.dot(partial_densities, chemical_potentials) - energy_density
    )
    z_minus_one = pressure_residual_over_rt / molar_density
    compressibility_factor = 1.0 + z_minus_one
    log_compressibility = jnp.log1p(z_minus_one)
    reduced_helmholtz = energy_density / molar_density

    return TRhoState(
        molar_density=molar_density,
        pressure=(
            MOLAR_GAS_CONSTANT
            * temperature
            * (molar_density + pressure_residual_over_rt)
        ),
        compressibility_factor=compressibility_factor,
        reduced_residual_helmholtz=reduced_helmholtz,
        reduced_residual_chemical_potentials=chemical_potentials,
        log_fugacity_coefficients=chemical_potentials - log_compressibility,
        reduced_residual_gibbs=(
            reduced_helmholtz + z_minus_one - log_compressibility
        ),
    )
