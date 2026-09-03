"""Thermodynamic derivatives from residual Helmholtz free energy."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from exoeos._arrays import common_dtype as _common_dtype
from exoeos._arrays import scalar_array as _scalar_array
from exoeos._arrays import vector_array as _vector_array
from exoeos.constants import MOLAR_GAS_CONSTANT
from exoeos.contracts import HelmholtzEOS, TPHelmholtzEOS
from exoeos.state import TRhoState


Array = jax.Array


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


def state_tp(
    eos: TPHelmholtzEOS,
    T: ArrayLike,
    P: ArrayLike,
    x: ArrayLike,
    phase: str = "vapor",
) -> TRhoState:
    """Evaluate a residual state at temperature and pressure.

    Density inversion and phase/root selection are delegated to the EOS. The
    resulting density is evaluated through :func:`state_trho`, so both entry
    points share the same thermodynamic derivatives and state type.

    Args:
        eos: Residual Helmholtz model with temperature-pressure inversion.
        T: Temperature in K.
        P: Absolute pressure in Pa.
        x: Mole fractions with shape ``(K,)``.
        phase: Static phase/root selector understood by the EOS.

    Returns:
        Residual state including ``rho``, ``Z``, ``lnphi``, and ``gres_RT``.
    """

    temperature = _scalar_array(T, "T")
    pressure = _scalar_array(P, "P")
    mole_fractions = _vector_array(x, "x")
    dtype = _common_dtype(eos, temperature, pressure, mole_fractions)
    temperature = temperature.astype(dtype)
    pressure = pressure.astype(dtype)
    mole_fractions = mole_fractions.astype(dtype)

    molar_density = jnp.asarray(
        eos.molar_density(
            temperature,
            pressure,
            mole_fractions,
            phase=phase,
        )
    )
    if molar_density.ndim != 0:
        raise ValueError("eos.molar_density must return a scalar for a single state.")

    return state_trho(eos, temperature, molar_density, mole_fractions)
