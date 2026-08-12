"""Thermodynamic derivatives from molar excess Gibbs energy."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from exoeos.contracts import GibbsExcessModel
from exoeos.state import SolutionState


Array = jax.Array


def _common_dtype(model: GibbsExcessModel, *values: ArrayLike):
    model_dtypes = []
    for leaf in jax.tree_util.tree_leaves(model):
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


def total_gex_RT(
    model: GibbsExcessModel,
    T: ArrayLike,
    P: ArrayLike,
    n: ArrayLike,
) -> Array:
    """Return total excess Gibbs energy divided by ``R T``.

    Args:
        model: Symmetric mole-fraction excess Gibbs-energy model.
        T: Temperature in K.
        P: Absolute pressure in Pa.
        n: Component amounts with shape ``(K,)``.

    Returns:
        ``G^E / (R T) = n_total g^E / (R T)``, in the same amount unit as
        ``n``.
    """

    temperature = _scalar_array(T, "T")
    pressure = _scalar_array(P, "P")
    amounts = _vector_array(n, "n")
    dtype = _common_dtype(model, temperature, pressure, amounts)
    temperature = temperature.astype(dtype)
    pressure = pressure.astype(dtype)
    amounts = amounts.astype(dtype)

    total_amount = jnp.sum(amounts)
    mole_fractions = amounts / total_amount
    reduced_molar_excess_gibbs = jnp.asarray(
        model.gex_RT(temperature, pressure, mole_fractions)
    )
    if reduced_molar_excess_gibbs.ndim != 0:
        raise ValueError("model.gex_RT must return a scalar for a single state.")
    return total_amount * reduced_molar_excess_gibbs


def solution_state(
    model: GibbsExcessModel,
    T: ArrayLike,
    P: ArrayLike,
    x: ArrayLike,
) -> SolutionState:
    """Evaluate excess Gibbs energy and activity coefficients.

    ``x`` is interpreted as the component amounts of a normalized one-mole
    system. The extensive value is divided by the supplied total amount for
    the molar state field, while its amount derivative gives ``ln(gamma_i)``.

    Args:
        model: Symmetric mole-fraction excess Gibbs-energy model.
        T: Temperature in K.
        P: Absolute pressure in Pa.
        x: Normalized mole fractions with shape ``(K,)``.

    Returns:
        Excess Gibbs state under the symmetric mole-fraction convention.
    """

    temperature = _scalar_array(T, "T")
    pressure = _scalar_array(P, "P")
    mole_fractions = _vector_array(x, "x")
    dtype = _common_dtype(model, temperature, pressure, mole_fractions)
    temperature = temperature.astype(dtype)
    pressure = pressure.astype(dtype)
    mole_fractions = mole_fractions.astype(dtype)

    total_amount = jnp.sum(mole_fractions)
    total_excess_gibbs, log_activity_coefficients = jax.value_and_grad(
        lambda amounts: total_gex_RT(model, temperature, pressure, amounts)
    )(mole_fractions)
    reduced_excess_gibbs = total_excess_gibbs / total_amount

    return SolutionState(
        reduced_excess_gibbs=reduced_excess_gibbs,
        log_activity_coefficients=log_activity_coefficients,
    )
