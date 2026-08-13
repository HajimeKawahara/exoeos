"""Density-form second-virial equation of state."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util
from jax.typing import ArrayLike

from exoeos.constants import MOLAR_GAS_CONSTANT


Array = jax.Array


def _scalar_array(value: ArrayLike, name: str) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar; use jax.vmap for batches.")
    return array


def _composition_array(value: ArrayLike, component_count: int) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.ndim != 1 or array.shape[0] != component_count:
        raise ValueError(f"x must have shape ({component_count},).")
    return array


@tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class SecondVirialEOS:
    """Second-virial EOS with constant symmetric pair coefficients.

    ``coefficients[i, j]`` is :math:`B_{ij}` in m^3 mol^-1. Symmetry and the
    numerical validity of state inputs are caller contracts; static shapes are
    checked by the implementation.
    """

    coefficients: Array

    def __init__(self, coefficients: ArrayLike) -> None:
        matrix = jnp.asarray(coefficients)
        if not jnp.issubdtype(matrix.dtype, jnp.inexact):
            matrix = matrix.astype(jnp.asarray(1.0).dtype)
        if (
            matrix.ndim != 2
            or matrix.shape[0] == 0
            or matrix.shape[0] != matrix.shape[1]
        ):
            raise ValueError("coefficients must be a non-empty square matrix.")
        object.__setattr__(self, "coefficients", matrix)

    @property
    def component_count(self) -> int:
        """Number of mixture components."""

        return self.coefficients.shape[0]

    def _inputs(
        self,
        T: ArrayLike,
        value: ArrayLike,
        value_name: str,
        x: ArrayLike,
    ) -> tuple[Array, Array, Array]:
        temperature = _scalar_array(T, "T")
        scalar_value = _scalar_array(value, value_name)
        mole_fractions = _composition_array(x, self.component_count)
        dtype = jnp.result_type(
            temperature,
            scalar_value,
            mole_fractions,
            self.coefficients,
            jnp.float32,
        )
        return (
            temperature.astype(dtype),
            scalar_value.astype(dtype),
            mole_fractions.astype(dtype),
        )

    def alphar(
        self,
        T: ArrayLike,
        rho: ArrayLike,
        x: ArrayLike,
    ) -> Array:
        """Return ``A^r / (n R T) = rho * B_mix``."""

        _, molar_density, mole_fractions = self._inputs(T, rho, "rho", x)
        mixture_coefficient = jnp.einsum(
            "i,ij,j->",
            mole_fractions,
            self.coefficients,
            mole_fractions,
        )
        return molar_density * mixture_coefficient

    def molar_density(
        self,
        T: ArrayLike,
        P: ArrayLike,
        x: ArrayLike,
        phase: str = "vapor",
    ) -> Array:
        """Return the mechanically stable low-density root in mol m^-3."""

        if phase != "vapor":
            raise ValueError("SecondVirialEOS supports only phase='vapor'.")
        temperature, pressure, mole_fractions = self._inputs(T, P, "P", x)
        mixture_coefficient = jnp.einsum(
            "i,ij,j->",
            mole_fractions,
            self.coefficients,
            mole_fractions,
        )
        ideal_density = pressure / (MOLAR_GAS_CONSTANT * temperature)
        discriminant = 1.0 + 4.0 * mixture_coefficient * ideal_density
        return 2.0 * ideal_density / (1.0 + jnp.sqrt(discriminant))

    def tree_flatten(self):
        """Return the JAX PyTree representation."""

        return (self.coefficients,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Restore a second-virial EOS from its JAX PyTree leaves."""

        del aux_data
        (coefficients,) = children
        return cls(coefficients)
