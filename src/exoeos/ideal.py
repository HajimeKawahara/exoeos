"""Ideal residual Helmholtz equation of state."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util
from jax.typing import ArrayLike


Array = jax.Array


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class IdealEOS:
    """Equation of state with zero residual Helmholtz energy."""

    def alphar(
        self,
        T: ArrayLike,
        rho: ArrayLike,
        x: ArrayLike,
    ) -> Array:
        """Return zero reduced residual Helmholtz energy.

        Args:
            T: Temperature in K.
            rho: Total molar density in mol m^-3.
            x: Mole fractions with shape ``(K,)``.

        Returns:
            A scalar zero in the common input dtype, promoted to at least float32.
        """

        temperature = jnp.asarray(T)
        molar_density = jnp.asarray(rho)
        mole_fractions = jnp.asarray(x)
        if temperature.ndim != 0 or molar_density.ndim != 0:
            raise ValueError("T and rho must be scalars; use jax.vmap for batches.")
        if mole_fractions.ndim != 1 or mole_fractions.shape[0] == 0:
            raise ValueError("x must be a non-empty one-dimensional array.")
        dtype = jnp.result_type(
            temperature,
            molar_density,
            mole_fractions,
            jnp.float32,
        )
        return jnp.zeros((), dtype=dtype)

    def tree_flatten(self):
        """Return the empty JAX PyTree representation."""

        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Restore an ideal EOS from an empty JAX PyTree."""

        del aux_data, children
        return cls()
