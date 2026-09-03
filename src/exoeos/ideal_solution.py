"""Ideal symmetric mole-fraction Gibbs-excess model."""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax import tree_util
from jax.typing import ArrayLike

from exoeos._arrays import as_inexact_array


Array = jax.Array


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class IdealSolution:
    """Solution model with zero excess Gibbs energy."""

    activity_basis: ClassVar[str] = "mole_fraction"
    standard_state_convention: ClassVar[str] = "symmetric"

    def gex_RT(
        self,
        T: ArrayLike,
        P: ArrayLike,
        x: ArrayLike,
    ) -> Array:
        """Return zero reduced molar excess Gibbs energy.

        Args:
            T: Temperature in K.
            P: Absolute pressure in Pa.
            x: Mole fractions with shape ``(K,)``.

        Returns:
            A scalar zero in the common input dtype, promoted to at least
            float32.
        """

        temperature = as_inexact_array(T)
        pressure = as_inexact_array(P)
        mole_fractions = as_inexact_array(x)
        if temperature.ndim != 0 or pressure.ndim != 0:
            raise ValueError("T and P must be scalars; use jax.vmap for batches.")
        if mole_fractions.ndim != 1 or mole_fractions.shape[0] == 0:
            raise ValueError("x must be a non-empty one-dimensional array.")
        dtype = jnp.result_type(
            temperature,
            pressure,
            mole_fractions,
            jnp.float32,
        )
        return jnp.zeros((), dtype=dtype)

    def tree_flatten(self):
        """Return the empty JAX PyTree representation."""

        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Restore an ideal solution model from an empty JAX PyTree."""

        del aux_data, children
        return cls()
