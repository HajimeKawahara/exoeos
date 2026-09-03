"""Shared private helpers for normalizing JAX array inputs."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


Array = jax.Array


def as_inexact_array(value: ArrayLike, *, atleast_1d: bool = False) -> Array:
    """Return an array whose exact inputs use the minimum real dtype."""

    array = jnp.asarray(value)
    if atleast_1d:
        array = jnp.atleast_1d(array)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        # Exact dtypes should not promote explicitly float32 peer inputs under
        # JAX's x64 mode. A later result_type call can still promote as needed.
        array = array.astype(jnp.float32)
    return array


def scalar_array(value: ArrayLike, name: str) -> Array:
    """Return an inexact scalar, preserving the public shape error contract."""

    array = as_inexact_array(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar; use jax.vmap for batches.")
    return array


def vector_array(value: ArrayLike, name: str) -> Array:
    """Return a non-empty inexact component vector."""

    array = as_inexact_array(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; use jax.vmap for batches.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one component.")
    return array


def common_dtype(tree: object, *values: ArrayLike):
    """Return a floating result dtype that includes inexact PyTree leaves."""

    model_dtypes = []
    for leaf in jax.tree_util.tree_leaves(tree):
        leaf_dtype = getattr(leaf, "dtype", None)
        if leaf_dtype is not None and jnp.issubdtype(leaf_dtype, jnp.inexact):
            model_dtypes.append(leaf_dtype)
    return jnp.result_type(*values, *model_dtypes, jnp.float32)
