"""Extensive solution energy with explicitly supplied standard potentials."""

import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy
from jax.typing import ArrayLike

from exoeos._arrays import common_dtype, scalar_array, vector_array
from exoeos.contracts import GibbsExcessModel
from exoeos.gibbs_excess import solution_state, total_gex_RT
from exoeos.state import TotalSolutionState


def _inputs(model, T, P, n, mu0_RT):
    temperature, pressure = scalar_array(T, "T"), scalar_array(P, "P")
    amounts, standards = vector_array(n, "n"), vector_array(mu0_RT, "mu0_RT")
    if amounts.shape != standards.shape or amounts.size == 0:
        raise ValueError("n and mu0_RT must have the same nonempty vector shape.")
    dtype = common_dtype(model, temperature, pressure, amounts, standards)
    return tuple(value.astype(dtype) for value in (temperature, pressure, amounts, standards))


def total_solution_gibbs_RT(
    model: GibbsExcessModel, T: ArrayLike, P: ArrayLike,
    n: ArrayLike, mu0_RT: ArrayLike,
) -> jax.Array:
    """Return ``G/(RT)`` from standards, ideal mixing, and excess energy.

    ``T`` is in K, ``P`` in Pa, and ``n`` is a nonnegative component vector
    in mol. The result has units of mol. ``mu0_RT`` must use the model's
    symmetric endmember standards at the supplied T/P and the same gas
    constant as ``gex_RT``. No standard-state alignment is inferred.

    A completely absent phase has exactly zero energy; it is never
    normalized. Zero individual amounts use the continuous ``n log(n)``
    limit. Derivatives are supported only inside the model's differentiable
    domain, with positive amounts for components being differentiated.
    In particular, a derivative at the all-zero vector is not a chemical
    potential. Use ``total_solution_state`` for explicit boundary semantics.
    Only static shapes are validated here; validate material inputs eagerly.
    """
    temperature, pressure, amounts, standards = _inputs(model, T, P, n, mu0_RT)

    def present(values):
        fractions = values / jnp.sum(values)
        return (jnp.dot(values, standards) + jnp.sum(xlogy(values, fractions))
                + total_gex_RT(model, temperature, pressure, values))

    return jax.lax.cond(
        jnp.all(amounts == 0), lambda values: jnp.zeros((), dtype=values.dtype),
        present, amounts,
    )


def total_solution_state(
    model: GibbsExcessModel, T: ArrayLike, P: ArrayLike,
    n: ArrayLike, mu0_RT: ArrayLike,
) -> TotalSolutionState:
    """Return extensive ``G/(RT)`` and component ``mu/(RT)`` from one model.

    Units and validation match ``total_solution_gibbs_RT``. An absent phase
    returns zero energy and NaN potentials, because it has no composition.
    At a present phase's supported component boundary the ideal contribution
    to the absent component potential is minus infinity. No amount floor is
    inserted and unsupported excess-model endpoints remain unsupported.
    """
    temperature, pressure, amounts, standards = _inputs(model, T, P, n, mu0_RT)
    energy = total_solution_gibbs_RT(model, temperature, pressure, amounts, standards)

    def present(values):
        fractions = values / jnp.sum(values)
        excess = solution_state(model, temperature, pressure, fractions)
        return standards + jnp.log(fractions) + excess.lngamma

    potentials = jax.lax.cond(
        jnp.all(amounts == 0), lambda values: jnp.full_like(values, jnp.nan),
        present, amounts,
    )
    return TotalSolutionState(energy, potentials)
