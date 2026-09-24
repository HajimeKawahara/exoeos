"""Extensive solute dilution on a total-liquid mass-fraction basis."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy
from jax.typing import ArrayLike


class MassFractionSoluteState(NamedTuple):
    """Full G/(RT) in mol and host/solute chemical potentials divided by RT."""

    gibbs_RT: jax.Array
    host_mu_RT: jax.Array
    solute_mu_RT: jax.Array


def mass_fraction_solute_state(
    host_gibbs_RT: ArrayLike,
    host_mu_RT: ArrayLike,
    host_amounts_mol: ArrayLike,
    host_molar_masses_kg_mol: ArrayLike,
    solute_amount_mol: ArrayLike,
    solute_molar_mass_kg_mol: ArrayLike,
    solute_standard_RT: ArrayLike,
) -> MassFractionSoluteState:
    """Add a mass-fraction solute law to an already mixed host's full energy.

    Set S = sum(W_i n_i)/W_s and h = n_s. The added scalar is
    h*mu0 + S*ln(S/(S+h)) + h*ln(h/(S+h)). Consequently mu_s/RT
    is mu0 + ln(w_s), where w_s is solute mass / total liquid mass.
    Each host potential also acquires (W_i/W_s)*ln(1-w_s).

    The host G and mu must be consistent and use the same R and T. Its
    existing mixing and pressure terms are not duplicated. The supplied
    solute standard must be independent of host composition; this helper
    neither fits a Henry coefficient nor transfers an experimental host
    calibration. For molecular H2 versus atomic H, use the corresponding
    molar mass and amount basis explicitly.

    The supported domain has positive host mass and nonnegative solute.
    Exactly zero solute retains the original host and an insertion mu=-inf.
    Pure solute and invalid values return NaNs. Static shape mismatches raise
    ValueError; positive-interior states support JAX differentiation/JIT/VMAP.
    """
    n, weights, host_mu = map(jnp.asarray, (host_amounts_mol,
                                          host_molar_masses_kg_mol, host_mu_RT))
    g, h, weight, standard = map(jnp.asarray, (host_gibbs_RT, solute_amount_mol,
                                             solute_molar_mass_kg_mol, solute_standard_RT))
    if n.ndim != 1 or not n.size or weights.shape != n.shape or host_mu.shape != n.shape:
        raise ValueError("Host amounts, molar masses and potentials must be matching nonempty vectors.")
    if any(value.ndim for value in (g, h, weight, standard)):
        raise ValueError("Host energy, solute amount, molar mass and standard must be scalars.")
    if any(jnp.issubdtype(value.dtype, jnp.complexfloating) for value in (n, weights, host_mu, g, h, weight, standard)):
        raise ValueError("Mass-fraction dilution requires real inputs.")
    equivalent_host = jnp.dot(weights, n) / weight
    total = equivalent_host + h
    log_host_fraction = -jnp.log1p(h / equivalent_host)
    solute_fraction = h / total
    mixing = equivalent_host * log_host_fraction + xlogy(h, solute_fraction)
    valid = (jnp.all(jnp.isfinite(n)) & jnp.all(n >= 0)
             & jnp.all(jnp.isfinite(weights)) & jnp.all(weights > 0)
             & jnp.isfinite(weight) & (weight > 0)
             & jnp.isfinite(equivalent_host) & (equivalent_host > 0)
             & jnp.isfinite(h) & (h >= 0) & jnp.isfinite(total)
             & jnp.isfinite(g) & jnp.isfinite(standard)
             & jnp.all(jnp.where(n > 0, jnp.isfinite(host_mu), True)))
    return MassFractionSoluteState(
        jnp.where(valid, g + h * standard + mixing, jnp.nan),
        jnp.where(valid, host_mu + weights / weight * log_host_fraction, jnp.nan),
        jnp.where(valid, standard + jnp.log(solute_fraction), jnp.nan),
    )
