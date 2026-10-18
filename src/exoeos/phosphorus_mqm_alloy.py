"""Conditional twenty-component host with an equilibrated binary Fe-P MQM.

This is an explicit extension, not a published multicomponent assessment.
All equilibrium and atom constraints belong to the caller. No P cap is added.
"""

import jax.numpy as jnp

from . import associated_metal, phosphorus_mqm as binary
from ._conditional_data import R
from .phosphorus_association import _host_excess

COMPONENTS = ("FeFe",)+associated_metal.COMPONENTS[1:4]+("PP",)+associated_metal.COMPONENTS[5:]+("FeP",)
PAIR_INDICES = (0, 4, 20)


def collapse_matrix(*, model):
    """Map 21 coordinates to the conserved twenty-component chemical ledger.

    FeFe replaces Fe at index 0; PP replaces P at index 4; FeP is last.
    Other coordinates still count chemical molecules, not pairs or atoms.
    """
    a = binary.atom_matrix(model=model)
    c = jnp.column_stack((jnp.eye(20), jnp.zeros(20)))
    c = c.at[0, 0].set(a[0, 0]).at[4, 4].set(a[1, 1])
    return c.at[0, 20].set(a[0, 2]).at[4, 20].set(a[1, 2])


def gibbs_j(temperature_k, pressure_pa, amounts_mol, pure_standards_rt,
            phosphorus_epsilon, additional_matrix, *, model, delta_v_p_m3_mol):
    """Extensive G [J]; standards use pure liquid Fe/P, not Henry P.

    Global ideal mixing uses the collapsed chemical ledger. Replace its
    binary Fe/P entropy by the complete MQM expression, exactly once. Host
    and P-cross interactions retain their old collapsed perspectives; the
    old quadratic P-self term is removed. Na remains pseudo-Fe only inside
    the host excess, never inside the binary pair polynomial.

    The caller must choose the constant P partial-volume *increment* relative
    to the supplied standards. Add n_P*(P-1 bar)*delta_v, not a second total
    alloy volume. Zero is a declared continuation, not a measured P law.
    """
    n, s = jnp.asarray(amounts_mol), jnp.asarray(pure_standards_rt)
    epsilon, matrix = jnp.asarray(phosphorus_epsilon), jnp.asarray(additional_matrix)
    if (n.shape != (21,) or s.shape != (20,) or epsilon.shape != (4,)
            or matrix.shape != (18, 18)):
        raise ValueError("Require 21 coordinates, 20 standards, four P terms and an 18x18 matrix.")
    collapsed = collapse_matrix(model=model) @ n
    pairs = n[jnp.array(PAIR_INDICES)]
    rt = R*temperature_k
    energy = rt*(collapsed @ s+binary._ideal(collapsed)
                 +_host_excess(temperature_k, pressure_pa, collapsed, epsilon,
                               matrix, self_interaction=False)
                 -binary._ideal(collapsed[jnp.array([0, 4])]))
    energy += binary.gibbs_j(temperature_k, pairs, jnp.zeros(2), model=model)
    return energy+collapsed[4]*(pressure_pa-1e5)*delta_v_p_m3_mol
