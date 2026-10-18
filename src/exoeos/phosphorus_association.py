"""Finite Fe-P liquid association and an explicit multicomponent extension.

The five parameters follow Oikawa & Ueshima (2026), Table 1 and Table S1,
doi:10.2355/tetsutohagane.TETSU-2025-090. Fe2P is a liquid associate, not a
separate solid phase. This module supplies G; the caller equilibrates species.
The extension to the twenty-component host is conditional, not calibrated by
the binary paper. It adds no phosphorus composition constraint or pressure law.
"""

import jax.numpy as jnp

from . import associated_metal
from ._conditional_data import R
from .ma_fe_si_o import MaFeSiOLiquid

COMPONENTS = associated_metal.COMPONENTS + ("Fe2P",)
FORMULAS = associated_metal.FORMULAS + ({"Fe": 2., "P": 1.},)
MODEL_ID = "oikawa2026_fe2p_henry_anchored_conditional_host_v1"


def parameters_j_mol(temperature_k):
    """Return association G, Fe/Fe2P L0,L1 and Fe2P/P L0,L1 [J/mol].

    These are the published binary coefficients, not new fitted values.
    Their evaluability at a temperature does not establish experimental scope.
    """
    t = jnp.asarray(temperature_k)
    return jnp.array([-147357.+27.537*t, -27783.+8.676*t,
                      -29128.+18.174*t, -31649., -37695.])


def henry_shift_rt(temperature_k):
    """Convert a binary infinite-dilution P standard to free-liquid P.

    K=exp(-(association_G+L0_Fe_Fe2P+L1_Fe_Fe2P)/RT) is the dilute
    Fe2P/free-P ratio. The equilibrated Henry standard is mu_P_free-ln(1+K).
    This exact reference conversion preserves the caller's measured gas
    anchor; it is not the paper's absolute SGTE/SSUB standard reconstruction.
    """
    p = parameters_j_mol(temperature_k)
    return jnp.logaddexp(0., -(p[0]+p[1]+p[2])/(R*temperature_k))


def binary_nonideal_j(temperature_k, amounts_mol):
    """Return association plus Redlich-Kister G for (Fe, P, Fe2P) [J].

    Amounts count chemical species. One Fe2P contributes two Fe and one P
    atoms. Ideal species entropy and elemental standards are excluded.
    """
    n = jnp.asarray(amounts_mol)
    if n.shape != (3,):
        raise ValueError("Require Fe, P, Fe2P species amounts.")
    total = jnp.sum(n)
    fe, p, a = n/jnp.where(total > 0, total, 1.)
    dg, l0, l1, m0, m1 = parameters_j_mol(temperature_k)
    return n[2]*dg+total*(fe*a*(l0+l1*(fe-a))+a*p*(m0+m1*(a-p)))


def collapse_amounts(amounts_mol):
    """Return the twenty-component ledger after dissociating Fe2P exactly."""
    n = jnp.asarray(amounts_mol)
    if n.shape != (21,):
        raise ValueError("Require the twenty parent species followed by Fe2P.")
    return n[:20].at[0].add(2*n[20]).at[4].add(n[20])


def _ideal(n):
    total = jnp.sum(n)
    return jnp.sum(n*jnp.log(jnp.where(n > 0, n/jnp.where(total > 0, total, 1.), 1.)))


def binary_gibbs_j(temperature_k, amounts_mol, elemental_standards_rt):
    """Return full binary G for Fe/P/Fe2P and supplied pure Fe/P standards.

    Equilibrium and atomic constraints remain external. For a Henry P input,
    the caller first adds ``henry_shift_rt`` to that input's P entry.
    """
    n, standard = jnp.asarray(amounts_mol), jnp.asarray(elemental_standards_rt)
    if n.shape != (3,) or standard.shape != (2,):
        raise ValueError("Require three chemical amounts and two elemental standards.")
    atoms = jnp.array([n[0]+2*n[2], n[1]+n[2]])
    return R*temperature_k*(atoms @ standard+_ideal(n))+binary_nonideal_j(temperature_k, n)


def _host_excess(t, p, collapsed, epsilon, matrix, *, self_interaction):
    # Retain the original pseudo-Fe Na and finite-host perspectives. Bound Fe
    # and P enter host/cross terms through their conserved collapsed amounts.
    core = collapsed[:18].at[0].add(collapsed[19])
    dry = jnp.sum(core[:3])
    x = jnp.where(dry > 0, core[:3]/jnp.where(dry > 0, dry, 1.), jnp.array([1., 0., 0.]))
    host = dry*MaFeSiOLiquid().gex_RT(t, p, x)
    five, total = jnp.sum(core[:5]), jnp.sum(core)
    phosphorus = core[4]*(epsilon[1:] @ core[1:4])
    if self_interaction:
        phosphorus += .5*epsilon[0]*core[4]**2
    return (host+phosphorus/jnp.where(five > 0, five, 1.)
            +.5*(core @ matrix @ core)/jnp.where(total > 0, total, 1.))


def gibbs_j(temperature_k, pressure_pa, amounts_mol, standard_potentials_rt,
            phosphorus_epsilon, additional_matrix, *, model):
    """Return an extensive conditional alloy G with explicit finite-P choice.

    ``quadratic`` is the unchanged parent scalar, continuously extended to
    its pure-P face. ``association`` uses 21 species, replaces only the P-P
    self term with the published Fe-P association scalar, and preserves the
    binary Henry P standard. Cross interactions use collapsed Fe/P amounts;
    the published binary RK terms use actual free Fe/P/Fe2P species, not Na.
    Global ideal mixing is counted once. Unmeasured multicomponent effects,
    pressure continuation and other host constraints remain caller concerns.
    """
    if model not in ("quadratic", "association"):
        raise ValueError("Select quadratic or association explicitly.")
    n, standard = jnp.asarray(amounts_mol), jnp.asarray(standard_potentials_rt)
    epsilon, matrix = jnp.asarray(phosphorus_epsilon), jnp.asarray(additional_matrix)
    if (n.shape != ((21,) if model == "association" else (20,)) or standard.shape != (20,)
            or epsilon.shape != (4,) or matrix.shape != (18, 18)):
        raise ValueError("Require matching alloy amounts, twenty standards, four P terms and an 18x18 matrix.")
    collapsed = collapse_amounts(n) if model == "association" else n
    energy = R*temperature_k*(collapsed @ standard+_ideal(n)+_host_excess(
        temperature_k, pressure_pa, collapsed, epsilon, matrix, self_interaction=model == "quadratic"))
    if model == "association":
        energy += (R*temperature_k*collapsed[4]*henry_shift_rt(temperature_k)
                   +binary_nonideal_j(temperature_k, n[jnp.array([0, 4, 20])]))
    return energy
