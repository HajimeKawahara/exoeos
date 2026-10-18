"""You (2020) Table 3.2 stoichiometric phosphide candidates.

Molar energies count formula units. They share the SGTE alpha-Fe/white-P
SER convention. HT-Fe3P alone is provided, not its low-temperature polymorph,
solid solutions, a complete phase diagram, or an empirical high-pressure law.
"""

import jax.numpy as jnp

from . import phosphorus_mqm as reference
from ._conditional_data import R

PHASES = ("Fe3P_HT", "Fe2P", "FeP", "FeP2")
# Rows: Fe and P atoms; columns: formula units in PHASES order.
ATOM_MATRIX = ((3., 2., 1., 1.), (1., 1., 1., 2.))


def _cp_integrals(t0, t1, a, b, c, d=0.):
    """Integrals of Cp and Cp/T for Cp=a+b*T+c/T**2+d*T**2."""
    h = a*(t1-t0)+b*(t1*t1-t0*t0)/2-c*(1/t1-1/t0)+d*(t1**3-t0**3)/3
    s = a*jnp.log(t1/t0)+b*(t1-t0)-c*(1/t1**2-1/t0**2)/2+d*(t1*t1-t0*t0)/2
    return h, s


def _fe3p_magnetic_j(t):
    # Eq.3.4 is per mole atoms; the Fe3P formula contains four atoms.
    tau, p = t/713., .28
    denominator = 518/1125+11692/15975*(1/p-1)
    low = 1-(79/(140*p*tau)+474/497*(1/p-1)
             *(tau**3/6+tau**9/135+tau**15/600))/denominator
    high = -(tau**-5/10+tau**-15/315+tau**-25/1500)/denominator
    return 4*R*t*jnp.log(1.6)*jnp.where(tau <= 1, low, high)


def standard_gibbs_j(temperature_k, *, high_temperature_continuation=False):
    """Return four G values [J/mol formula] on the original SER convention.

    Integrate the tabulated Cp from 298.15 K with the stated Hf and S.
    HT-Fe3P adds 1000*(1-T/698) and the four-atom magnetic contribution.
    The explicit Fe3P upper Cp endpoint is 2000 K; no endpoints are printed
    for the other three compounds. Conservatively require an explicit flag
    above 2000 K for this candidate set, continuing only the final Cp branch
    up to 3000 K. Neither flag nor mathematical range establishes stability
    or measured accuracy. Below 298.15 K and above 3000 K return NaN.
    """
    t = jnp.asarray(temperature_k)
    h0, s0 = _cp_integrals(298.15, jnp.minimum(t, 727.), 106.76, .0108, -780000.)
    h1, s1 = _cp_integrals(727., jnp.maximum(t, 727.), 109.84, .0287, -8000000.)
    fe3p = (-164830+h0+h1-t*(110.15+s0+s1)
            +1000*(1-t/698)+_fe3p_magnetic_j(t))
    values = [fe3p]
    for hf, s, a, b, c, d in ((-157643., 83.12, 71.555, .02375, -370000., 0.),
                              (-126100., 47.77, 43.7878, .01985, -232000., 0.),
                              (-191100., 51.05, 77.52563, .009348, -443846., -1.1e-6)):
        dh, ds = _cp_integrals(298.15, t, a, b, c, d)
        values.append(hf+dh-t*(s+ds))
    limit = 3000. if high_temperature_continuation else 2000.
    return jnp.where((t >= 298.15) & (t <= limit), jnp.stack(values), jnp.nan)


def common_gibbs_j(temperature_k, pressure_pa, fe_liquid_standard_rt,
                   gas_p2_standard_rt, *, delta_volumes_m3_mol_formula,
                   high_temperature_continuation=False):
    """Translate all four solids to the caller's Fe-liquid/P2 gas gauge.

    Use the SAME native Fe liquid and one-bar P2 standards used to anchor
    the liquid MQM. This keeps each reaction G invariant under elemental
    gauge shifts. Caller-specified constant volume increments are separate
    per formula, referenced to one bar; no empirical values are supplied.
    """
    volumes = jnp.asarray(delta_volumes_m3_mol_formula)
    if volumes.shape != (4,):
        raise ValueError("Require four volume increments, in m3/mol formula.")
    t = temperature_k
    shifts = jnp.array([R*t*fe_liquid_standard_rt-reference.sgte_liquid_fe_j(t),
                       .5*(R*t*gas_p2_standard_rt-reference.nist_p2_gas_j(t))])
    return (standard_gibbs_j(t, high_temperature_continuation=high_temperature_continuation)
            +shifts @ jnp.asarray(ATOM_MATRIX)+(pressure_pa-1e5)*volumes)
