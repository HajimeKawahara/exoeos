"""Native JAX rhyolite-MELTS 1.2.0 (mode 4) carbon-bearing liquid.

This separate model appends independent CO2 to the twelve-component hydrous
basis and relaxes CaSiO3 + CO2 = SiO2 + CaCO3 inside the liquid. It changes
the water standard and interactions even at zero carbon. No S/N, redox,
carbonate solids, phase selection or fluid saturation is included.

MAGMA 705a0fb315e5054d18275a580562f6121c8e458c: liquid_CO2_H2O.c,
param_struct_data_CO2_H2O.h, gibbs.c and fluidPhase.c. Scalar K/Pa/mol
inputs give extensive J. Enable x64; caller validates finite T >= 700 K,
P > 0 and nonnegative amounts. Derivatives require the documented domain.
"""

import jax
import jax.numpy as jnp

from exoeos import magma
from exoeos._magma_carbon import _co2_liquid_gibbs, _water_mode4_gibbs


COMPONENTS = magma.HYDROUS_COMPONENTS + ("CO2",)
OXIDES = magma.HYDROUS_OXIDES + ("CO2",)
SPECIES = COMPONENTS + ("CaCO3",)
# Upper triangle; reuse exactly the unchanged 55 dry pairs (J/mol).
_WATER_W = (27357., 20909.9, 30936.5, 9715., 11768.4, 50105.2,
            88199.2, 0., -82460.1, 1057.2, 44133.3)
_CO2_W = (0., -40853.6, -32464.5, 30012.5, 0., -3186.9,
          -19265.7, 0., 0., 0., -3472.8, 23255.4)
_CARBONATE_W = (63281.2, -24872.6, -72996.8, 37534.1, 46716.1, 65508.7,
                -79202.7, 0., -311011.3, -27865., 2011.9, 7873., 0.)
_REACTION = (1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., -1., 1.)


def _interaction_matrix():
    w = jnp.zeros((14, 14)).at[:11, :11].set(jnp.asarray(magma._LIQUID_W))
    w = w.at[:11, 11].set(jnp.asarray(_WATER_W))
    w = w.at[:12, 12].set(jnp.asarray(_CO2_W))
    w = w.at[:13, 13].set(jnp.asarray(_CARBONATE_W))
    return w+w.T


def _amounts(n):
    n = jnp.asarray(n, dtype=float)
    if n.shape != (13,):
        raise ValueError("n must have shape (13,) in magma_carbonate.COMPONENTS order.")
    return n


def liquid_moles_from_oxides(m):
    """Map OXIDES mol to COMPONENTS mol; preserve signed host residuals."""
    m = _amounts(m)
    return jnp.append(magma.liquid_moles_from_oxides(m[:12]), m[12])


def oxide_moles_from_liquid(n):
    """Inverse atom-conserving conversion; CO2 passes through unchanged."""
    n = _amounts(n)
    return jnp.append(magma.oxide_moles_from_liquid(n[:12]), n[12])


def liquid_standard_gibbs(T, P):
    """Return 13 independent-component standards, J/mol, at scalar K/Pa.

    The internal CaCO3 standard is g_CaSiO3 + g_CO2 - g_SiO2 + delta,
    with delta = -17574.497522747 - 1.9034060173857*(P/1e5-1).
    These are species standards, not the chemical potentials of a mixture.
    The dry SiO2 Cp branch at 1480 K and water's T >= 700 K domain apply.
    """
    T, P = magma._tp(T, P)
    return jnp.concatenate((magma.liquid_standard_gibbs(T, P, include_dry11=True),
                            jnp.stack((_water_mode4_gibbs(T, P), _co2_liquid_gibbs(T, P)))))


def _reaction_gibbs(P):
    return -17574.497522747-1.9034060173857*(P/1e5-1)


def _species(n, extent):
    return jnp.append(n, 0.)+extent*jnp.asarray(_REACTION)


def _xlogx(x):
    # Continuous energy and fixed-face T/P limits; no composition floor.
    return x*jnp.log(jnp.where(x > 0, x, 1.))


def _mixing_gibbs(T, P, x, extent, w):
    y = _species(x, extent)
    dry = jnp.sum(y[:11])+y[12]+y[13]
    ideal = jnp.sum(_xlogx(y))+_xlogx(y[11])+_xlogx(dry)
    return extent*_reaction_gibbs(P)+magma.R*T*ideal+y@w@y/2


def _extent(T, P, x, w):
    upper = jnp.minimum(x[3], x[12])

    def active(_):
        reaction = jnp.asarray(_REACTION)

        def residual(q):
            y = _species(x, upper*q)
            return ((_reaction_gibbs(P)+reaction@w@y)/(magma.R*T)
                    + jnp.log(y[0])+jnp.log(y[13])-jnp.log(y[3])-jnp.log(y[12]))

        def solve(fun, guess):
            def step(_, bracket):
                lo, hi = bracket
                mid = (lo+hi)/2
                return jax.lax.cond(fun(mid) < 0, lambda: (mid, hi), lambda: (lo, mid))
            lo, hi = jax.lax.fori_loop(0, 80, step, (jnp.zeros_like(guess), jnp.ones_like(guess)))
            q = (lo+hi)/2
            # Reject an unrepresentable root instead of flooring species.
            return jnp.where(jnp.abs(fun(q)) < 1e-8, q, jnp.nan)

        # Positive reaction curvature (113246.6 J/mol plus ideal curvature)
        # makes the root unique. AD differentiates stationarity, not bisection.
        q = jax.lax.custom_root(residual, jnp.array(.5), solve,
                               lambda linear, rhs: rhs/linear(jnp.ones_like(rhs)))
        return upper*q

    return jax.lax.cond(upper > 0, active, lambda _: jnp.zeros_like(upper), None)


def liquid_species_moles(T, P, n):
    """Return 14 relaxed species amounts in SPECIES order, in mol.

    CaCO3 is an internal species, not an independent reservoir. Both the
    total component amount and every element are conserved. Zero carbon
    or zero CaSiO3 forces zero reaction extent; an absent phase returns zeros.
    """
    T, P = magma._tp(T, P)
    n = _amounts(n)

    def present(n):
        return _species(n, n.sum()*_extent(T, P, n/n.sum(), _interaction_matrix()))

    return jax.lax.cond(jnp.all(n == 0), lambda n: jnp.zeros(14, dtype=n.dtype), present, n)


def liquid_gibbs(T, P, n):
    """Extensive mode-4 liquid G in J; 13 supplied independent mol amounts.

    Internal reaction relaxation is included in first and second JAX
    derivatives. Composition derivatives require positive independent
    amounts and a representable interior reaction root. At zero carbon,
    zero calcium or other component faces, use energy and fixed-composition
    T/P derivatives only. An absent phase has G=0 and no chemical potentials.
    No mode-1 equality is implied on the zero-carbon hydrous face.
    """
    T, P = magma._tp(T, P)
    n = _amounts(n)

    def present(n):
        x, w = n/n.sum(), _interaction_matrix()
        s = _extent(T, P, x, w)
        return n@liquid_standard_gibbs(T, P)+n.sum()*_mixing_gibbs(T, P, x, s, w)

    return jax.lax.cond(jnp.all(n == 0), lambda n: jnp.zeros_like(T+P+n.sum()), present, n)


def hydrous_liquid_gibbs(T, P, n):
    """Return mode-4 G [J] restricted to the carbon-free hydrous face.

    Supply twelve mol amounts in ``magma.HYDROUS_COMPONENTS`` order.
    Carbon and carbonate are fixed at zero, so the internal reaction has
    identically zero extent. First and second composition derivatives are
    supported for positive supplied amounts, tangent to this face only;
    they do not describe carbon insertion. T >= 700 K and P > 0 apply.
    An absent phase has G=0 and no chemical potentials.
    """
    T, P = magma._tp(T, P)
    n = jnp.asarray(n, dtype=float)
    if n.shape != (12,):
        raise ValueError("n must have shape (12,) in magma.HYDROUS_COMPONENTS order.")

    def present(n):
        full = jnp.append(n, 0.)
        return (n @ liquid_standard_gibbs(T, P)[:12]
                + n.sum()*_mixing_gibbs(T, P, full/n.sum(), 0., _interaction_matrix()))

    return jax.lax.cond(jnp.all(n == 0), lambda n: jnp.zeros_like(T+P+n.sum()), present, n)
