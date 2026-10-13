"""Mn-free rhombohedral Fe--Ti--Mg--Al oxide and pure rutile/corundum.

The Ghiorso--Evans model retains two Fe/Mg--Ti order parameters and the
fixed source SRO=0.0730205. Its corundum component standard is 20 kJ/mol
above the separate pure corundum phase; these references are not aliases.
"""

import jax.numpy as jnp

from exoeos.magma import R, _tp
from ._common import berman, expression, extensive, lambda_anomaly, pure_reference, relax
from ._coefficients import RHM_OXIDE, RHM_OXIDE_REFERENCES

COMPONENTS = ('MgTiO3', 'Fe2O3', 'FeTiO3', 'Al2O3')
_ENERGY, _SITES = expression(RHM_OXIDE)


def _pure_order_lower(T, P):
    # In s=2u-1, the curvature zeros obey
    # 6*B*z**2+(A-7*B)*z+R*T-A+B=0, z=s**2. The larger root
    # brackets the ordered minimum even when the disordered state is stable.
    dp = P/1e5-1
    a, b = 17477.+.010758*dp, 3189.+.035089*dp
    discriminant = jnp.maximum((a-7*b)**2-24*b*(R*T-a+b), 0.)
    z = jnp.clip((7*b-a+jnp.sqrt(discriminant))/(12*b), 0., 1.)
    return (1+jnp.sqrt(z))/2+1e-12


_REFERENCE = pure_reference(RHM_OXIDE_REFERENCES, ordered_lower=_pure_order_lower)
_CORUNDUM = (-1675700., 50.820, 2.558, 155.02, -828.4, -3.8614e6, 4.0908e8,
             -.385e-6, .375e-12, 21.342e-6, 47.180e-10)
_ROWS = (
    (-1572560., 74.56, 3.086, 146.20, -416., -3.9998e6, 4.0233e8,
     -.584e-6, 1.230e-12, 27.248e-6, 29.968e-10),
    (-825627., 87.437, 3.027, 146.86, 0., -5.5768e6, 5.2563e8,
     -.479e-6, .304e-12, 38.310e-6, 1.650e-10),
    (-1231947., 108.628, 3.170, 150., -441.6, -3.3237e6, 3.4815e8,
     -.584e-6, 1.230e-12, 27.248e-6, 29.968e-10),
    (_CORUNDUM[0]+20000., *_CORUNDUM[1:]),
)


def rutile_gibbs(T, P):
    """Pure TiO2 G (J/mol), with the pinned runtime H offset -9318.13 J/mol."""
    return berman(T, P, ((-944750.-9318.13, 50.460, 1.882, 77.84, 0., -3.3678e6, 4.0294e8,
                          -.454e-6, .584e-12, 25.716e-6, 15.409e-10),))[0]


def corundum_gibbs(T, P):
    """Separate pure Al2O3 G (J/mol), without the solution's 20 kJ offset."""
    return berman(T, P, (_CORUNDUM,))[0]


def standard_gibbs(T, P):
    """Return [Gk, Hm, Ilm, Cor] G0 in J/mol, including hematite lambda Cp."""
    T, P = _tp(T, P)
    return berman(T, P, _ROWS).at[1].add(lambda_anomaly(T, 955., 1287., -.07403, .00027921))


def _order(T, P, x):
    gk, _, il, cor = x

    def coordinates(q):
        return jnp.array([il*q[0], il*(1-q[0]), gk*q[1], gk*(1-q[1]), cor])

    free = jnp.array([il > 0, gk > 0])
    starts = jnp.where(free, jnp.array([[.5, .5], [.95, .95], [.95, .05]]), .5)
    q = relax(lambda q: _ENERGY(T, P, coordinates(q))/(R*T), jnp.full(2, .5), free,
              lambda q: _SITES(coordinates(q)), starts=starts)
    return coordinates(q)


def order(T, P, n):
    """Return [Fe2_A, Fe2_B, Mg_A, Mg_B, Al_A=Al_B] site fractions.

    Requires a present admissible phase; an unresolved root returns NaN.
    Symmetry-related A/B branches have identical relaxed thermodynamics.
    """
    T, P = _tp(T, P)
    n = jnp.asarray(n)*1.
    if n.shape != (4,):
        raise ValueError('n must have shape (4,) in COMPONENTS order.')
    return _order(T, P, n/n.sum())


def _mixing(T, P, x):
    return _ENERGY(T, P, _order(T, P, x))-x@_REFERENCE(T, P)


def mixing_gibbs(T, P, n):
    """Extensive relaxed mixing G in J; includes pure-reference subtraction."""
    return extensive(T, P, n, 4, _mixing)


def gibbs(T, P, n):
    """Extensive rhombohedral oxide G in J for nonnegative mol amounts.

    JIT/vmap and implicit first/second AD are supported at 1200--2200 K,
    1--5000 bar on smooth stable ordering branches. Derivatives at ordering
    transitions are not defined by this API. Composition AD requires a site
    interior; energies and fixed-face T/P AD include absent components.
    Caller validates finite positive T/P and the Mn-free composition.
    """
    return extensive(T, P, n, 4, lambda t, p, x: x@standard_gibbs(t, p)+_mixing(t, p, x))
