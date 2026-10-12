"""Sack--Ghiorso Fe--Mg--Al--Cr--Fe3--Ti spinel, with three site orders."""

import jax.numpy as jnp

from exoeos.magma import R, _tp
from ._common import berman, expression, extensive, lambda_anomaly, pure_reference, relax
from ._coefficients import SPINEL, SPINEL_REFERENCES

COMPONENTS = ('FeCr2O4', 'FeAl2O4', 'Fe3O4', 'MgAl2O4', 'Fe2TiO4')
_ENERGY, _SITES = expression(SPINEL)
_REFERENCE = pure_reference(SPINEL_REFERENCES)
_ROWS = (
    (-1445490., 142.676, 4.4010, 236.874, -1679.6, 0., -1.6765e8, 0., 0., 0., 0.),
    (-1947681., 115.362, .973948*4.184, 235.190, -1437., -4.6913e6, 6.4564e8, 0., 0., 0., 0.),
    (-1117403., 146.114, 4.452, 207.93, 0., -7.2433e6, 6.6436e8,
     -.582e-6, 1.751e-12, 30.291e-6, 138.470e-10),
    (-2300313., 84.535, 3.977, 235.90, -1766.6, -1.7104e6, 4.062e7,
     -.489e-6, 0., 21.691e-6, 50.528e-10),
    (-1488500., 185.447, 4.682, 249.63, -1817.4, 0., -5.453e7, 0., 0., 0., 0.),
)


def standard_gibbs(T, P):
    """Return [Chr, Hc, Mt, Sp, Usp] G0 in J/mol, including magnetite lambda Cp."""
    T, P = _tp(T, P)
    return berman(T, P, _ROWS).at[2].add(lambda_anomaly(T, 848., 1565., -.19502, .00061037))


def _coordinates(x):
    cr, hc, mt, sp, uv = x
    totals = jnp.array([sp, 2*(hc+sp), 2*mt])
    fe2 = 1+uv-sp
    mobile = 3-2*cr-uv
    initial = jnp.full(3, 1/mobile)
    # On a Fe2-free face the most abundant cation closes the tetrahedral sum.
    pivot = jnp.argmax(totals)
    dependent = (jnp.arange(3) == pivot) & (fe2 == 0)
    free = (totals > 0) & ~dependent

    def coordinates(q):
        tetra = totals*q
        remainder = 1-jnp.sum(jnp.where(dependent, 0., tetra))
        tetra = jnp.where(dependent, remainder, tetra)
        octa = (totals-tetra)/2
        return jnp.concatenate((tetra, octa, jnp.array([cr])))

    return coordinates, initial, free


def _order(T, P, x):
    coordinates, initial, free = _coordinates(x)
    q = relax(lambda q: _ENERGY(T, P, coordinates(q))/(R*T), initial, free,
              lambda q: _SITES(coordinates(q)))
    return coordinates(q)


def order(T, P, n):
    """Return [Mg_T, Al_T, Fe3_T, Mg_O, Al_O, Fe3_O, Cr_O] site fractions.

    Tetrahedral multiplicity is one and octahedral multiplicity is two.
    Requires a present admissible phase; an unresolved root returns NaN.
    """
    T, P = _tp(T, P)
    n = jnp.asarray(n)*1.
    if n.shape != (5,):
        raise ValueError('n must have shape (5,) in COMPONENTS order.')
    return _order(T, P, n/n.sum())


def _mixing(T, P, x):
    return _ENERGY(T, P, _order(T, P, x))-x@_REFERENCE(T, P)


def mixing_gibbs(T, P, n):
    """Extensive mixing G in J, with relaxed pure-reference subtraction."""
    return extensive(T, P, n, 5, _mixing)


def gibbs(T, P, n):
    """Extensive relaxed G in J at scalar K/Pa, with amounts in mol.

    Nonnegative endmember amounts are admissible; signed amounts may also
    describe physical sites (see documentation). Caller validates values.
    At 1200--2200 K and 1--5000 bar, JIT/vmap and implicit first/second
    derivatives are supported on stable order branches. Composition AD
    requires a site interior; at edges use energies and fixed-face T/P AD.
    No redox equilibrium or global phase selection is performed.
    """
    return extensive(T, P, n, 5, lambda t, p, x: x@standard_gibbs(t, p)+_mixing(t, p, x))
