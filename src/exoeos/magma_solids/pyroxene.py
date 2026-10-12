"""Seven-endmember Sack--Ghiorso clino/orthopyroxenes with two site orders.

Coordinates are mol per six-oxygen formula, NOT magma.PYROXENE_COMPONENTS.
Signed endmember amounts are allowed when every physical site is admissible.
Fe in buffonite and essenite is ferric; hedenbergite carries ferrous Fe.
Both structures subtract the monoclinic pure-order reference curves.
"""

import jax.numpy as jnp

from exoeos.magma import R, _tp
from ._common import berman, expression, extensive, pure_reference, relax
from ._coefficients import CLINOPYROXENE, ORTHOPYROXENE, CLINOPYROXENE_REFERENCES

COMPONENTS = ('CaMgSi2O6', 'Mg2Si2O6', 'CaFeSi2O6', 'CaTi0.5Mg0.5AlSiO6',
              'CaTi0.5Mg0.5FeSiO6', 'CaFeAlSiO6', 'NaAlSi2O6')
_MODELS = {name: expression(data) for name, data in
           (('clinopyroxene', CLINOPYROXENE), ('orthopyroxene', ORTHOPYROXENE))}
_REFERENCE = pure_reference(CLINOPYROXENE_REFERENCES)
_ROWS = (
    (-3200583., 142.5, 6.620, 305.41, -1604.9, -7.166e6, 9.2184e8,
     -.872e-6, 1.707e-12, 27.795e-6, 83.082e-10),
    (-3086083., 135.164, 6.3279, 333.16, -2401.2, -4.5412e6, 5.583e8,
     -.749e-6, .447e-12, 24.656e-6, 74.670e-10),
    (-2842221., 174.2, 6.7894, 307.89, -1597.3, -6.9925e6, 9.3522e8,
     -.9925e-6, 1.4835e-12, 31.371e-6, 83.672e-10),
    (-3275265.-8565.18, 143.745, 6.356, 297.499, -1355.96, -6.70219e6, 7.59082e8,
     -.870e-6, 2.171e-12, 22.250e-6, 52.863e-10),
    (-2836709.+7932.05, 161.985+18.756, 6.7217, 303.909, -1417.67, -4.35654e6, 3.52523e8,
     -.870e-6, 2.171e-12, 22.250e-6, 52.863e-10),
    (-2860211.-7267.01, 158.991+18.756, 6.7217, 317.11, -1733.3, -5.10965e6, 5.4222e8,
     -.870e-6, 2.171e-12, 22.250e-6, 52.863e-10),
    (-3025118., 133.574, 6.034, 311.29, -2005.1, -5.3503e6, 6.6257e8,
     -.860e-6, 2.149e-12, 23.118e-6, 25.785e-10),
)


def standard_gibbs(T, P):
    """Return seven monoclinic G0 standards (J/mol), shared by both phases."""
    return berman(T, P, _ROWS)


def _model(phase):
    if phase not in _MODELS:
        raise ValueError("phase must be 'clinopyroxene' or 'orthopyroxene'.")
    return _MODELS[phase]


def _coordinates(x):
    _, en, hd, ca, cf, es, jd = x
    lower = jnp.array([jnp.maximum(0., -ca), jnp.maximum(0., hd+es+jd+(ca+cf)/2-1)])
    upper = jnp.array([jnp.minimum(es+jd, cf+es), jnp.minimum(hd, en)])

    def coordinates(q):
        ferric, ferrous = lower+(upper-lower)*q
        return jnp.array([hd-ferrous, es+jd-ferric, ferric, (ca+cf)/2,
                          ferrous, en-ferrous, jd, (ca+ferric)/2])

    return coordinates, upper > lower


def _order(T, P, x, model):
    energy, sites = model
    coordinates, free = _coordinates(x)
    q = relax(lambda q: energy(T, P, coordinates(q))/(R*T), jnp.full(2, .5), free,
              lambda q: sites(coordinates(q)))
    return coordinates(q)


def order(T, P, n, *, phase='clinopyroxene'):
    """Return [Fe2_M1, Al_M1, Fe3_M1, Ti_M1, Fe2_M2, Mg_M2, Na_M2, Al_T].

    Site fractions are dimensionless. A present admissible phase is required;
    failed stationary solves return NaN. Si and other complementary sites
    follow the source site balance. T/P must be scalar K/Pa.
    """
    T, P = _tp(T, P)
    n = jnp.asarray(n)*1.
    if n.shape != (7,):
        raise ValueError('n must have shape (7,) in COMPONENTS order.')
    return _order(T, P, n/n.sum(), _model(phase))


def _mixing(T, P, x, model):
    q = _order(T, P, x, model)
    return model[0](T, P, q)-x@_REFERENCE(T, P)


def mixing_gibbs(T, P, n, *, phase='clinopyroxene'):
    """Extensive relaxed mixing G in J, including pure-reference subtraction."""
    model = _model(phase)
    return extensive(T, P, n, 7, lambda t, p, x: _mixing(t, p, x, model))


def gibbs(T, P, n, *, phase='clinopyroxene'):
    """Extensive relaxed G in J for the physical seven-component site domain.

    Caller validates finite positive T/P, positive total amount, and the site
    domain (see documentation). Zero phase G is exactly zero. JIT/vmap and
    implicit first/second AD cover 1200--2200 K, 1--5000 bar, stable order
    branches. Composition AD requires a site interior; fixed-face T/P AD and
    continuous energies are supported at edges. This does not select phases.
    """
    model = _model(phase)
    return extensive(T, P, n, 7, lambda t, p, x: x@standard_gibbs(t, p)+_mixing(t, p, x, model))
