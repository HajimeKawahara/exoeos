"""Ca--Mg--Fe(II) clino/orthoamphiboles with relaxed Fe/Mg site ordering.

Formula amounts use Cum--Grn--Tr; signed Cum is needed for Fe-rich calcic
amphiboles. Both structures share the monoclinic Berman standards; the
orthorhombic endmember offsets belong to its mixing polynomial.
"""

import jax.numpy as jnp

from exoeos.magma import R, _tp
from ._common import berman, expression, extensive, relax, xlogx
from ._hydrous_coefficients import CLINOAMPHIBOLE, ORTHOAMPHIBOLE

COMPONENTS = ('Mg7Si8O22(OH)2', 'Fe7Si8O22(OH)2', 'Ca2Mg5Si8O22(OH)2')
_MODELS = {name: (*expression(data), expression({**data, 'sites': []})[0]) for name, data in
           (('clinoamphibole', CLINOAMPHIBOLE), ('orthoamphibole', ORTHOAMPHIBOLE))}
_ROWS = (
    (-12067920.38, 540.2587, 26.34, 1233.792, -7133.98, -221.638e5, 233.393749e7,
     -1.1393847e-6, 0., 28.1048234e-6, 62.894e-10),
    (-9623550., 725., 27.840, 1347.83, -9356.91, -202.2848e5, 303.919e7,
     -1.670299e-6, 8.68919e-12, 28.400e-6, 0.),
    (-12307863., 549.500, 27.312, 1229.36, -6401.9, -320.899e5, 420.881e7,
     -1.4763e-6, 8.9462e-12, 24.374e-6, 98.338e-10),
)


def standard_gibbs(T, P):
    """Return the three shared monoclinic standards in J/mol."""
    return berman(T, P, _ROWS)


def _model(phase):
    if phase not in _MODELS:
        raise ValueError("phase must be 'clinoamphibole' or 'orthoamphibole'.")
    return _MODELS[phase]


def _coordinates(x):
    # Parameterize the minority cation to preserve trace Mg near grunerite.
    # Compute M4 capacity and Mg inventory without subtracting nearly equal
    # unit fractions; the entropy below also retains both occupations.
    ca, capacity = x[2], x[0]+x[1]
    fe, mg = 7*x[1], 7*x[0]+5*x[2]
    iron_minor = fe <= mg
    total = jnp.where(iron_minor, fe, mg)
    lo = jnp.maximum(0., (total-5)/2)
    hi = jnp.minimum(capacity, total/2)

    def bounds(a):
        return jnp.maximum(0., (total-2*a-2)/3), jnp.minimum(1., (total-2*a)/3)

    def coordinates(q):
        a = lo+(hi-lo)*q[0]
        lower, upper = bounds(a)
        b = lower+(upper-lower)*q[1]
        c = (total-2*a-3*b)/2
        minor, major = jnp.array([a, b, c]), jnp.array([capacity-a, 1-b, 1-c])
        return jnp.where(iron_minor, minor, major), jnp.where(iron_minor, major, minor), ca

    lower, upper = bounds((lo+hi)/2)
    return coordinates, jnp.array([hi > lo, upper > lower])


def _site_energy(T, P, state, model):
    fe, mg, ca = state
    ideal = jnp.array([2., 3., 2.])@(xlogx(fe)+xlogx(mg))+2*xlogx(ca)
    return model[2](T, P, jnp.r_[fe, ca])+R*T*ideal


def _order(T, P, x, model):
    coordinates, free = _coordinates(x)
    q = relax(lambda q: _site_energy(T, P, coordinates(q), model)/(R*T), jnp.full(2, .5), free,
              lambda q: jnp.concatenate(coordinates(q)[:2]))
    return coordinates(q)


def order(T, P, n, *, phase='clinoamphibole'):
    """Return [Fe_M4, Fe_M13, Fe_M2, Ca_M4] site fractions of a present phase.

    Multiplicities are 2, 3 and 2. Stationarity and positive ordering
    curvature are required; failure returns NaN, not a clipped root.
    """
    T, P = _tp(T, P)
    n = jnp.asarray(n)*1.
    if n.shape != (3,):
        raise ValueError('n must have shape (3,) in COMPONENTS order.')
    fe, _, ca = _order(T, P, n/n.sum(), _model(phase))
    return jnp.r_[fe, ca]


def mixing_gibbs(T, P, n, *, phase='clinoamphibole'):
    """Extensive mixing G (J), relaxed within the specified structure."""
    model = _model(phase)
    return extensive(T, P, n, 3, lambda t, p, x: _site_energy(t, p, _order(t, p, x, model), model))


def gibbs(T, P, n, *, phase='clinoamphibole'):
    """Extensive relaxed G (J); scalar K/Pa and formula amounts in mol.

    For x=n/sum(n), require 0<=x[2]<=1 and 0<=7*x[1]<=7-2*x[2].
    Signed Cum is allowed. Caller validates finite positive T/P and the site
    domain. Enable x64. First/second AD is implicit on smooth stable order
    branches; continuous G and fixed-face T/P AD are supported at edges.
    Zero phase G=0; its AD sentinel is not a chemical potential. No structure
    selection or global phase-stability claim is made. Numerical validation:
    800--2200 K, 1--5000 bar, not a new experimental calibration range.
    """
    model = _model(phase)
    return extensive(T, P, n, 3, lambda t, p, x: x@standard_gibbs(t, p)
                      +_site_energy(t, p, _order(t, p, x, model), model))
