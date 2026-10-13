"""MELTS annite--phlogopite biotite; K, Pa, formula mol and J.

The pinned biotite.c uses one binary entropy, not three independent Fe/Mg
site entropies. This transfers that model without recalibration.
"""

from exoeos.magma import R
from ._common import berman, extensive, xlogx

COMPONENTS = ('KFe3Si3AlO10(OH)2', 'KMg3Si3AlO10(OH)2')
_ROWS = (
    (-5142800., 420., 15.408, 727.208, -4775.040, -138.319e5, 211.906e7,
     -1.6969784e-6, 0., 34.4473262e-6, 0.),
    (-6210391., 334.346, 14.977, 610.37988, -2083.781, -215.33008e5, 284.1040896e7,
     -1.6969784e-6, 0., 34.4473262e-6, 0.),
)


def standard_gibbs(T, P):
    """Return [annite, phlogopite] Berman standards in J/mol."""
    return berman(T, P, _ROWS)


def _mixing(T, P, x):
    return 13430.64*x[0]*x[1]+R*T*xlogx(x).sum()


def mixing_gibbs(T, P, n):
    """Extensive mixing G (J), with continuous zero-component limits."""
    return extensive(T, P, n, 2, _mixing)


def gibbs(T, P, n):
    """Extensive G (J); scalar K/Pa and nonnegative formula amounts (mol).

    Enable x64. Caller validates finite positive T/P and the composition.
    First/second AD requires positive amounts; fixed-face T/P AD is valid.
    An absent phase has G=0, but its AD sentinel is not a chemical potential.
    Numerical comparisons cover 800--2200 K and 1--5000 bar, not a new
    experimental calibration or stability range. No internal order solve.
    """
    return extensive(T, P, n, 2, lambda t, p, x: x@standard_gibbs(t, p)+_mixing(t, p, x))
