"""MELTS pargasite--ferropargasite--magnesiohastingsite hornblende.

The independent Fe(II)/Mg and Fe(III)/Al substitutions span a rectangle:
formal pargasite amounts may be negative. This is not a full amphibole model.
"""

from exoeos.magma import R
from ._common import berman, extensive, xlogx

COMPONENTS = ('NaCa2Mg4Al3Si6O22(OH)2', 'NaCa2Fe4Al3Si6O22(OH)2',
              'NaCa2Mg4FeAl2Si6O22(OH)2')
_ROWS = (
    (-3016624.*4.184, 160.*4.184, 27.35, 1267.25, -6654.34, -303.787e5, 391.353e7,
     -1.392e-6, 3.481e-12, 24.374e-6, 98.338e-10),
    (-3016624.*4.184+4*(-9623300.+12067517.)/7, 185.5*4.184, 27.989,
     1342.61, -8348.62, -247.604e5, 348.507e7, -1.392e-6, 3.481e-12, 24.374e-6, 98.338e-10),
    (-3016624.*4.184+(-2836709.+7932.05)-(-3275265.-8565.18), 163.8*4.184, 27.38,
     1273.66, -6716.06, -280.331e5, 350.697e7, -1.392e-6, 3.481e-12, 24.374e-6, 98.338e-10),
)


def standard_gibbs(T, P):
    """Return [pargasite, ferropargasite, magnesiohastingsite] G0 (J/mol)."""
    return berman(T, P, _ROWS)


def _mixing(T, P, x):
    fe2, fe3 = x[1:]
    mg, al = x[0]+x[2], x[0]+x[1]
    return (28116.48*fe2*mg+16780.*fe3*al
            +R*T*(4*(xlogx(fe2)+xlogx(mg))+xlogx(fe3)+xlogx(al)))


def mixing_gibbs(T, P, n):
    """Extensive site mixing G (J); no additional internal order variable."""
    return extensive(T, P, n, 3, _mixing)


def gibbs(T, P, n):
    """Extensive G (J), with scalar T in K, P in Pa, formula amounts in mol.

    For N=sum(n)>0, require 0<=n[1]/N<=1 and 0<=n[2]/N<=1; n[0] may
    be negative. Caller validates this site domain and finite positive T/P.
    First/second AD applies in the open site rectangle; continuous G and
    fixed-face T/P derivatives apply on edges. The zero-phase AD sentinel
    is not a chemical potential. Enable x64. Verified at 800--2200 K,
    1--5000 bar; these are numerical, not experimental stability bounds.
    """
    return extensive(T, P, n, 3, lambda t, p, x: x@standard_gibbs(t, p)+_mixing(t, p, x))
