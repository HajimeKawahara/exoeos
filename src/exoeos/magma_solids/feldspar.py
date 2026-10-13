"""Na--Ca--K feldspars: Ab--An--Or, with ordered pure feldspar standards.

Amounts are mol of NaAlSi3O8, CaAl2Si2O8, KAlSi3O8. Choose plagioclase
or alkali-feldspar explicitly: their ternary mixing polynomials differ.
The plagioclase polynomial is pinned to alphaMELTS 2.3.2 (see provenance).
The solution has no additional order variable; albite's two coupled order
variables and sanidine's integrated disorder correction belong to G0.
"""

import jax.numpy as jnp

from exoeos.magma import R, _tp, anorthite_gibbs
from ._common import berman, expression, extensive, relax
from ._coefficients import ALKALI_FELDSPAR, PLAGIOCLASE

COMPONENTS = ('NaAlSi3O8', 'CaAl2Si2O8', 'KAlSi3O8')
_MIXING = {name: expression(data)[0] for name, data in
           (('plagioclase', PLAGIOCLASE), ('alkali-feldspar', ALKALI_FELDSPAR))}
_ROWS = (
    (-3921618., 224.412, 10.083, 393.64, -2415.5, -7.8928e6, 1.07064e9,
     -1.945e-6, 4.861e-12, 26.307e-6, 32.407e-10),
    (-3970791.+3400., 214.145, 10.869, 381.37, -1941., -12.0373e6, 1.83643e9,
     -1.805e-6, 5.112e-12, 15.145e-6, 54.850e-10),
)


def _albite_energy(T, P, q):
    a, b = q
    h = (-5.479*1251*a*a/2+6854*a**4/4-41.620*824.1*b*b/2
         -9301*b**4/4+43600*b**6/6+(-2.171+.001569*T*T-2*.000002109*T**3)*a*b)
    s = -(5.479*a*a/2+41.620*b*b/2+(-3.043-2*.001569*T+3*.000002109*T*T)*a*b)
    return h*(1+(P/1e5-1)/335282.925)-T*s


def albite_order(T, P):
    """Return the two relaxed Salje order parameters (dimensionless)."""
    T, P = _tp(T, P)
    return relax(lambda q: _albite_energy(T, P, q)/(R*T), jnp.array([.6, .9]),
                 jnp.ones(2, dtype=bool), lambda q: jnp.ones(1),
                 starts=jnp.array([[.6, .9], [0., 0.]]))


def _sanidine_disorder(T, P):
    tr, u = 298.15, jnp.clip(T, 298.15, 1436.)
    d0, d1, d2, d3, d4 = 282.98, -4830., 3.621e6, -.15733, 34.770e-6
    h = (d0*(u-tr)+2*d1*(jnp.sqrt(u)-jnp.sqrt(tr))-d2*(1/u-1/tr)
         +d3*(u*u-tr*tr)/2+d4*(u**3-tr**3)/3)
    s = (d0*jnp.log(u/tr)-2*d1*(u**-.5-tr**-.5)-d2*(u**-2-tr**-2)/2
         +d3*(u-tr)+d4*(u*u-tr*tr)/2)
    return h-T*s+(P/1e5-1)*h/410630.


def standard_gibbs(T, P, *, phase='plagioclase'):
    """Return [Ab, An, Or] G0 in J/mol, including phase-specific references.

    The pinned runtime adds 44905.5-15.012*T J/mol to plagioclase Or,
    and 19487 J/mol to alkali-feldspar An. These documented runtime offsets
    are separate from the published Berman/ordering terms and mixing G.
    """
    _model(phase)
    T, P = _tp(T, P)
    ab, k = berman(T, P, _ROWS)
    reference = jnp.array([0., 0., 44905.5-15.012*T]) if phase == 'plagioclase' else jnp.array([0., 19487., 0.])
    return jnp.array([ab+_albite_energy(T, P, albite_order(T, P)),
                      anorthite_gibbs(T, P), k+_sanidine_disorder(T, P)])+reference


def _model(phase):
    if phase not in _MIXING:
        raise ValueError("phase must be 'plagioclase' or 'alkali-feldspar'.")
    return _MIXING[phase]


def mixing_gibbs(T, P, n, *, phase='plagioclase'):
    """Extensive mixing G in J; nonnegative n, continuous energies at edges."""
    model = _model(phase)
    return extensive(T, P, n, 3, lambda t, p, x: model(t, p, x[:2]))


def gibbs(T, P, n, *, phase='plagioclase'):
    """Extensive relaxed G in J; scalar K/Pa, nonnegative amounts in mol.

    JIT/vmap and first/second AD are supported at 1200--2200 K, 1--5000
    bar away from order transitions and the sanidine cutoff (1436 K).
    Composition AD requires all amounts positive; face T/P AD is supported.
    The caller validates finite positive T/P and nonnegative amounts.
    """
    model = _model(phase)
    return extensive(T, P, n, 3, lambda t, p, x: x@standard_gibbs(t, p, phase=phase)+model(t, p, x[:2]))
