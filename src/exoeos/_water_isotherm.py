"""Integrate a supplied two-power water isotherm; no empirical coefficients."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.nn import softmax
import numpy as np

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(64)
_U, _Q = (_NODES+1)/2, _WEIGHTS/2


@jax.custom_jvp
def _log_pressure(log_w, log_c, alpha):
    """Invert log(w)=logsumexp(log(C)+alpha*log(p)) monotonically."""
    upper = jnp.min((log_w-log_c)/alpha)
    lower = upper-jnp.log(2.)/jnp.min(alpha)

    def bisect(_, bounds):
        lo, hi = bounds
        mid = (lo+hi)/2
        above = logsumexp(log_c+alpha*mid) > log_w
        return jnp.where(above, lo, mid), jnp.where(above, mid, hi)

    lo, hi = jax.lax.fori_loop(0, 64, bisect, (lower, upper))
    return (lo+hi)/2


@_log_pressure.defjvp
def _log_pressure_jvp(primals, tangents):
    log_w, log_c, alpha = primals
    dw, dc, da = tangents
    z = _log_pressure(log_w, log_c, alpha)
    shares = softmax(log_c+alpha*z)
    return z, (dw-jnp.sum(shares*(dc+z*da)))/jnp.sum(shares*alpha)


def two_power_water_g_rt(water, equivalent, capacities, exponents):
    """Return the extensive mixing contribution, excluding the gas standard.

    w=h/(S+h)=sum(C_i*p**a_i), p=water driving pressure / 1 bar, S=dry mass
    divided by water molar mass. The scalar is h*ln(p)-S*integral[w/(1-w)
    dln(p)] from zero pressure. A transformed 64-point Gauss rule resolves
    the zero-pressure endpoint. Implicit root derivatives include C and a.

    The numerical domain is C>0, 0.25<=a<=1, S>0 and 0<=w<=0.5.
    These limits contain the Sun--Yao exponents and the whole dry11 host
    oxygen constraint, but are not an empirical calibration claim. Interior
    derivatives are supported; the zero-water chemical potential is singular.
    """
    c, a = jnp.asarray(capacities), jnp.asarray(exponents)
    if c.shape != (2,) or a.shape != (2,):
        raise ValueError("Require two water capacities and two exponents.")
    w = water/(equivalent+water)
    # Evaluate a harmless interior dummy only on the exact zero-water face.
    log_w = jnp.log(jnp.where(w > 0, w, .1))
    log_c = jnp.log(c)
    z = _log_pressure(log_w, log_c, a)
    terms = jnp.exp(log_c+a*z)
    scale = 8/jnp.min(a)
    u = jnp.asarray(_U)
    at_u = jnp.sum(terms*u[:, None]**(scale*a), axis=1)
    integral = scale*jnp.sum(jnp.asarray(_Q)*at_u/(u*(1-at_u)))
    value = jnp.where(water == 0, 0., water*z-equivalent*integral)
    valid = (jnp.all(jnp.isfinite(c) & (c > 0)) & jnp.all(jnp.isfinite(a) & (a >= .25) & (a <= 1))
             & jnp.isfinite(equivalent) & (equivalent > 0) & (water >= 0) & (w <= .5))
    return jnp.where(valid, value, jnp.nan)
