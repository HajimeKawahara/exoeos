"""Pinned MELTS CO2 gas/1-bar standards and mode-4 dissolved water.

MAGMA 705a0fb315e5054d18275a580562f6121c8e458c, fluidPhase.c and gibbs.c.
Only the Duan--Zhang (2006) pure-CO2 1-bar root is needed here. The mixed
fluid API uses a separately selected EOS. All legacy constants are retained.
"""

import jax
import jax.numpy as jnp

from exoeos._magma_water import _haar_density, _haar_helmholtz


def _co2_ideal_gibbs(T):
    """Analytic idealGasCO2 H-T*S at 1 bar, J/mol."""
    c = jnp.array([-1.8188731, 12.903022, -9.6634864, 4.2251879,
                   -1.0421640, .12683515, -.0049939675, 2.4950242,
                   -.82723750, .15372481, -.015861243, .00086017150,
                   -.000019222165])
    t = T/1000
    k, ell = jnp.arange(1., 7.), jnp.arange(1., 6.)
    h = (jnp.sum(c[:7]*t**jnp.arange(1., 8.)/jnp.arange(1., 8.))
         + c[7]*jnp.log(t) - jnp.sum(c[8:]*t**(-ell)/ell))
    s = (c[0]*jnp.log(t) + jnp.sum(c[1:7]*t**k/k)
         - jnp.sum(c[7:]*t**(-k)/k))
    return 8.31451*(1000*h-T*s)-385358.2260-T*210.0304


def _co2_reference_gibbs(T):
    """Real pure CO2 at 1 bar from the pinned low-pressure Duan EOS."""
    tr = T/304.1282
    a = jnp.array([[.114400435, -.938526684, .721857006],
                   [.00881072902, .0636473911, -.0770822213],
                   [.000901506064, -.00681834166, .00732364258],
                   [-.000110288237, .00126524193, -.00149730823]])
    b, c, d, e = a[:, 0]+a[:, 1]/tr**2+a[:, 2]/tr**3
    f, beta, gamma = .00781940730/tr**3, -4.22918013, .1585
    vc = 8.314467*304.1282/73.773  # J/bar/mol

    def zm1(q):
        u = gamma*q*q
        return b*q+c*q*q+d*q**4+e*q**5+f*q*q*(beta+u)*jnp.exp(-u)

    def residual(q):
        return q*(1+zm1(q))*8.314467*T/vc-1

    def solve(fun, guess):
        return jax.lax.fori_loop(0, 8, lambda _, q: q-fun(q)/jax.grad(fun)(q), guess)

    q = jax.lax.custom_root(residual, vc/(8.314467*T), solve,
                            lambda linear, rhs: rhs/linear(jnp.ones_like(rhs)))
    u = gamma*q*q
    ar = (b*q+c*q*q/2+d*q**4/4+e*q**5/5
          + f/(2*gamma)*(-(beta+1)*jnp.expm1(-u)-u*jnp.exp(-u)))
    lnphi = ar+zm1(q)-jnp.log1p(zm1(q))
    return _co2_ideal_gibbs(T)+8.3143*T*lnphi


def _water_mode4_gibbs(T, P):
    """Mode-4 H2O standard; Ochs--Lange pressure term, J/mol."""
    rho = _haar_density(T)
    haar = (1.80152*(_haar_helmholtz(T, rho)+1/rho)
            - 285829.96-298.15*69.9146+54955.23970014)
    robie = (2.9147*jnp.log(T)-9.6863e-4*T+6.8593e-8*T*T
             + 77.8899/jnp.sqrt(T)-28954.8/T-2263.27/T**2-15.8997)
    dp = P/1e5-1
    return (haar+8.3143*T*(-33676/T+18.3527-robie)
            + 2783.6851512128-T*2.3838467967178
            + (2.775+1.086e-3*(T-1673.15))*dp-.382e-4*dp*dp/2)


def _co2_liquid_gibbs(T, P):
    dp = P/1e5-1
    return (_co2_reference_gibbs(T)-630.93193811701+T*109.39331414050
            + (4.0157994267547+1.213189e-3*(T-1673.0))*dp
            - .4267387e-4*dp*dp/2)
