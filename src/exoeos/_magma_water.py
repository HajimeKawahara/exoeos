"""MELTS dissolved-H2O standard; private high-temperature Haar reference.

MAGMA 705a0fb315e5054d18275a580562f6121c8e458c, water.c and gibbs.c.
Only the 1 bar vapor root is needed (T >= 700 K); this is not a fluid EOS
API. Keep the original specific-energy units, constants and reference shift.
"""

import jax
import jax.numpy as jnp


def _haar_helmholtz(T, rho):
    # rho: g/cm3; A: bar cm3/g. All 40 residual terms from water.c.
    tau = T / 647.073
    b = .7478629 - .3540782*jnp.log(tau) + .007159876/tau**3 - .003528426/tau**5
    bb = 1.1278334 - .5944001/tau - 5.010996/tau**2 + .63684256/tau**4
    y = rho*b/4
    base = (-jnp.log1p(-y) - (133/3-1)/(1-y)
            + (11+133/3+1)/(2*(1-y)**2) + rho*(bb-3.5*b)
            - (11-133/3+3)/2 + jnp.log(rho*4.6152*T/1.01325))
    g = jnp.array([
        -.53062968529023e4, .22744901424408e5, .78779333020687e4, -.69830527374994e3,
        .17863832875422e6, -.39514731563338e6, .33803884280753e6, -.13855050202703e6,
        -.25637436613260e7, .48212575981415e7, -.34183016969660e7, .12223156417448e7,
        .11797433655832e8, -.21734810110373e8, .10829952168620e8, -.25441998064049e7,
        -.31377774947767e8, .52911910757704e8, -.13802577177877e8, -.25109914369001e7,
        .46561826115608e8, -.72752773275387e8, .41774246148294e7, .14016358244614e8,
        -.31555231392127e8, .47929666384584e8, .40912664781209e7, -.13626369388386e8,
        .69625220862664e7, -.10834900096447e8, -.22722827401688e7, .38365486000660e7,
        .68833257944332e5, .21757245522644e6, -.26627944829770e5, -.70730418082074e6,
    ])
    # Floating exponent arrays also support implicit-root AD in JAX 0.4.30.
    k = jnp.array([1]*4 + [2]*4 + [3]*4 + [4]*4 + [5]*4 + [6]*4 + [7]*4 + [9]*4 + [3, 3, 1, 5], dtype=float)
    ell = jnp.array([1, 2, 4, 6]*8 + [0, 3, 3, 3], dtype=float)
    residual = jnp.sum(g/k/tau**ell * (-jnp.expm1(-rho))**k)
    delta = rho/jnp.array([.319, .310, .310, 1.55]) - 1
    exponent = (-jnp.array([34., 40., 30., 1050.])*jnp.concatenate((delta[:3]**2, delta[3:]**4))
                - jnp.array([2e4, 2e4, 4e4, 25.])*(T/jnp.array([640., 640., 641.6, 270.])-1)**2)
    residual += jnp.sum(jnp.array([-2.25, -16.8*delta[1]**2, .55, -930.])
                        * jnp.where(exponent > -100, jnp.exp(exponent), 0.))
    c = jnp.array([-.483429455355, .605743189245e1, 22.56023885, -9.87532442,
                  -.43135538513e1, .458155781, -.47754901883e-1, .41238460633e-2,
                  -.27929052852e-3, .14481695261e-4, -.56473658748e-6, .16200446e-7,
                  -.3303822796e-9, .451916067368e-11, -.370734122708e-13, .137546068238e-15])
    tr = T/100
    ideal = 1 + (19.730271018/tr + 20.9662681977)*jnp.log(tr) + jnp.sum(c*tr**jnp.arange(-3., 13.))
    return 4.6152*T*(base-ideal) + residual


def _haar_density(T):
    # Solve rho**2 A_rho = 1 bar. The ideal vapor guess is already close
    # for T >= 700 K. Implicit differentiation avoids tracing the iterations.
    def residual(rho):
        return rho**2 * jax.grad(_haar_helmholtz, 1)(T, rho) - 1

    def solve(fun, guess):
        return jax.lax.fori_loop(0, 8, lambda _, r: r-fun(r)/jax.grad(fun)(r), guess)

    return jax.lax.custom_root(residual, 1/(4.6152*T), solve,
                               lambda linear, value: value/linear(jnp.ones_like(value)))


def _water_standard_gibbs(T, P):
    """Return MELTS dissolved H2O G in J/mol; K, Pa, scalar T >= 700 K.

    The pressure polynomial uses absolute bar, not P_bar - 1. The Haar
    reference is always evaluated at 1 bar, independently of liquid P.
    Caller validates the domain and enables x64, as for magma standards.
    """
    rho = _haar_density(T)
    haar = 1.80152*(_haar_helmholtz(T, rho) + 1/rho) - 285829.96 - 298.15*69.9146 + 54955.23970014
    p = P/1e5
    phi = ((.110/T + 4.432e-5 + 1.405e-7*T - 2.394e-11*T*T)*p
           + (7.337e-8/T - 1.170e-8 - 9.502e-13*T)*p**2
           + (1.876e-10/T + 4.586e-13)*p**3 - 1.191e-14*p**4/T)
    robie = (2.9147*jnp.log(T) - 9.6863e-4*T + 6.8593e-8*T*T
             + 77.8899/jnp.sqrt(T) - 28954.8/T - 2263.27/T**2 - 15.8997)
    return haar + 8.3143*T*(-33676./T + 18.3527 + phi - robie)
