"""Native JAX MELTS MgO--SiO2 liquid and pure crystalline forsterite.

The fixed liquid basis is (SiO2, Mg2SiO4), not oxide mole fractions.
Temperatures are K, pressures Pa, amounts mol, and Gibbs energies J.
Enable JAX x64 for thermodynamic derivatives and phase comparisons.

Equations and coefficients: MAGMA revision
705a0fb315e5054d18275a580562f6121c8e458c, sources/gibbs.c,
includes/{liq_struct_data,sol_struct_data,param_struct_data_v34}.h.
This is the MELTS (not pMELTS) polynomial EOS and liquid mixing model.
It does not establish stability against omitted phases or a calibrated
MgO--SiO2 phase diagram. See documents/native_magma.rst for provenance.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy


COMPONENTS = ("SiO2", "Mg2SiO4")
R = 8.3143  # J/(mol K); retain the MELTS entropy convention.
W = 3421.0  # J/mol; SiO2--Mg2SiO4 liquid interaction.
_TR = 298.15


def _tp(T, P):
    temperature, pressure = jnp.asarray(T) * 1.0, jnp.asarray(P) * 1.0
    if temperature.ndim != 0 or pressure.ndim != 0:
        raise ValueError("T and P must be scalars; use jax.vmap for batches.")
    return temperature, pressure


def _forsterite_h_s(T):
    # Integrate Cp = k0 + k1/sqrt(T) + k3/T**3 from the Berman reference.
    k0, k1, k3 = 238.64, -2001.3, -1.1624e8
    h = (-2174420.0 + k0 * (T - _TR)
         + 2.0 * k1 * (jnp.sqrt(T) - jnp.sqrt(_TR))
         - 0.5 * k3 * (T**-2 - _TR**-2))
    s = (94.010 + k0 * jnp.log(T / _TR)
         - 2.0 * k1 * (T**-0.5 - _TR**-0.5)
         - k3 / 3.0 * (T**-3 - _TR**-3))
    return h, s


def forsterite_gibbs(T, P):
    """Return crystalline Mg2SiO4 standard Gibbs energy in J/mol.

    ``T`` and ``P`` are scalars in K and Pa. Positive finite inputs must
    be validated by the caller outside JAX transformations. Pressure uses
    the Berman polynomial relative to 1 bar, with volume in J/bar/mol.
    """
    T, P = _tp(T, P)
    h, s = _forsterite_h_s(T)
    dt, dp = T - _TR, P / 1e5 - 1.0
    pressure_g = 4.366 * (
        (1.0 + 29.464e-6 * dt + 88.633e-10 * dt**2) * dp
        - 0.791e-6 * dp**2 / 2.0 + 1.351e-12 * dp**3 / 3.0
    )
    return h - T * s + pressure_g


def _silica_h_s(T):
    # The liquid SiO2 special case is not its tabulated fusion model.
    a, b, c, d = 127.200, -10.777e-3, 4.3127e5, -1463.8

    def below(t):
        h = (-901554.0 + a * (t - _TR) + b * (t**2 - _TR**2) / 2.0
             - c * (1.0 / t - 1.0 / _TR)
             + 2.0 * d * (jnp.sqrt(t) - jnp.sqrt(_TR)))
        s = (48.475 + a * jnp.log(t / _TR) + b * (t - _TR)
             - c / 2.0 * (t**-2 - _TR**-2)
             - 2.0 * d * (t**-0.5 - _TR**-0.5))
        return h, s

    def above(t):
        h, s = below(1480.0)
        return h + 81.373 * (t - 1480.0), s + 81.373 * jnp.log(t / 1480.0)

    return jax.lax.cond(T >= 1480.0, above, below, T)


def liquid_standard_gibbs(T, P):
    """Return liquid standards [SiO2, Mg2SiO4] in J/mol at scalar K/Pa.

    The SiO2 branch switches at 1480 K, preserving G and its first
    temperature derivative; its heat capacity can jump there. The
    Mg2SiO4 standard uses fusion at 2163 K and constant liquid Cp.
    Input validation and x64 configuration belong to the caller.
    """
    T, P = _tp(T, P)
    h_q, s_q = _silica_h_s(T)
    h_f, s_f = _forsterite_h_s(2163.0)
    h_f = h_f + 2163.0 * 57.2 + 271.0 * (T - 2163.0)
    s_f = s_f + 57.2 + 271.0 * jnp.log(T / 2163.0)
    thermal = jnp.stack((h_q - T * s_q, h_f - T * s_f))
    # Kress EOS, integrated in bar from 1 bar. Reference T is 1673 K.
    dt, dp = T - 1673.0, P / 1e5 - 1.0
    v, dvdt, dvdp, d2vdtp, d2vdp2 = jnp.array([
        [2.690, 4.980], [0.0, 5.24e-4], [-1.89e-5, -1.35e-5],
        [1.3e-8, -1.3e-8], [3.6e-10, 4.14e-10],
    ], dtype=thermal.dtype)
    return (thermal + (v + dvdt * dt) * dp
            + (dvdp + d2vdtp * dt) * dp**2 / 2.0
            + d2vdp2 * dp**3 / 6.0)


def liquid_gibbs(T, P, n):
    """Return extensive liquid G in J for n=[n_SiO2, n_Mg2SiO4] in mol.

    Supply nonnegative finite amounts with shape (2,) and positive finite
    scalar K/Pa. Only static shapes are checked here. Zero component
    amounts use the continuous energy limit, including G(T,P,[0,0])=0.
    Composition derivatives are supported for strictly positive amounts;
    an absent phase has no chemical potentials, and an absent component's
    ideal chemical potential tends to minus infinity. No floor is added.
    For the existing reduced convention, divide by the caller's R*T.
    """
    T, P = _tp(T, P)
    amounts = jnp.asarray(n) * 1.0
    if amounts.shape != (2,):
        raise ValueError("n must have shape (2,) in (SiO2, Mg2SiO4) order.")

    def present(values):
        x = values / jnp.sum(values)
        return (jnp.dot(values, liquid_standard_gibbs(T, P))
                + R * T * jnp.sum(xlogy(values, x))
                + W * values[0] * x[1])

    return jax.lax.cond(
        jnp.all(amounts == 0), lambda values: jnp.zeros_like(T + P + values.sum()),
        present, amounts,
    )
