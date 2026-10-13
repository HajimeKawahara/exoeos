"""Pure H2O and mixed H2O--CO2 Gibbs energies on the magma reference convention.

Water uses the dilute limit of the pinned MELTS Haar model, including its
Berman reference shift; CO2 uses the pinned idealGasCO2 caloric standard. An explicitly selected existing EOS
supplies the residual contribution. This is a hybrid property model, not
the MELTS fluid EOS or a saturation/phase-amount solver. Enable JAX x64.
"""

import jax
import jax.numpy as jnp

from exoeos._arrays import scalar_array
from exoeos._magma_water import _haar_ideal
from exoeos._magma_carbon import _co2_ideal_gibbs
from exoeos.constants import MOLAR_GAS_CONSTANT
from exoeos.helmholtz import state_tp


FLUID_COMPONENTS = ("H2O", "CO2")


def co2_gas_standard_gibbs(T):
    """Ideal CO2 gas G at 1 bar, J/mol; scalar K, T >= 700 K, x64.

    Analytic H-T*S from idealGasCO2 in the pinned MAGMA fluidPhase.c.
    Retains its 8.31451 Cp constant and H/S reference offsets. This is
    the zero-density standard, not the finite-pressure Duan fluid at 1 bar.
    """
    return _co2_ideal_gibbs(scalar_array(T, "T").astype(float))


def mixed_fluid_gibbs(T, P, n, *, eos):
    """Extensive H2O--CO2 fluid G, J; scalar K/Pa and shape-(2,) mol.

    Order is FLUID_COMPONENTS. Explicitly supply a matching binary EOS,
    e.g. ZhangDuanEOS.from_species(FLUID_COMPONENTS); species identity is
    the caller's responsibility. Its vapor root and differentiability
    restrictions apply. This hybrid uses the native magma gas standards
    with the selected EOS, not the MELTS mixed-fluid EOS or a saturation
    solver. The published ZD09 binary coefficients are retained by that EOS.

    G = n@g0 + R*T*(sum(n_i*ln(x_i)) + N*(ln(P/1e5)+gres_RT)).
    All derivatives follow this scalar. Positive T >= 700 K, P and finite
    nonnegative n are caller contracts. Composition derivatives require
    both amounts positive. Pure endpoints support G and fixed-face T/P
    derivatives; the all-zero phase has G=0 and no chemical potentials.
    """
    T, P = (scalar_array(value, name).astype(float) for value, name in ((T, "T"), (P, "P")))
    n = jnp.asarray(n, dtype=float)
    if n.shape != (2,):
        raise ValueError("n must have shape (2,) in (H2O, CO2) order.")

    def present(n):
        x = n/n.sum()
        standards = jnp.stack((water_gas_standard_gibbs(T), co2_gas_standard_gibbs(T)))
        residual = state_tp(eos, T, P, x).gres_RT
        ideal = jnp.sum(n*jnp.log(jnp.where(x > 0, x, 1.)))
        return n@standards+MOLAR_GAS_CONSTANT*T*(ideal+n.sum()*(jnp.log(P/1e5)+residual))

    return jax.lax.cond(jnp.all(n == 0), lambda n: jnp.zeros_like(T+P+n.sum()), present, n)


def water_gas_standard_gibbs(T):
    """Return ideal H2O gas standard G at 1 bar in J/mol; scalar T in K.

    This is the analytic zero-density limit of MAGMA water.c (revision
    705a0fb315e5054d18275a580562f6121c8e458c), not its real vapor G at 1 bar.
    The original 1 atm logarithm, 1.80152 conversion, 4.6152 gas constant
    and Berman shift are retained. It shares the native liquid's enthalpy
    and entropy reference convention; no solubility correction is fitted.

    Caller contract: finite scalar T >= 700 K; verified through 2500 K.
    Only shapes are checked. Differentiate this function inside the full
    G calculation for thermal properties; do not freeze its value first.
    """
    T = scalar_array(T, "T").astype(float)
    shift = -285829.96 - 298.15*69.9146 + 54955.23970014
    return 1.80152*4.6152*T*(1-_haar_ideal(T)-jnp.log(1.01325)) + shift


def water_fluid_gibbs(T, P, n, *, eos):
    """Return extensive pure H2O fluid G in J; scalar K, Pa, mol.

    ``eos`` must be an explicitly selected pure-water TPHelmholtzEOS
    (e.g. ZhangDuanEOS.from_species(("H2O",)) or the IdealEOS control).
    The EOS is evaluated at x=[1] on its ``vapor`` branch. Its own domain
    and differentiable-root restrictions apply in addition to T >= 700 K,
    P > 0 and n >= 0. Species identity and value ranges are caller contracts.

    G = n * (g0(T) + R*T*(log(P/1e5) + log(phi))). R is the existing
    EOS gas constant; the legacy constants inside g0 and the liquid are
    preserved. All properties follow by differentiating this same scalar.
    An absent phase returns exactly zero G; AD at n=0 is not a chemical
    potential. Evaluate grad(..., argnums=2) at a positive amount (e.g.
    1 mol) to obtain a candidate fluid insertion potential.
    """
    T, P, n = (scalar_array(value, name) for value, name in
               ((T, "T"), (P, "P"), (n, "n")))
    # The shared Haar coefficients use JAX's configured default float dtype.
    dtype = jnp.result_type(T, P, n, jnp.asarray(0., dtype=float))
    T, P, n = (value.astype(dtype) for value in (T, P, n))

    def present(amount):
        residual = state_tp(eos, T, P, jnp.ones(1, dtype=dtype)).gres_RT
        molar = (water_gas_standard_gibbs(T)
                 + MOLAR_GAS_CONSTANT*T*(jnp.log(P/1e5) + residual))
        return amount*molar

    return jax.lax.cond(n == 0, lambda amount: jnp.zeros_like(amount), present, n)
