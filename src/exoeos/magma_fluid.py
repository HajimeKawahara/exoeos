"""Pure H2O fluid Gibbs energy on the native magma reference convention.

The ideal-gas standard is the dilute limit of the pinned MELTS Haar model,
including its Berman reference shift. An explicitly selected existing EOS
supplies the residual contribution. This is a hybrid property model, not
the MELTS fluid EOS or a saturation/phase-amount solver. Enable JAX x64.
"""

import jax
import jax.numpy as jnp

from exoeos._arrays import scalar_array
from exoeos._magma_water import _haar_ideal
from exoeos.constants import MOLAR_GAS_CONSTANT
from exoeos.helmholtz import state_tp


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
