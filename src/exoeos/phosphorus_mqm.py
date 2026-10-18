"""Binary Fe-P modified quasichemical liquid Gibbs energies.

Bernhard et al. (2020), doi:10.1016/j.calphad.2020.101795, Table 2 models
I/II; You (2020), McGill thesis Chapter 3, Table 3.2 (related article:
doi:10.1007/s11663-020-01939-0). No coefficients are fitted here.
Amounts count pairs, not atoms or chemical molecules. The caller equilibrates
pairs subject to ``atom_matrix`` and supplies pure-liquid elemental standards.
These binary liquids do not supply solids, pressure or multicomponent laws.
"""

import jax.numpy as jnp

from ._conditional_data import R

PAIRS = ("FeFe", "PP", "FeP")
MODELS = ("bernhard2020_I", "bernhard2020_II", "you2020")


def parameters_j_mol(temperature_k, *, model):
    """Coefficients of (1, X_FeFe, X_FeFe**2, X_PP), J/mol pair reaction.

    Delta g forms two unlike pairs. The unlike coordination numbers
    Z_FeP(Fe), Z_FeP(P) are (3,6) for model I/You and (6,6) for model II.
    """
    coefficients = {
        "bernhard2020_I": ((-57655.52, 8.2425), (-2615., 1.5899),
                           (0., 0.), (-3711.21, -5.23)),
        "bernhard2020_II": ((-44768.8, 5.8576), (-23765.12, 0.),
                            (11296.8, 1.8828), (0., 5.35552)),
        "you2020": ((-56902.4, 6.56888), (5481.04, 3.0334),
                    (-11966.24, 2.5104), (-9623.2, 0.)),
    }
    if model not in MODELS:
        raise ValueError("Select a named Fe-P MQM parameter set.")
    c = jnp.asarray(coefficients[model])
    return c[:, 0] + c[:, 1]*temperature_k


def atom_matrix(*, model):
    """Map (FeFe, PP, FeP) pair moles to (Fe, P) atom moles.

    Eliminating the composition-dependent coordination numbers gives
    n_Fe=2*n_FeFe/6+n_FeP/Z_FeP(Fe), n_P=2*n_PP/6+n_FeP/6.
    The pair exchange at fixed atoms is therefore not (-1,-1,2) in model I.
    """
    if model not in MODELS:
        raise ValueError("Select a named Fe-P MQM parameter set.")
    z = 6. if model == "bernhard2020_II" else 3.
    return jnp.array([[1/3, 0., 1/z], [0., 1/3, 1/6]])


def _ideal(n):
    total = jnp.sum(n)
    return jnp.sum(n*jnp.log(jnp.where(n > 0, n/jnp.where(total > 0, total, 1.), 1.)))


def gibbs_j(temperature_k, pair_amounts_mol, pure_standards_rt, *, model):
    """Extensive G [J] with pure-liquid Fe/P standards in units of RT.

    Nonnegative pair amounts and positive T are caller preconditions. Values
    have continuous pure/empty limits; derivatives are for positive amounts.
    The full configurational entropy and composition-dependent pair energy
    must be minimized together; a constant-delta-g mass-action shortcut would
    omit derivatives of both the pair polynomial and coordination mapping.
    """
    n, standard = jnp.asarray(pair_amounts_mol), jnp.asarray(pure_standards_rt)
    if n.shape != (3,) or standard.shape != (2,):
        raise ValueError("Require three pair amounts and two pure-liquid standards.")
    atoms = atom_matrix(model=model) @ n
    total = jnp.sum(n)
    a, b, _ = n/jnp.where(total > 0, total, 1.)
    neighbours = jnp.array([n[0]+n[2]/2, n[1]+n[2]/2])
    minus_entropy_r = _ideal(atoms)+_ideal(n)-2*_ideal(neighbours)-n[2]*jnp.log(2.)
    delta = parameters_j_mol(temperature_k, model=model) @ jnp.array([1., a, a*a, b])
    return R*temperature_k*(atoms @ standard+minus_entropy_r)+.5*n[2]*delta


def henry_log_gamma(temperature_k, *, model):
    """Exact ln(gamma_P at infinite dilution), relative to pure liquid P.

    As x_P -> 0, n_FeP/n_P -> 6 and the configurational correction is
    higher order. Thus mu_P(Henry)-G_P(pure)=3*delta_g(X_FeFe=1).
    This is derived from G, not an independently adjustable reference shift.
    """
    return 3*jnp.sum(parameters_j_mol(temperature_k, model=model)[:3])/(R*temperature_k)


def sgte_liquid_p_j(temperature_k):
    """SGTE UNARY 5.0 GLIQP high-T branch, J/mol P, white-P SER.

    Source: https://www.sgte.net/en/free-pure-elements-database (2009 file).
    Return NaN outside 317.3--3000 K. This unary branch's range does not
    establish an empirical Fe-P solution domain or a pressure correction.
    Its linear-T coefficient differs by -0.013 J/mol/K from the later
    Oikawa Table S1 transcription; the versions are deliberately distinct.
    """
    t = jnp.asarray(temperature_k)
    value = -7232.449+133.291873*t-26.326*t*jnp.log(jnp.where(t > 0, t, 1.))
    return jnp.where((t >= 317.3) & (t <= 3000.), value, jnp.nan)


def sgte_liquid_fe_j(temperature_k):
    """SGTE UNARY 5.0 GLIQFE, J/mol Fe on the alpha-Fe SER.

    The liquid unary is metastable below its melting point. Its tabulation
    range (298.15--6000 K) is not a liquid-alloy admission range.
    """
    t = jnp.asarray(temperature_k)
    logt = jnp.log(jnp.where(t > 0, t, 1.))
    ser = (1225.7+124.134*t-23.5143*t*logt-4.39752e-3*t**2
           -.058927e-6*t**3+77359/t)
    low = 12040.17-6.55843*t-367.516e-23*t**7+ser
    high = -10838.83+291.302*t-46*t*logt
    return jnp.where((t >= 298.15) & (t <= 6000.),
                     jnp.where(t < 1811., low, high), jnp.nan)


def nist_p2_gas_j(temperature_k):
    """NIST SRD69 P2(g), one bar, J/mol molecule; 298--6000 K.

    https://webbook.nist.gov/cgi/cbook.cgi?ID=C12185090&Mask=1
    Chase (1998) Shomate coefficients, last reviewed June 1961. Restore the
    formation enthalpy, so coefficient H cancels from H(T)-H(298)+Hf(298).
    This function supplies a reference, not a gas EOS at finite pressure.
    """
    t = jnp.asarray(temperature_k)
    x = t/1000.
    a, b, c, d, e, f, g = (35.92198, 1.483806, -.339811, .028919,
                           -.401919, 131.5232, 258.8557)
    h = 1000*(a*x+b*x*x/2+c*x**3/3+d*x**4/4-e/x+f)
    s = a*jnp.log(jnp.where(x > 0, x, 1.))+b*x+c*x*x/2+d*x**3/3-e/(2*x*x)+g
    return jnp.where((t >= 298.) & (t <= 6000.), h-t*s, jnp.nan)


def pure_p_standard_rt(temperature_k, gas_p2_standard_rt):
    """Translate SGTE liquid P into the caller's one-bar P2 elemental gauge.

    The supplied gas standard excludes ln(partial pressure/bar). No Henry
    shift is added: the MQM scalar generates its own concentration law.
    """
    t = temperature_k
    return .5*gas_p2_standard_rt+(sgte_liquid_p_j(t)-.5*nist_p2_gas_j(t))/(R*t)
