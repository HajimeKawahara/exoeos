"""Conditional twenty-species Fe-Si-O-H-P/Jung/K/Na extensive Gibbs law.

The model preserves the independently tabulated standards, ideal species
mixing, finite-host perspectives and pseudo-Fe Na prescription. Its numerical
composition domain and empirical admission remain caller responsibilities.
"""

import jax.numpy as jnp
import numpy as np

from . import magma
from ._conditional_data import DATA, R
from .ma_fe_si_o_h import MaFeSiOHLiquid

COMPONENTS = ("Fe", "Si", "O", "H", "P", "Mg", "Ca", "Al", "Cr", "Ti",
              "MgO", "CaO", "AlO", "CrO", "TiO", "Al2O", "Cr2O", "Ti2O", "K", "Na")
FORMULAS = tuple([{name: 1.} for name in COMPONENTS[:10]]
                 + [{name: 1., "O": 1.} for name in ("Mg", "Ca", "Al", "Cr", "Ti")]
                 + [{name: 2., "O": 1.} for name in ("Al", "Cr", "Ti")]
                 + [{"K": 1.}, {"Na": 1.}])
MODEL_ID = "ma_p_jung_k_sodium_pseudoiron_v1"
GAS_STANDARD_SPECIES = ("P1", "O2", "Mg1", "Ca1", "Al1", "Cr1", "Ti1", "K1")
_SOURCE = DATA["associate"]


def _mass_to_mole(epsilon, mass):
    return 100*np.log(10.)*mass/55.845*epsilon + 1 - mass/55.845


def interaction_parameters(temperature_k, *, temperature_policy="constant", hydrogen_oxygen_model="omitted"):
    """Return the original P coefficients and symmetric added-species matrix.

    The policy refers only to the explicitly continued extra coefficients.
    Published a/T+b terms retain their own temperature dependence.
    """
    if temperature_policy not in ("constant", "enthalpic"):
        raise ValueError("Select constant or enthalpic interaction continuation.")
    if hydrogen_oxygen_model not in ("omitted", "schenck1961_abstract"):
        raise ValueError("Select a declared hydrogen/oxygen interaction model.")
    t = jnp.asarray(temperature_k) * 1.0
    factor = 1. if temperature_policy == "constant" else 1873.15/t
    epsilon = factor*jnp.asarray([7.3, 11.9, _mass_to_mole(.06, 30.973761998), _mass_to_mole(.015, 30.973761998)])
    matrix = jnp.zeros((18, 18), dtype=t.dtype)
    for element, (a, b) in _SOURCE["self_epsilon"].items():
        i = COMPONENTS.index(element)
        matrix = matrix.at[i, i].set(a/t+b)
    for pair, (a, b) in _SOURCE["cross_epsilon"].items():
        i, j = (COMPONENTS.index(name) for name in pair.split(","))
        matrix = matrix.at[i, j].set(a/t+b).at[j, i].set(a/t+b)
    for pair, row in _SOURCE["extra_cross"].items():
        i, j = (COMPONENTS.index(name) for name in pair.split(","))
        value = row["epsilon"] if "epsilon" in row else _mass_to_mole(row["mass_percent_e_H_Cr"], row["solute_molar_mass_g_mol"])
        matrix = matrix.at[i, j].set(value*factor).at[j, i].set(value*factor)
    if hydrogen_oxygen_model != "omitted":
        row = DATA["hydrogen_oxygen"]
        scale = 1. if temperature_policy == "constant" else row["reported_temperature_K"]/t
        value = row["reported_log10_gamma_H_derivative_wrt_O_mole_fraction"]*np.log(10.)*scale
        matrix = matrix.at[2, 3].set(value).at[3, 2].set(value)
    return epsilon, matrix


def excess_gibbs_rt(temperature_k, pressure_pa, amounts_mol, phosphorus_epsilon, additional_matrix):
    """Return extensive excess G/(R*T) [mol]; explicit dynamic coefficients."""
    n = jnp.asarray(amounts_mol)
    epsilon, matrix = jnp.asarray(phosphorus_epsilon), jnp.asarray(additional_matrix)
    if n.shape != (20,) or epsilon.shape != (4,) or matrix.shape != (18, 18):
        raise ValueError("Require twenty species, four P coefficients and an eighteen-species matrix.")
    # Na changes excess-energy coordinates only; atomic ledgers stay separate.
    core = n[:18].at[0].add(n[19])
    four, five, total = jnp.sum(core[:4]), jnp.sum(core[:5]), jnp.sum(core)
    parent = four*MaFeSiOHLiquid().gex_RT(temperature_k, pressure_pa, core[:4]/four)
    phosphorus = (.5*epsilon[0]*core[4]**2 + core[4]*(epsilon[1:] @ core[1:4]))/five
    return parent + phosphorus + .5*(core @ matrix @ core)/total


def gibbs_j(temperature_k, pressure_pa, amounts_mol, standard_potentials_rt,
            phosphorus_epsilon, additional_matrix):
    """Return full G [J], including standards and ideal twenty-species mixing."""
    n, standard = jnp.asarray(amounts_mol), jnp.asarray(standard_potentials_rt)
    if n.shape != (20,) or standard.shape != n.shape:
        raise ValueError("Require twenty amounts and twenty common-R standards.")
    mixing = jnp.sum(n*jnp.log(jnp.where(n > 0, n/jnp.sum(n), 1.)))
    return R*temperature_k*(n @ standard + mixing + excess_gibbs_rt(
        temperature_k, pressure_pa, n, phosphorus_epsilon, additional_matrix))


def _janaf_rt(temperature, rows, hf298):
    data = np.asarray(rows, dtype=float)
    grid, entropy, enthalpy = data.T
    values = hf298*1000 + enthalpy*1000 - grid*entropy
    t = jnp.asarray(temperature)
    index = jnp.clip(jnp.searchsorted(jnp.asarray(grid), t, side="right")-1, 0, len(grid)-2)
    width = jnp.asarray(np.diff(grid))[index]
    z = (t-jnp.asarray(grid)[index])/width
    g0, g1 = jnp.asarray(values)[index], jnp.asarray(values)[index+1]
    m0, m1 = -jnp.asarray(entropy)[index], -jnp.asarray(entropy)[index+1]
    value = (2*z**3-3*z**2+1)*g0 + (z**3-2*z**2+z)*width*m0
    value += (-2*z**3+3*z**2)*g1 + (z**3-z**2)*width*m1
    return jnp.where((t >= grid[0]) & (t <= grid[-1]), value/(R*t), jnp.nan)


def sodium_reference_parameters(*, projection, temperature_policy):
    """Select the pinned conditional fit; no hidden default projection."""
    for row in DATA["sodium_fits"]:
        if (row["projection"] == projection and row["temperature_policy"] == temperature_policy
                and row["liquid_model"] == "published"):
            return np.asarray([row["intercept_A_rt"], row["inverse_temperature_B_K"]])
    raise ValueError("Unknown sodium reference projection or temperature policy.")


def standard_potentials_rt(temperature_k, pressure_pa, host_standards_rt, gas_standards_rt,
                           *, potassium_offset_rt, sodium_delta, phosphorus_shift_kcal_mol=0.):
    """Evaluate actual twenty-species standards in the declared common gauge.

    Host inputs are Fe/Si/O/H standards including the original Ma conversion,
    before any paper-specific O/H offsets. Gas inputs follow
    GAS_STANDARD_SPECIES. P uses the continued P(g) dissolution law; gas
    anchors are never replaced by arbitrary zeros. Na uses current native dry
    virtual standards and the explicitly selected delta=A+B/T.
    """
    t, p = jnp.asarray(temperature_k), jnp.asarray(pressure_pa)
    host, gas = jnp.asarray(host_standards_rt), jnp.asarray(gas_standards_rt)
    delta = jnp.asarray(sodium_delta)
    if host.shape != (4,) or gas.shape != (8,) or delta.shape != (2,):
        raise ValueError("Require four host standards, eight gas standards and two sodium coefficients.")
    phosphorus = (gas[0] + jnp.log(1.01325) + (-95.3+.0155*t+phosphorus_shift_kcal_mol)*4184/(R*t)
                  + jnp.log(100*30.973761998/55.845))
    free = {}
    for index, element in enumerate(COMPONENTS[5:10], start=2):
        correction = 0.
        if element in _SOURCE["pure_phase_janaf"]:
            pure, atom = _SOURCE["pure_phase_janaf"][element], _SOURCE["atomic_janaf"][element]
            correction = _janaf_rt(t, pure["rows"], pure["hf298"])-_janaf_rt(
                t, [[row[0], row[1], row[3]] for row in atom["data"]], atom["formation_enthalpy_298_kJ_mol"])
        a, b = _SOURCE["henry"][element]["ln_gamma"]
        free[element] = gas[index]+correction+a/t+b
    a, b = _SOURCE["henry"]["O"]["ln_gamma"]
    oxygen = .5*gas[1]+a/t+b
    values = [*host, phosphorus, *(free[e] for e in COMPONENTS[5:10])]
    for name, formula in zip(COMPONENTS[10:18], FORMULAS[10:18]):
        metal = next(e for e in formula if e != "O")
        values.append(formula[metal]*free[metal]+oxygen+_SOURCE["association_delta_g_J_mol"][name]/(R*t))
    values.append(gas[-1]+potassium_offset_rt)
    native = magma.liquid_standard_gibbs(t, p, include_fe=True, include_ca_al=True, include_ferric=True, include_dry11=True)/(R*t)
    values.append(.5*native[8]-.25*native[0]-.25*native[2]+.5*host[0]+delta[0]+delta[1]/t)
    return jnp.where((t >= 2100) & (t <= 2400), jnp.stack(values), jnp.nan)
