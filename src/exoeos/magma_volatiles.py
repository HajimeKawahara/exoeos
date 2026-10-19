"""Native dry MELTS with the existing conditional H2O, H2 and trace-He laws.

These are constitutive continuations, not a wet-BSE calibration. Select one
water contribution; the default remains the integrated Thompson law.
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import magma, magma_carbonate, magma_fluid
from ._conditional_data import DATA, R
from ._water_isotherm import two_power_water_g_rt

COMPONENTS = magma.DRY11_COMPONENTS + ("H2O", "H2", "He")
MODEL_ID = "native_dry11_thompson_water_hirschmann_h2_trace_he_v1"
_OXIDE_MASSES = np.array([DATA["oxide_masses_kg_mol"][name.lower()] for name in magma.DRY11_OXIDES])
_OXIDES = np.asarray(jax.vmap(magma.oxide_moles_from_liquid)(jnp.eye(11)))
_MASSES = _OXIDES @ _OXIDE_MASSES
_PREDICTOR = _OXIDES.copy()
_PREDICTOR[:, magma.DRY11_OXIDES.index("FeO")] += 2 * _PREDICTOR[:, magma.DRY11_OXIDES.index("Fe2O3")]
_PREDICTOR[:, magma.DRY11_OXIDES.index("Fe2O3")] = 0
_WEIGHTS = np.zeros(11)
for _name, _value in zip(DATA["water"]["oxide_order"], DATA["water"]["B_K"]):
    _WEIGHTS[magma.DRY11_OXIDES.index(_name)] = _value
_COUNTS, _TEMPERATURE_WEIGHTS = _PREDICTOR.sum(axis=1), _PREDICTOR @ _WEIGHTS
_WATER_MASS = DATA["oxide_masses_kg_mol"]["h2o"]
_LOG_CAPACITY = np.log(DATA["water"]["A_per_sqrt_bar"] * DATA["water"]["water_kg_mol"] / DATA["water"]["OH_kg_mol"])


def helium_capacity(temperature_k, *, host="olivine"):
    """Return mol He/(kg dry host bar fugacity); reciprocal-T interpolation.

    The selected Guillot--Sator simulated dry host and its temperature range
    are retained. No pressure-volume correction or wet-host accuracy is added.
    Out-of-range temperatures return NaN.
    """
    rows = sorted((r for r in DATA["helium"]["He_rows"] if r["host"] == host), key=lambda r: r["T_K"])
    if len(rows) < 2:
        raise ValueError("Select a declared simulated helium host.")
    t = jnp.asarray(temperature_k)
    inverse = jnp.asarray([1/r["T_K"] for r in rows[::-1]])
    log_s = jnp.asarray([np.log(r["S"]) for r in rows[::-1]])
    value = jnp.exp(jnp.interp(1/t, inverse, log_s)) * 1000 / (1e6 * R * 273.15 / 101325)
    return jnp.where((t >= rows[0]["T_K"]) & (t <= rows[-1]["T_K"]), value, jnp.nan)


def water_parameters():
    """Return the exact dry-component coefficients for independent audits."""
    return dict(component_order=magma.DRY11_COMPONENTS,
                dry_molar_masses_kg_mol=_MASSES.copy(), water_molar_mass_kg_mol=_WATER_MASS,
                predictor_oxide_counts=_COUNTS.copy(), predictor_temperature_weights_K=_TEMPERATURE_WEIGHTS.copy(),
                log_capacity_prefactor=_LOG_CAPACITY,
                entropy_coefficient=magma.R/R,
                quadratic_matrix_J_mol=np.asarray(magma._LIQUID_W)+np.asarray(magma._LIQUID_W).T)


def _xlog_ratio(n, denominator):
    return n * jnp.log(jnp.where(n > 0, n / denominator, 1.))


def gibbs_j(temperature_k, pressure_pa, amounts_mol, gas_standards_rt,
            helium_dry_molar_masses_kg_mol, *, h2_offset_rt=0., water_offset_rt=0., helium_host="olivine",
            water_model="thompson2025", water_capacity_1bar=None, water_pressure_exponent=None):
    """Return full extensive G [J] for dry11 + H2O + H2 + He [mol].

    ``gas_standards_rt`` follows H2O, H2, He on the caller's common R*T and
    one-bar reference. He dry masses follow DRY11_COMPONENTS and must use the
    consumer's declared mass basis; they exclude water and molecular H2.
    T/P/amounts, gas standards, masses and offsets are dynamic JAX inputs.
    The supported interior has positive dry components and nonnegative guests,
    with H2O amount no greater than dry-host oxygen. No composition floor or
    phase solver is supplied. The SiO2 1480 K branch remains nonsmooth.

    ``water_model`` is a static selection: ``thompson2025`` (default),
    ``power_law``, ``two_power_law``, ``melts_mode1`` or ``melts_mode4``. The power law requires
    positive supplied C=``water_capacity_1bar`` and alpha=
    ``water_pressure_exponent`` for w=C*(f_H2O/bar)**alpha, where w is H2O
    mass/(dry mass + H2O mass). Its integrated scalar includes host dilution.
    ``two_power_law`` takes length-two capacities and exponents for
    w=sum(C_i*(p_water/bar)**alpha_i), with 0.25<=alpha_i<=1 and w<=0.5.
    The caller must evaluate composition-dependent capacities inside the
    differentiated scalar; freezing them loses the host response. ExoEOS
    supplies no fitted coefficients. Partial-pressure laws require an ideal
    gas or an explicit additional gas-model convention from the caller.
    H2 dilution also contributes to water's chemical potential when present.
    Native hydrous models use the same dry standards and align their H2O gas
    reference to the supplied one-bar standard without fitting an offset.
    Mode 4 fixes carbon at zero. Each water model retains its own empirical
    limitations; selecting it does not establish a shared calibration domain.
    """
    n = jnp.asarray(amounts_mol)
    gas, he_masses = jnp.asarray(gas_standards_rt), jnp.asarray(helium_dry_molar_masses_kg_mol)
    if n.shape != (14,) or gas.shape != (3,) or he_masses.shape != (11,):
        raise ValueError("Require fourteen liquid amounts, three gas standards and eleven dry masses.")
    t, p = jnp.asarray(temperature_k), jnp.asarray(pressure_pa)
    dry, water, hydrogen, helium = n[:11], n[11], n[12], n[13]
    dry_total = jnp.sum(dry)
    if water_model == "two_power_law":
        if water_capacity_1bar is None or water_pressure_exponent is None:
            raise ValueError("Supply two capacities and exponents for two_power_law.")
        equivalent = (jnp.asarray(_MASSES) @ dry) / _WATER_MASS
        water_g = water*(gas[0]+water_offset_rt)
        water_g += two_power_water_g_rt(water, equivalent, water_capacity_1bar, water_pressure_exponent)
        base_g = magma.liquid_gibbs(t, p, dry)
    elif water_model in ("thompson2025", "power_law"):
        equivalent = (jnp.asarray(_MASSES) @ dry) / _WATER_MASS
        if water_model == "thompson2025":
            log_c = _LOG_CAPACITY + (jnp.asarray(_TEMPERATURE_WEIGHTS) @ dry) / (t * (jnp.asarray(_COUNTS) @ dry))
            beta = 2.
        else:
            if water_capacity_1bar is None or water_pressure_exponent is None:
                raise ValueError("Supply the one-bar mass fraction and pressure exponent for power_law.")
            c, alpha = jnp.asarray(water_capacity_1bar), jnp.asarray(water_pressure_exponent)
            log_c = jnp.log(jnp.where(c > 0, c, jnp.nan))
            beta = 1/jnp.where(alpha > 0, alpha, jnp.nan)
        # Integrated mass-fraction water law, including its host response.
        water_g = water * (gas[0] - beta*log_c + water_offset_rt)
        water_g += beta * (-equivalent*jnp.log1p(water/equivalent) + _xlog_ratio(water, equivalent+water))
        base_g = magma.liquid_gibbs(t, p, dry)
    elif water_model in ("melts_mode1", "melts_mode4"):
        hydrous = magma.liquid_gibbs if water_model == "melts_mode1" else magma_carbonate.hydrous_liquid_gibbs
        base_g = hydrous(t, p, n[:12])
        water_g = water * (gas[0] - magma_fluid.water_gas_standard_gibbs(t)/(R*t) + water_offset_rt)
    else:
        raise ValueError("Unknown water model.")
    # Only dilution by H2 is added to the already mixed dry/water host.
    host = dry_total + water
    hydrogen_g = hydrogen * (gas[1] + 11.403 + .76*p/1e9 + h2_offset_rt)
    hydrogen_g += -host*jnp.log1p(hydrogen/host) + _xlog_ratio(hydrogen, host+hydrogen)
    capacity = helium_capacity(t, host=helium_host)
    helium_g = helium * (gas[2] - 1.) + _xlog_ratio(helium, (he_masses @ dry)*capacity)
    value = base_g + R*t*(water_g + hydrogen_g + helium_g)
    oxygen = magma.element_matrix(magma.DRY11_COMPONENTS)[magma.ELEMENTS.index("O")] @ dry
    valid = (jnp.all(n >= 0) & (dry_total > 0) & (water <= oxygen) & (t > 0) & (p > 0))
    return jnp.where(valid, value, jnp.nan)
