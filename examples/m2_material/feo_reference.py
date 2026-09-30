"""Primary Fe-O saturation and an independent associated-liquid reference.

This comparison does not replace Ma Fe-Si-O-H or transfer a binary phase
boundary to an arbitrary multicomponent alloy. Linear pure-component
standards cancel in the binary common-tangent construction.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

R = 8.31446261815324
M_FE, M_O = 55.845, 15.999
SOURCE_PATH = Path(__file__).with_name("steel_data") / "feo_reference_sources.json"


def interactions_rt(temperature_K, pressure_bar=1.):
    """Frost2010 Eqs8-9: J/mol associated species; pressure in bar."""
    t, p = float(temperature_K), float(pressure_bar)
    if not np.isfinite(t+p) or t <= 0 or p <= 0:
        raise ValueError("Require finite positive temperature and pressure.")
    return np.array([135943.-31.122*t-.059*p, 83307.-8.978*t-.09*p])/(R*t)


def mixing_state_rt(temperature_K, pressure_bar, associated_amounts):
    """Return extensive mixing G/RT and Fe,FeO potentials/RT.

    The atom map is n_Fe_atoms=n_Fe_species+n_FeO, n_O_atoms=n_FeO.
    Thus the atomic O potential is mu_FeO-mu_Fe. Associated and atomic
    mole fractions must not be identified.
    """
    n = np.asarray(associated_amounts, float)
    if n.shape != (2,) or np.any(~np.isfinite(n)) or np.any(n <= 0):
        raise ValueError("Require positive Fe and FeO associated amounts.")
    a, b = interactions_rt(temperature_K, pressure_bar)
    total = n.sum()
    y = n[1]/total
    x = 1-y
    g = total*(x*np.log(x)+y*np.log(y)+x*y*(a*x+b*y))
    mu = np.array([np.log(x)+y*y*(b+2*(a-b)*x),
                   np.log(y)+x*x*(a+2*(b-a)*y)])
    return float(g), mu


def binary_coexistence(temperature_K, pressure_bar=1.):
    """Replay the two-liquid branch; not an all-domain phase certificate."""
    def difference(y):
        return (mixing_state_rt(temperature_K, pressure_bar, [1-y[0], y[0]])[1]
                -mixing_state_rt(temperature_K, pressure_bar, [1-y[1], y[1]])[1])
    solved = least_squares(difference, [.02,.96], bounds=([1e-10,.5],[.2,1-1e-10]),
                           xtol=1e-13, ftol=1e-13, gtol=1e-13)
    residual = float(np.max(np.abs(difference(solved.x))))
    if not solved.success or residual > 1e-9:
        raise ValueError("Two-liquid coexistence branch was not resolved.")
    return {"FeO_associated_mole_fractions": solved.x.tolist(),
            "O_mass_percent": (100*M_O*solved.x/(M_FE+M_O*solved.x)).tolist(),
            "max_abs_common_tangent_mu_residual_RT": residual}


def oxygen_activity_shape(temperature_K, pressure_bar, oxygen_mass_percent):
    """Compare binary O activities after aligning infinite-dilution standards."""
    w = float(oxygen_mass_percent)/100
    if not 0 < w < M_O/(M_FE+M_O):
        raise ValueError("Require oxygen strictly between pure Fe and FeO mass fractions.")
    y = w*M_FE/(M_O*(1-w))
    x_o = y/(1+y)
    mu = mixing_state_rt(temperature_K, pressure_bar, [1-y,y])[1]
    a, b = interactions_rt(temperature_K, pressure_bar)
    frost = float(mu[1]-mu[0]-np.log(x_o)-a)
    ma = float(16500./temperature_K*np.log1p(-x_o))
    return {"O_mass_percent": oxygen_mass_percent, "O_atomic_mole_fraction": x_o,
            "FeO_associated_mole_fraction": y,
            "Frost_ln_gamma_O_minus_infinite_dilution": frost,
            "Ma_ln_gamma_O_minus_infinite_dilution": ma,
            "difference_RT": frost-ma,
            "Frost_dilute_atomic_epsilon_OO": float(2+2*b-4*a),
            "Ma_dilute_atomic_epsilon_OO": -16500./temperature_K}


def reference_report(temperature_K=2173.15, pressure_bar=269.6545152505794,
                     oxygen_mass_percent=.08530558765855167):
    sources = json.loads(SOURCE_PATH.read_text())
    observations = []
    for raw in sources["Distin1971"]["observations"]:
        t = raw["temperature_C"]+273.15
        model = binary_coexistence(t)
        fit = 10**(-6380./t+2.765)
        observations.append({**raw, "T_K": t, "published_fit_O_mass_percent": fit,
            "Frost_binary_O_mass_percent": model["O_mass_percent"][0],
            "Frost_minus_measured_mass_percent": model["O_mass_percent"][0]-raw["O_mass_percent"],
            "published_fit_minus_measured_mass_percent": fit-raw["O_mass_percent"],
            "coexistence": model})
    binary = binary_coexistence(temperature_K, pressure_bar)
    return {"record_id": "primary_high_temperature_FeO_reference_v1",
            "T_K": temperature_K, "P_bar": pressure_bar, "observations": observations,
            "target_published_ambient_saturation_fit_O_mass_percent": 10**(-6380./temperature_K+2.765),
            "target_Frost_binary_coexistence": binary,
            "target_binary_activity_shape": oxygen_activity_shape(temperature_K,pressure_bar,oxygen_mass_percent),
            "Frost_excess_only_pressure_shift_O_infinite_dilution_RT": -.059*(pressure_bar-1)/(R*temperature_K),
            "Frost_excess_only_pressure_slope_reported_sigma_RT": .006*(pressure_bar-1)/(R*temperature_K),
            "limitations": sources["limitations"], "accepted_coupled_material_domain": None}


if __name__ == "__main__":
    print(json.dumps(reference_report(), indent=2))
