"""Explicit atomic-H standard scenarios from Kato's pure-Fe measurements.

This converts the existing low-pressure mass-basis reference to a caller's
mole-fraction convention, without adopting an FeSiOH pressure/activity law.
"""

import hashlib
import json

import numpy as np

from .hydrogen_reference import ATM_PA, DATA_PATH, kato1970_hydrogen_mass_ppm
from .oxygen_calibration import _finite_scalar


H_TO_FE_MASS_RATIO = 1.00794/55.845


def atomic_h_standard_scenario(temperature_K, pressure_Pa, *, hydrogen_standard_rt,
                               hydrogen_lngamma_infinite_dilution,
                               hydrogen_gas_standard_rt, standard_pressure_Pa=1e5,
                               allow_temperature_extrapolation=False):
    """Return an H-only offset matching 0.5 H2(g) = H(metal) in dilute Fe.

    Inputs are atomic-H mu0/(RT), its matching ln(gamma_H,infinity), and
    molecular-H2 mu0/(RT), all on the same R/T and elemental gauge. The
    Henry mass-fraction standard equals mu0_x + ln(gamma_infinity)
    - ln(M_H/M_Fe). The gas p0 is explicit; 1 atm is not silently 1 bar.

    A standard offset is added to the baseline *instead of* adding the
    target standard itself. It changes only a linear G term, not Ma gE.
    Extrapolation and application at other total pressures remain explicit
    conditional scenarios with unknown transfer errors.
    """
    names = ("T_K", "P_Pa", "standard_pressure_Pa", "hydrogen_standard_rt",
             "hydrogen_lngamma_infinite_dilution", "hydrogen_gas_standard_rt")
    values = (temperature_K, pressure_Pa, standard_pressure_Pa, hydrogen_standard_rt,
              hydrogen_lngamma_infinite_dilution, hydrogen_gas_standard_rt)
    t, p, p0, h, gamma, gas = [_finite_scalar(v, name, positive=i < 3)
                               for i, (v, name) in enumerate(zip(values, names))]
    if not isinstance(allow_temperature_extrapolation, bool):
        raise ValueError("allow_temperature_extrapolation must be bool.")
    inside = 1843.15 <= t <= 2013.15
    if not inside and not allow_temperature_extrapolation:
        raise ValueError("Temperature extrapolation requires explicit opt-in.")
    ppm = (kato1970_hydrogen_mass_ppm(t) if inside
           else float(1e4*10.**(-1874./t-1.601)))
    measured_w = ppm*1e-6
    target_henry = .5*(gas+np.log(ATM_PA/p0))-np.log(measured_w)
    adopted_henry = h+gamma-np.log(H_TO_FE_MASS_RATIO)
    offset = target_henry-adopted_henry
    return {"role": "explicit_measured_atomic_H_standard_scenario",
            "reaction": "0.5 H2(g) = H(metal)",
            "concentration_basis": "Henry atomic-H mass fraction in dilute pure liquid Fe",
            "source_doi": "10.2355/tetsutohagane1955.56.5_521",
            "source_data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            "T_K": t, "P_Pa": p, "standard_pressure_Pa": p0,
            "reference_p_H2_Pa": ATM_PA, "reference_total_pressure_Pa": ATM_PA,
            "reference_H_mass_ppm": ppm,
            "input_standards_rt": dict(zip(names[3:], (h,gamma,gas))),
            "H_mass_fraction_per_atomic_mole_fraction_at_infinite_dilution": H_TO_FE_MASS_RATIO,
            "adopted_Henry_H_standard_rt": float(adopted_henry),
            "target_Henry_H_standard_rt": float(target_henry),
            "target_H_standard_rt": float(h+offset),
            "standard_offsets_rt": {"H_metal": float(offset)},
            "inside_measured_temperature_envelope": inside,
            "temperature_distance_K": max(1843.15-t,t-2013.15,0.),
            "pressure_distance_Pa": abs(p-ATM_PA),
            "temperature_transfer_error_bound": None,
            "pressure_transfer_error_bound": None,
            "FeSiOH_composition_transfer_error_bound": None,
            "excess_Gibbs_model_changed": False, "baseline_replaced": False,
            "accepted_coupled_material_domain": None}


def kato_reference_observations():
    """Preserve the measured pure-Fe receipts; do not digitize unseen curves."""
    data = json.loads(DATA_PATH.read_text())["kato_1970"]
    pure, dilute = data["pure_iron"], data["lower_partial_pressure_observation"]
    return [{"T_K": 1873.15, "p_H2_Pa": ATM_PA,
             "observed_H_mass_ppm": pure["measured_1600_C_H_mass_ppm"],
             "reported_sigma_ppm": None, "replicates_ppm": None},
            {"T_K": dilute["temperature_K"], "p_H2_Pa": dilute["H2_partial_pressure_Pa"],
             "observed_H_mass_ppm": dilute["reported_mean_H_mass_ppm"],
             "reported_sigma_ppm": dilute["reported_standard_deviation_H_mass_ppm"],
             "replicates_ppm": dilute["H_mass_ppm_replicates"]}]
