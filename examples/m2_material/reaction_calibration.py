"""Replay measured water/H references and audit their reaction standards.

Run as ``python -m examples.m2_material.reaction_calibration --output FILE``.
No reference here replaces the current M2 native liquid or alloy standards.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import jax
import numpy as np

from .hydrogen_reference import (ATM_PA, kato_atomic_h_standard_RT,
                                 kato1970_hydrogen_mass_ppm, reference_report)
from .water_calibration import water_calibration_report


R_J_MOL_K = 8.31446261815324
SOURCE_PATH = Path(__file__).with_name("reaction_calibration_sources.json")


def source_h2_standard_audit(temperature_K, pressure_Pa, *, standard_pressure_Pa=1e5):
    """Audit the *adopted* formal H2 standard, not recalibrate its coefficient.

    Its solute basis is molecular mole fraction, with x=K f[bar]. The
    source's experimental-to-MELTS mole denominator remains unverified.
    No temperature dependence of ln(K) was calibrated; evaluating its
    pressure algebra at another temperature does not establish validity.
    """
    t, p, p0 = map(float, (temperature_K, pressure_Pa, standard_pressure_Pa))
    if not all(np.isfinite(v) and v > 0 for v in (t, p, p0)):
        raise ValueError("Require finite positive T, pressure and standard pressure.")
    ln_k_per_bar = -11.403 - .76 * p / 1e9
    ln_k = ln_k_per_bar + np.log(p0/1e5)
    dg_rt = -ln_k
    return {"role": "adopted_M2_formal_standard_audit",
            "reaction": "H2(g) = H2(melt)", "concentration_basis": "molecular mole fraction",
            "ln_K_dimensionless": ln_k, "delta_G0_RT": dg_rt,
            "delta_G0_J_mol": R_J_MOL_K*t*dg_rt,
            "d_ln_K_d_pressure_Pa": -.76e-9,
            "d_ln_K_d_temperature_K_in_adopted_formula": 0.,
            "T_K": t, "P_Pa": p, "standard_pressure_Pa": p0,
            "experimental_mole_basis_to_MELTS_verified": False,
            "temperature_extrapolation_error_bound": None,
            "composition_transfer_error_bound": None,
            "accepted_coupled_material_domain": None}


def illustrative_low_pressure_h2_standard_RT(gas_h2_standard_RT, *, standard_pressure_Pa=1e5):
    """Convert Chaudhari's 515 ppm/GPa illustration to a mass-basis standard.

    This is a pure-H2/ideal-mixture *illustration*, not a measured low-P
    BSE law or an error bound. It is not the paper's buffered Eq.1, whose
    total-pressure slope cannot be treated as arbitrary H2 fugacity.
    mu_H2(melt)/RT = returned_mu0 + ln(w_H2), with molecular H2 amounts.
    """
    gas, p0 = float(gas_h2_standard_RT), float(standard_pressure_Pa)
    if not np.isfinite(gas) or not np.isfinite(p0) or p0 <= 0:
        raise ValueError("Require a finite gas standard and positive p0.")
    k_mass_per_Pa = 515e-6 / 1e9
    return float(gas - np.log(k_mass_per_Pa * p0))


def kato_reaction_audit(temperature_K, *, gas_h2_standard_RT=0., standard_pressure_Pa=1e5):
    """Return the measured-domain dilute pure-Fe atomic-H reaction standard.

    The mass-fraction activity and half-molecular gas coefficient must both
    be retained. This low-pressure comparison is an alternative to, never
    an additive correction on top of, the inherited Okuchi standard.
    """
    t, gas, p0 = map(float, (temperature_K, gas_h2_standard_RT, standard_pressure_Pa))
    mu0 = kato_atomic_h_standard_RT(t, gas, standard_pressure_Pa=p0)
    dg_rt = mu0 - .5*gas
    w = kato1970_hydrogen_mass_ppm(t) * 1e-6
    return {"role": "independent_low_pressure_reference",
            "reaction": "0.5 H2(g) = H(metal)", "concentration_basis": "atomic H mass fraction",
            "T_K": t, "total_pressure_Pa": ATM_PA, "p_H2_Pa": ATM_PA,
            "standard_pressure_Pa": p0, "metal_atomic_H_standard_RT": mu0,
            "ln_K_dimensionless": -dg_rt, "delta_G0_RT": dg_rt,
            "delta_G0_J_mol_H": R_J_MOL_K*t*dg_rt,
            "d_ln_K_d_temperature_K": 1874*np.log(10.)/t**2,
            "equilibrium_reaction_residual_RT": mu0 + np.log(w) - .5*(gas+np.log(ATM_PA/p0)),
            "parameter_covariance": None, "pressure_transfer_error_bound": None,
            "FeSiOH_composition_transfer_error_bound": None,
            "accepted_coupled_material_domain": None}


def audit_metal_water_exchange(temperature_K, *, fe_metal_mu0_RT, feo_liquid_mu0_RT,
                               water_liquid_mu0_RT, h_metal_mu0_RT):
    """Compare supplied standards to the inherited source exchange fit.

    All potentials must use the same R/T and consistent formula/standard
    conventions. A native-MELTS basis transformation must be supplied by
    its owner; this function does not manufacture a pure-oxide standard.
    It detects that changing the host standards need not preserve the
    old fit. A zero residual is a standard bookkeeping check, not evidence
    for applying the high-pressure experiment at an atmospheric interface.
    """
    t = float(temperature_K)
    values = np.array([fe_metal_mu0_RT, feo_liquid_mu0_RT, water_liquid_mu0_RT, h_metal_mu0_RT], dtype=float)
    if not np.isfinite(t) or t <= 0 or not np.all(np.isfinite(values)):
        raise ValueError("Require positive T and finite scalar standard potentials.")
    target = (143589.7-69.1*t)/(R_J_MOL_K*t)
    actual = values[1] + 2*values[3] - values[0] - values[2]
    return {"reaction": "Fe(metal) + H2O(liquid) = FeO(liquid) + 2 H(metal)",
            "inherited_delta_G0_RT": target, "supplied_delta_G0_RT": float(actual),
            "standard_exchange_residual_RT": float(actual-target),
            "scope": "Inherited Young/Okuchi standard bookkeeping; not low-pressure validation.",
            "accepted_coupled_material_domain": None}


def calibration_report():
    return {"report_kind": "reaction_reference_calibration_v1",
            "execution": {"python": platform.python_version(), "numpy": np.__version__,
                          "jax": jax.__version__, "jax_enable_x64": bool(jax.config.x64_enabled),
                          "R_J_mol_K": R_J_MOL_K},
            "sources_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "water": water_calibration_report(), "measured_hydrogen_references": reference_report(),
            "adopted_h2_standard_audit": source_h2_standard_audit(2173.15, 267.20283416157343e5),
            "low_pressure_illustration_h2": {
                "role": "descriptive_extrapolation_not_calibration",
                "mass_basis_delta_G0_RT_at_1bar": illustrative_low_pressure_h2_standard_RT(0.),
                "source": "Chaudhari2025 printed p15 pure-H2 515 mass ppm/GPa illustration",
                "error_bound": None},
            "kato_atomic_h_reaction": kato_reaction_audit(1873.15),
            "native_MELTS_standard_exchange": {"status": "not_evaluated",
                "reason": "Requires supplied, basis-consistent oxide and alloy standards; no fabricated cross-phase alignment."},
            "accepted_coupled_material_domain": None,
            "scientific_acceptance": {"M2_A": "pending", "M2_B": "pending"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = calibration_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
