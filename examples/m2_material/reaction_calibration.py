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


def audit_formal_reduction_standards(temperature_K, pressure_Pa, *, standards_rt,
                                     standard_conventions, standard_pressure_Pa=1e5):
    """Audit supplied model reaction sums without claiming empirical alignment.

    Supply mu0/(R*T) on the common R below for six named components. Liquid
    standards are native pure endmembers, not pure oxide standards. Metal
    inputs must retain the caller's applied alloy convention; a linear
    standard parameter need not equal its pure-component limit. Gas inputs
    must come from the actual selected model at its stated standard pressure.
    Three nonempty convention descriptions prevent a silently mixed basis;
    they document the caller's assertion, not independently verify it.

    The virtual FeO combination is half (Fe2SiO4 - SiO2). Its reaction sum
    is meaningful bookkeeping, not a measured/native pure-FeO standard.
    ExoEOS does not import a consumer or extract its model standards here.
    """
    t, p, p0 = map(float, (temperature_K, pressure_Pa, standard_pressure_Pa))
    names = ("sio2_liquid", "fe2sio4_liquid", "Fe_metal", "Si_metal", "H2_gas", "H2O_gas")
    if (not all(np.isfinite(v) and v > 0 for v in (t, p, p0))
            or not isinstance(standards_rt, dict) or set(standards_rt) != set(names)
            or not isinstance(standard_conventions, dict)
            or set(standard_conventions) != {"liquid", "metal", "gas"}
            or any(not isinstance(v, str) or not v.strip() for v in standard_conventions.values())):
        raise ValueError("Supply positive T/P, exactly six standards and three explicit phase conventions.")
    if any(isinstance(standards_rt[name], (bool, np.bool_, str)) for name in names):
        raise ValueError("Standard potentials must be finite real scalars.")
    try:
        mu = np.array([standards_rt[name] for name in names], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Standard potentials must be finite real scalars.") from error
    if mu.shape != (6,) or not np.all(np.isfinite(mu)):
        raise ValueError("Standard potentials must be finite real scalars.")
    # Rows: Fe, Si, O, H. Columns follow the explicit standards above.
    formula = np.array([[0, 2, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
                        [2, 4, 0, 0, 0, 1], [0, 0, 0, 0, 2, 2]])
    reactions = (
        ("Fe2SiO4(liquid) + 2 H2(g) = 2 Fe(metal) + SiO2(liquid) + 2 H2O(g)",
         [1., -1., 2., 0., -2., 2.], False),
        ("SiO2(liquid) + 2 H2(g) = Si(metal) + 2 H2O(g)",
         [-1., 0., 0., 1., -2., 2.], False),
        ("FeO(virtual) + H2(g) = Fe(metal) + H2O(g)",
         [.5, -.5, 1., 0., -1., 1.], True))
    rows = []
    for reaction, coefficients, virtual in reactions:
        coefficients = np.asarray(coefficients)
        imbalance = formula @ coefficients
        if np.any(imbalance != 0):
            raise RuntimeError("The declared reduction reaction does not conserve atoms.")
        dg = float(coefficients @ mu)
        if not np.isfinite(dg):
            raise ValueError("The reaction sum exceeds finite floating-point range.")
        log_k = -dg
        # Preserve logarithms instead of overflowing or inventing exact-zero K.
        k = float(np.exp(log_k)) if np.log(np.nextafter(0., 1.)) <= log_k <= np.log(np.finfo(float).max) else None
        rows.append({"reaction": reaction, "stoichiometry": coefficients.tolist(),
                     "element_imbalance": imbalance.tolist(), "uses_virtual_FeO": virtual,
                     "delta_G0_RT": dg, "delta_G0_J_mol_reaction": R_J_MOL_K*t*dg,
                     "ln_K_dimensionless": log_k, "K_dimensionless": k})
    return {"role": "supplied_model_reaction_bookkeeping", "T_K": t, "P_Pa": p,
            "R_J_mol_K": R_J_MOL_K, "standard_pressure_Pa": p0,
            "component_order": list(names), "standards_rt": dict(zip(names, mu.tolist())),
            "standard_conventions": dict(standard_conventions), "element_order": ["Fe", "Si", "O", "H"],
            "reactions": rows, "virtual_FeO_mu0_RT": float(.5*(mu[1]-mu[0])),
            "virtual_FeO_is_native_or_measured_pure_standard": False,
            "conventions_independently_verified": False, "empirical_alignment_accepted": False,
            "accepted_coupled_material_domain": None,
            "scope": "Formal sums in the supplied model conventions; neither an empirical equilibrium constant nor a phase-applicability certificate."}


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
