"""Reproduce Thompson (2025) OH capacity without changing native MELTS.

The author's fitting notebook defines its ``XOH`` as OH_Conc_ppm / 1e6,
despite the paper's mole-fraction terminology. Oxide predictors are dry
oxide mole fractions. This implementation follows that published code.
Literal OH-group mass/atom conversion remains an explicit interpretation:
the pooled Sossi water-to-OH columns use a different stoichiometric factor.
The functions evaluate a reference law, not an admitted BSE material model.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from exoeos import mass_fraction_solute_state


DATA_DIR = Path(__file__).with_name("water_data")
METADATA_PATH = DATA_DIR / "provenance.json"
OXIDES = ("CaO", "SiO2", "MgO", "Al2O3", "FeO", "Na2O")
# Table 4, Eq.18. Na2O remains in the dry denominator but has no fitted term.
B_K = (-1511.1, 886.5, -1015.2, 890.6, -1755.9)
B_SIGMA_K = (513.3, 315.9, 268.9, 795.4, 364.5)
A_PER_SQRT_BAR = 7.19e-4
OXIDE_MASSES_KG_MOL = (.0560774, .06008, .0403044, .10196, .071844, .0619789)
OH_KG_MOL = .017008
WATER_KG_MOL = .0180153
MODEL_ID = "thompson2025_eq18_author_mass_basis_v1"


def load_observations():
    """Read byte-pinned author data; blanks are retained, never fitted as zeros."""
    metadata = json.loads(METADATA_PATH.read_text())
    for record in metadata["archived_files"]:
        payload = (DATA_DIR / record["name"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError("Published water calibration data hash mismatch.")
    with (DATA_DIR / "CombinedDataset_withS23N17.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows


def log_oh_capacity(temperature_K, oxide_mole_fractions):
    """Return ln(C), where w_OH = C sqrt(f_H2O / bar), not a mole fraction.

    Supply the five fitted dry fractions in B_K order; do not renormalize
    away other oxides. This differentiable algebra makes no domain claim.
    Invalid inputs return NaN; fractions may sum to less than one.
    """
    t, x = jnp.asarray(temperature_K), jnp.asarray(oxide_mole_fractions)
    if t.ndim or x.shape != (5,):
        raise ValueError("Require scalar T and five dry oxide mole fractions.")
    if any(jnp.issubdtype(v.dtype, jnp.complexfloating) for v in (t, x)):
        raise ValueError("Water capacity requires real inputs.")
    valid = (jnp.isfinite(t) & (t > 0) & jnp.all(jnp.isfinite(x))
             & jnp.all(x >= 0) & (jnp.sum(x) <= 1 + 1e-8))
    return jnp.where(valid, jnp.log(A_PER_SQRT_BAR) + jnp.dot(jnp.asarray(B_K), x) / t, jnp.nan)


def thompson_oh_response(temperature_K, fugacity_Pa, oxide_mole_fractions):
    """Return the author's OH-ppm/1e6 response; zero fugacity gives zero.

    Fugacity, not total pressure, enters this law. The derivative with
    respect to fugacity is singular at zero; no trace floor is introduced.
    The name follows the notebook's mass scaling, not a resolved atomic
    normalization of every pooled study; see sossi_basis_audit().
    """
    f = jnp.asarray(fugacity_Pa)
    if f.ndim or jnp.issubdtype(f.dtype, jnp.complexfloating):
        raise ValueError("Require a real scalar H2O fugacity in Pa.")
    value = jnp.exp(log_oh_capacity(temperature_K, oxide_mole_fractions)) * jnp.sqrt(f / 1e5)
    return jnp.where(jnp.isfinite(f) & (f >= 0), value, jnp.nan)


def water_reference(temperature_K, fugacity_Pa, oxide_mole_fractions, *,
                    total_pressure_Pa=1e5, allow_extrapolation=False):
    """Evaluate one observed condition, or explicitly label an extrapolation.

    Matching a published point establishes its provenance, not its fit
    accuracy. The scalar model has no total-pressure correction. An
    envelope of separate experiments is never declared a joint domain.
    """
    t, f, p = map(float, (temperature_K, fugacity_Pa, total_pressure_Pa))
    x = np.asarray(oxide_mole_fractions, dtype=float)
    value = float(thompson_oh_response(t, f, x))
    if not np.isfinite(value) or not np.isfinite(p) or p <= 0:
        raise ValueError("Require valid T, total pressure, H2O fugacity and oxide fractions.")
    matches = [row["Sample"] for row in load_observations()
               if p == 1e5 and t == float(row["Temperature_K"])
               and np.isclose(f, float(row["fH2O"]) * 1e5, rtol=1e-12, atol=0)
               and np.allclose(x, [float(row[name + "_MF"]) for name in OXIDES[:5]], rtol=0, atol=1e-10)]
    if not matches and not allow_extrapolation:
        raise ValueError("No matching published condition; explicitly allow extrapolation to evaluate it.")
    b = np.asarray(B_K)
    return {
        "model_id": MODEL_ID,
        "author_OH_response": value,
        "H2O_equivalent_mass_fraction_if_literal_OH": value * WATER_KG_MOL / (2 * OH_KG_MOL),
        "atomic_conversion_status": "requires_confirmation_of_pooled_data_conventions",
        "d_author_OH_response_d_T_K": float(-value * (b @ x) / t**2),
        "d_author_OH_response_d_fugacity_Pa": None if f == 0 else value / (2 * f),
        "d_author_OH_response_d_oxide_fraction": (value * b / t).tolist(),
        "matched_observations": matches,
        "evaluation_scope": "published_predictor_point" if matches else "explicit_extrapolation",
        "total_pressure_correction": "not_fitted",
        "uncertainty": {
            "sigma_ln_capacity_upper_from_parameter_marginals": float(np.abs(x) @ np.asarray(B_SIGMA_K) / t),
            "bound_meaning": "Cauchy-Schwarz bound on parameter-only standard deviation for any covariance with the published marginal SDs; not a confidence interval or total error bound.",
            "parameter_covariance": None,
            "temperature_composition_fugacity_measurement_covariance": None,
            "absorption_coefficient_systematic_error_bound": None,
            "extrapolation_error_bound": None,
        },
        "accepted_coupled_material_domain": None,
    }


class WaterDissolutionState(NamedTuple):
    """Added extensive G/(RT) [mol], host shifts, and water mu/(RT)."""

    gibbs_RT: jax.Array
    host_mu_RT: jax.Array
    water_mu_RT: jax.Array
    OH_mass_fraction: jax.Array


def water_dissolution_state(temperature_K, dry_oxide_amounts_mol, water_amount_mol,
                            gas_water_standard_RT, *, response_interpretation,
                            standard_pressure_Pa=1e5):
    """Integrable, independent completion of the reference water capacity.

    Explicitly select response_interpretation="literal_OH_group_mass".
    This is a conditional completion: the combined dataset's physical
    atom normalization remains unresolved, as recorded by sossi_basis_audit.
    Dry host order is OXIDES. One added component is H2O (2 H, 1 O), whose
    two OH groups include one oxygen already in the dry host. Thus
    w_OH = 2 W_OH n_water / (M_dry + W_water n_water). No OH reservoir or
    extra oxygen is added. At fixed host, mu_water = mu0_gas +
    2 ln(w_OH/C) + ln(1 bar/p0), giving the fitted fugacity equilibrium.

    Twice the integrated mass-fraction dilution supplies the mixing scalar.
    The composition dependence of C also shifts every host potential;
    omitting those derivatives breaks Gibbs consistency. Add this result
    to a *dry* host only, never to native MELTS already containing water.
    This mathematical completion is not a fitted complete liquid G model,
    liquid stability proof, or validation at unmeasured conditions.
    Positive-interior states support JAX differentiation. At zero water,
    G and host shifts vanish and the insertion water mu is -infinity.
    """
    if response_interpretation != "literal_OH_group_mass":
        raise ValueError("Explicitly select the literal_OH_group_mass interpretation; no atomic-basis validation is implied.")
    t, n, h, gas_mu, p0 = map(jnp.asarray, (temperature_K, dry_oxide_amounts_mol,
                                          water_amount_mol, gas_water_standard_RT,
                                          standard_pressure_Pa))
    if n.shape != (6,) or any(v.ndim for v in (t, h, gas_mu, p0)):
        raise ValueError("Require six dry oxide amounts and scalar T, water amount, gas standard and p0.")
    if any(jnp.issubdtype(v.dtype, jnp.complexfloating) for v in (t, n, h, gas_mu, p0)):
        raise ValueError("Water dissolution requires real inputs.")
    total = jnp.sum(n)
    x = n / total
    log_c = log_oh_capacity(t, x[:5])
    standard = gas_mu - 2 * log_c + 2 * jnp.log(2 * OH_KG_MOL / WATER_KG_MOL) + jnp.log(1e5 / p0)
    dilute = mass_fraction_solute_state(0., jnp.zeros(6), n, jnp.asarray(OXIDE_MASSES_KG_MOL), h, WATER_KG_MOL, 0.)
    b = jnp.asarray((*B_K, 0.))
    d_log_c_dn = (b - jnp.dot(b, x)) / (t * total)
    mass = jnp.dot(n, jnp.asarray(OXIDE_MASSES_KG_MOL)) + h * WATER_KG_MOL
    # Each added H2O consumes one pre-existing host oxygen to form two OH.
    host_oxygen_mol = jnp.dot(n, jnp.asarray((1., 2., 1., 3., 1., 1.)))
    valid = (jnp.isfinite(standard) & (p0 > 0) & jnp.isfinite(p0)
             & (h <= host_oxygen_mol))
    return WaterDissolutionState(
        jnp.where(valid, h * standard + 2 * dilute.gibbs_RT, jnp.nan),
        jnp.where(valid, 2 * dilute.host_mu_RT - 2 * h * d_log_c_dn, jnp.nan),
        jnp.where(valid, standard + 2 * dilute.solute_mu_RT, jnp.nan),
        jnp.where(valid & jnp.isfinite(dilute.gibbs_RT), 2 * OH_KG_MOL * h / mass, jnp.nan),
    )


def sossi_basis_audit():
    """Compare two archived author tables without correcting either one.

    If H2O_ppm_mean denotes literal H2O mass ppm, its two H atoms imply
    twice the OH-group mass used by the combined table's numeric mapping.
    This arithmetic check does not resolve the authors' spectroscopy or
    equivalent-species conventions and is not a claim of an author error.
    """
    combined = {row["Sample"]: row for row in load_observations()}
    with (DATA_DIR / "Sossi23_Data.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        if row["Sample"] not in combined:
            continue
        water = float(row["H2O_ppm_mean"])
        reported = float(combined[row["Sample"]]["OH_Conc_ppm"])
        result.append({"sample": row["Sample"], "raw_H2O_ppm_mean_column": water,
                       "combined_OH_ppm_column": reported,
                       "single_molar_mass_ratio_prediction_ppm": water * OH_KG_MOL / WATER_KG_MOL,
                       "literal_two_OH_groups_prediction_ppm": 2 * water * OH_KG_MOL / WATER_KG_MOL,
                       "combined_over_literal_two_group_prediction": reported / (2 * water * OH_KG_MOL / WATER_KG_MOL)})
    return {"status": "physical_atomic_normalization_unresolved",
            "interpretation": "If the raw H2O column is H2O mass ppm, the combined OH column follows one OH/H2O molar-mass factor rather than two. Preserve both and obtain convention clarification before physical coupling.",
            "observations": result}


def sossi2023_water_response(fugacity_water_Pa, fugacity_h2_Pa, *, absorption_basis):
    """Sossi Eq.11 total water-equivalent mass fraction at nominal 2173K.

    Select one of the two published IR calibrations, not their average.
    The H2 term describes OH production, not dissolved molecular H2. This
    two-gas regression is not itself a complete gas/melt redox Gibbs model.
    Algebraic evaluation away from the 1bar experiment is an extrapolation.
    """
    pairs = {"basalt_epsilon_6.3": (524., 183.), "peridotite_epsilon_5.1": (647., 225.)}
    if absorption_basis not in pairs:
        raise ValueError("Select an explicit published absorption-coefficient basis.")
    f = np.asarray([fugacity_water_Pa, fugacity_h2_Pa], dtype=float)
    if f.shape != (2,) or not np.all(np.isfinite(f)) or np.any(f < 0):
        raise ValueError("Require two finite nonnegative fugacities in Pa.")
    return float(np.asarray(pairs[absorption_basis]) @ np.sqrt(f/1e5) * 1e-6)


def sossi_regression_replay():
    """Replay the available original water columns without the pooled OH conversion."""
    load_observations()  # Hash-check both archived CSV files.
    with (DATA_DIR / "Sossi23_Data.csv").open(encoding="utf-8-sig", newline="") as stream:
        raw = [row for row in csv.DictReader(stream) if row["Sample"]]
    records = []
    for row in raw:
        for suffix, basis in (("63", "basalt_epsilon_6.3"), ("51", "peridotite_epsilon_5.1")):
            predicted = 1e6 * sossi2023_water_response(float(row["fH2O"])*1e5, float(row["fH2"])*1e5,
                                                       absorption_basis=basis)
            observed = float(row["H2O_ppm_"+suffix])
            sigma = float(row["H2O_ppm_"+suffix+"_unc"])
            records.append({"sample": row["Sample"], "T_K": float(row["Temperature_K"]),
                            "absorption_basis": basis, "predicted_water_equivalent_mass_ppm": predicted,
                            "observed_water_equivalent_mass_ppm": observed,
                            "residual_over_observed_sigma": (predicted-observed)/sigma})
    return {"scope": "Sossi Eq11 replay on12 rows distributed in Thompson's data; not the complete original14-sample dataset or held-out validation.",
            "nominal_T_K": 2173., "total_pressure_Pa": 1e5,
            "absorption_branches_are_independent_observations": False,
            "H2_term_species": "OH contribution expressed as equivalent water; not molecular H2",
            "observations": records}


def water_calibration_report():
    """Independently replay rounded Table 4 coefficients against all raw rows.

    This is a reproduction on the fitting data, not held-out validation,
    refitting, or a reproduction of unavailable posterior samples.
    """
    rows, records = load_observations(), []
    for index, row in enumerate(rows):
        x = np.array([float(row[name + "_MF"]) for name in OXIDES[:5]])
        t, f = float(row["Temperature_K"]), float(row["fH2O"])
        prediction = float(thompson_oh_response(t, f * 1e5, x)) * 1e6
        observed, sigma = float(row["OH_Conc_ppm"]), float(row["OH_Conc_ppm_unc"])
        records.append({"sample": row["Sample"], "csv_row_index": index,
                        "study": "Thompson2025" if index < 44 else "Sossi2023" if index < 56 else "Newcombe2017",
                        "used_in_author_fit": f > 0, "T_K": t, "f_H2O_bar": f,
                        "f_H2_bar": float(row["fH2"]), "total_pressure_Pa": 1e5,
                        "predicted_OH_mass_ppm": prediction, "observed_OH_mass_ppm": observed,
                        "observed_sigma_ppm": sigma, "residual_ppm": prediction - observed,
                        "residual_over_observed_sigma": (prediction - observed) / sigma})
    included = [row for row in records if row["used_in_author_fit"]]
    residual = np.array([row["residual_ppm"] for row in included])
    observed = np.array([row["observed_OH_mass_ppm"] for row in included])
    return {
        "model_id": MODEL_ID,
        "report_kind": "independent_replay_of_fitting_observations",
        "coefficient_source": "Thompson2025 Table4 rounded posterior means, not a refit",
        "concentration_basis": "Author OH_Conc_ppm / 1e6 response; literal OH atom normalization is unresolved across pooled studies",
        "pooled_sossi_basis_audit": sossi_basis_audit(),
        "sossi_original_water_regression_replay": sossi_regression_replay(),
        "fitted_observation_count": len(included), "blank_count": len(records) - len(included),
        "coefficient_mean_prediction_r_squared": float(1 - np.sum(residual**2) / np.sum((observed - observed.mean())**2)),
        "rmse_OH_mass_ppm": float(np.sqrt(np.mean(residual**2))),
        "max_absolute_residual_over_observed_sigma": max(abs(row["residual_over_observed_sigma"]) for row in included),
        "observations": records,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provenance_sha256": hashlib.sha256(METADATA_PATH.read_bytes()).hexdigest(),
        "accepted_coupled_material_domain": None,
        "scientific_acceptance": {"M2_A": "pending", "M2_B": "pending"},
    }
