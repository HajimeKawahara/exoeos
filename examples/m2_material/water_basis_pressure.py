"""Absorptivity-based water normalization and independent pressure replay.

The new interpretation converts Thompson's Eq.1 mass-scaled response
back to H2O-equivalent mass, using the H2O-based absorptivities inherited
from the primary experiments. It is not literal OH-group mass. Native
MELTS and the archived literal-OH scenario are not modified.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from exoeos import ZhangDuanEOS, state_tp
from .water_calibration import (OH_KG_MOL, WATER_KG_MOL, load_observations,
                                thompson_oh_response, water_dissolution_state)


DATA_DIR = Path(__file__).with_name("water_data")
SOURCE_PATH = DATA_DIR / "water_basis_pressure_sources.json"


def water_equivalent_response(temperature_K, fugacity_Pa, oxide_mole_fractions):
    """Return H2O-equivalent mass fraction from Thompson Eq.1 absorptivity.

    The calibrated 3550 cm^-1 coefficient counts H2O-equivalent moles.
    Multiplying its signal by M_OH (Thompson Eq.1) does not change that
    molar convention. Undo M_OH and multiply by M_H2O. A second factor of
    two would count the two hydroxyl H atoms twice. This unit conversion
    neither resolves absorption-systematic errors nor validates pressure
    extrapolation or an explicit molecular-H2O/OH speciation model.
    """
    return (thompson_oh_response(temperature_K, fugacity_Pa, oxide_mole_fractions)
            * WATER_KG_MOL / OH_KG_MOL)


class WaterEquivalentState(NamedTuple):
    """Extensive G/(RT), potentials and the author's mass-scaled response."""

    gibbs_RT: jax.Array
    host_mu_RT: jax.Array
    water_mu_RT: jax.Array
    author_response: jax.Array


def water_equivalent_dissolution_state(temperature_K, dry_oxide_amounts_mol,
                                       water_amount_mol, gas_water_standard_RT,
                                       *, standard_pressure_Pa=1e5):
    """Integrable dry-host completion on the reconstructed water basis.

    The legacy literal-OH completion has half this capacity. Changing
    capacity by two changes mu_water and G/(RT) by -2 ln(2) and
    -2 n_water ln(2), respectively. Host potentials and curvature are
    unchanged. The scalar still assumes dilute OH behavior, and cannot
    replace native hydrated MELTS without removing its water contribution.
    The final field is the author's mass-scaled response, not OH-group mass.
    """
    state = water_dissolution_state(
        temperature_K, dry_oxide_amounts_mol, water_amount_mol, gas_water_standard_RT,
        response_interpretation="literal_OH_group_mass", standard_pressure_Pa=standard_pressure_Pa)
    water = jnp.asarray(water_amount_mol)
    shift = -2 * jnp.log(2.)
    return WaterEquivalentState(state.gibbs_RT + water * shift, state.host_mu_RT,
                                state.water_mu_RT + shift, state.OH_mass_fraction / 2)


def normalization_replay():
    """Verify the new conversion against the original archived Sossi table."""
    with (DATA_DIR / "Sossi23_Data.csv").open(encoding="utf-8-sig", newline="") as stream:
        original = list(csv.DictReader(stream))
    pooled = {row["Sample"]: row for row in load_observations()}
    rows = []
    for raw in original:
        if not raw["Sample"]:
            continue  # Blank author spreadsheet footer, not an observation.
        combined = pooled[raw["Sample"]]
        observed = float(raw["H2O_ppm_mean"])
        reconstructed = float(combined["OH_Conc_ppm"]) * WATER_KG_MOL / OH_KG_MOL
        rows.append({"sample": combined["Sample"], "original_H2O_mass_ppm": observed,
                     "reconstructed_H2O_mass_ppm": reconstructed,
                     "difference_mass_ppm": reconstructed - observed})
    return {"conversion": "author_response * M_H2O / M_OH",
            "equivalent_molar_species": "H2O", "observations": rows,
            "max_abs_rounding_difference_mass_ppm": max(abs(r["difference_mass_ppm"]) for r in rows)}


def load_pressure_observations():
    sources = json.loads(SOURCE_PATH.read_text())
    path = DATA_DIR / sources["table"]["name"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != sources["table"]["sha256"]:
        raise ValueError("Independent water observations failed the source hash check.")
    with path.open(newline="") as stream:
        return [{key: (value if key == "sample" else None if value == "" else float(value))
                 for key, value in row.items()} for row in csv.DictReader(stream)]


def starting_oxide_fractions():
    """Table1 remelted composition; all ten dry oxides remain in denominator."""
    sources = json.loads(SOURCE_PATH.read_text())
    composition = sources["starting_composition"]
    moles = {key: value / composition["molar_mass_g_mol"][key]
             for key, value in composition["mass_percent"].items()}
    denominator = sum(moles.values())
    return np.asarray([moles[key] / denominator
                       for key in ("CaO", "SiO2", "MgO", "Al2O3", "FeO")])


def pressure_replay(*, gas_model="ideal_partial_pressure"):
    """Unfitted 500/1000 bar validation, with gas EOS declared separately.

    Zhang-Duan2009 is a provider-supported alternative to the original
    paper's Pitzer-Sterner1994 + Aranovich-Newton1999 recipe. Neither this
    alternative nor ideal partial pressure is labelled the paper's EOS.
    Starting Fe is represented as FeO-total, as in the pooled predictors;
    variable Fe redox/loss and melt/glass speciation remain limitations.
    Empirical residuals describe these measured cases, not a confidence
    envelope for unmeasured BSE at a different T, P and composition.
    """
    if gas_model not in ("ideal_partial_pressure", "zhang_duan2009"):
        raise ValueError("Choose an explicit supported gas model.")
    t = 1523.15
    x = starting_oxide_fractions()
    if gas_model == "zhang_duan2009":
        eos = ZhangDuanEOS.from_species(("H2O", "CO2"))
        lnphi = jax.jit(lambda p, y: state_tp(eos, t, p, jnp.array([y, 1-y])).log_fugacity_coefficients[0])
    rows = []
    for raw in load_pressure_observations():
        p = raw["P_MPa"] * 1e6
        y = raw["fluid_H2O_mole_fraction"]
        log_phi = 0. if gas_model == "ideal_partial_pressure" else float(lnphi(p, y))
        fugacity = p * y * np.exp(log_phi)
        predicted = 100 * float(water_equivalent_response(t, fugacity, x))
        if not np.isfinite(predicted):
            raise ValueError("Pressure replay returned a nonfinite prediction.")
        comparison = {}
        for column in ("KFT_H2O_wt_percent", "NIR_total_H2O_wt_percent",
                       "NIR_OH_H2O_equivalent_wt_percent"):
            measured = raw[column]
            comparison[column] = None if measured is None else {
                "predicted_over_measured": predicted / measured,
                "difference_wt_percent": predicted - measured,
                "legacy_literal_OH_predicted_over_measured": predicted / (2 * measured)}
        rows.append({**raw, "T_K": t, "P_Pa": p, "log_phi_H2O": log_phi,
                     "f_H2O_Pa": fugacity, "predicted_H2O_equivalent_wt_percent": predicted,
                     "comparisons": comparison})
    statistics = {}
    for column in rows[0]["comparisons"]:
        comparisons = [row["comparisons"][column] for row in rows if row["comparisons"][column] is not None]
        ratios = np.asarray([v["predicted_over_measured"] for v in comparisons])
        residuals = np.asarray([v["difference_wt_percent"] for v in comparisons])
        statistics[column] = {"count": len(comparisons), "ratio_min": float(min(ratios)),
                              "ratio_max": float(max(ratios)),
                              "RMSE_wt_percent": float(np.sqrt(np.mean(residuals**2))),
                              "max_multiplicative_discrepancy": float(max(max(ratios), 1/min(ratios)))}
    return {"status": "independent_unfitted_pressure_comparison", "gas_model": gas_model,
            "original_paper_gas_EOS_reproduced": False,
            "prediction_basis": "H2O_equivalent_mass_from_published_absorptivity",
            "normalization": normalization_replay(), "statistics": statistics,
            "observations": rows, "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
            "limitations": json.loads(SOURCE_PATH.read_text())["limitations"],
            "accepted_coupled_material_domain": None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gas-model", choices=("ideal_partial_pressure", "zhang_duan2009"), default="ideal_partial_pressure")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(pressure_replay(gas_model=args.gas_model), indent=2, allow_nan=False) + "\n")
