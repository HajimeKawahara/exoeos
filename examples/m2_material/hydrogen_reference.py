"""Source-restricted molecular-H2 and atomic-H solubility references.

These examples reproduce measured-host regressions. They do not supply a
joint BSE/MELTS/Fe-Si-O-H calibration or a general high-pressure alloy EOS.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DATA_PATH = Path(__file__).with_name("hydrogen_reference.json")
ATM_PA = 101325.


def chaudhari_buffered_h2_mass_ppm(host, temperature_K, pressure_Pa, *, buffer):
    """Evaluate the published Fe-FeO-H2O-buffer regression at its named host.

    Pressure is total experimental pressure, not H2 fugacity. The output
    uncertainty propagates the reported slope uncertainty only; it does not
    represent the scatter of every observation or the uncertainty of BSE
    transfer. The concentration includes the complete glass mass, including
    native dissolved water. No new water-solubility correction is supplied.
    """
    if buffer != "Fe-FeO-H2O":
        raise ValueError("This pressure fit requires the experimental Fe-FeO-H2O buffer.")
    data = json.loads(DATA_PATH.read_text())["chaudhari_2025"]
    if host not in data["hosts"]:
        raise ValueError("Select a published Fe-free experimental host; BSE is unsupported.")
    model = data["hosts"][host]
    slope = model["slope_H2_mass_ppm_per_GPa"]
    uncertainty = model["slope_reported_uncertainty_H2_mass_ppm_per_GPa"]
    pairs = [(row["temperature_K"], row["pressure_Pa"]) for row in data["observations"]
             if row["host"] == host and not row["excluded_from_further_analysis_by_authors"]]
    if host == "haplogranite":
        supported = (temperature_K, pressure_Pa) in pairs
    else:
        supported = temperature_K == 1673.15 and min(p for _, p in pairs) <= pressure_Pa <= max(p for _, p in pairs)
    if not np.isfinite(temperature_K) or not np.isfinite(pressure_Pa) or not supported:
        raise ValueError("Require the measured host/T/P conditions; only fixed-temperature pressure interpolation is supported.")
    return {"molecular_h2_mass_ppm": slope * pressure_Pa / 1e9,
            "slope_uncertainty_ppm": uncertainty * pressure_Pa / 1e9,
            "concentration_basis": "H2 mass / total glass mass",
            "evidence": "Published buffered-pressure regression, not an arbitrary-fugacity Henry standard.",
            "source_doi": "10.1007/s00410-025-02272-y"}


def kato1970_hydrogen_mass_ppm(temperature_K, *, hydrogen_partial_pressure_Pa=ATM_PA,
                             silicon_mass_percent=0.):
    """Return dilute atomic-H mass ppm in pure Fe or low-Si Fe-Si liquid.

    The operational domain is restricted to the one-atmosphere-H2 reference.
    Total liquid pressure, O-containing alloys and high H fractions are not
    inferred from this law. The pure-Fe and Si-effect fits overlap only at
    1843.15--1873.15 K. The Si variable is mass percent, not atomic fraction.
    """
    if (not np.isfinite(temperature_K) or not np.isfinite(hydrogen_partial_pressure_Pa)
            or hydrogen_partial_pressure_Pa != ATM_PA
            or not np.isfinite(silicon_mass_percent) or not 0 <= silicon_mass_percent < 2.5
            or not 1843.15 <= temperature_K <= (2013.15 if silicon_mass_percent == 0 else 1873.15)):
        raise ValueError("Require the measured Fe/Fe-Si temperature/composition domain and p_H2=101325 Pa.")
    return float(1e4 * 10**(-1874. / temperature_K - 1.601 - .033 * silicon_mass_percent))


def kato_atomic_h_standard_RT(temperature_K, gas_h2_standard_RT, *, standard_pressure_Pa=1e5):
    """Convert the pure-Fe mass-basis equilibrium to an atomic-H standard.

    Supply the molecular gas H2 mu0/(RT) at the declared ideal-gas p0. The
    returned mass-fraction standard obeys mu_H = mu0_H + ln(w_H), with the
    atom-balanced exchange 0.5 H2(g) = H(metal). This is an alternative to
    the inherited high-pressure H standard, never an additive correction.
    Its calibration applies only to the pure-Fe dilute reference above.
    """
    w = kato1970_hydrogen_mass_ppm(temperature_K) * 1e-6
    if (not np.isfinite(gas_h2_standard_RT) or not np.isfinite(standard_pressure_Pa)
            or standard_pressure_Pa <= 0):
        raise ValueError("Supply a finite gas standard and a positive standard pressure in Pa.")
    return float(.5 * gas_h2_standard_RT - np.log(w) + .5 * np.log(ATM_PA / standard_pressure_Pa))


def reference_report():
    """Preserve regression residuals instead of declaring every datum accepted."""
    data = json.loads(DATA_PATH.read_text())
    records = []
    for row in data["chaudhari_2025"]["observations"]:
        if row["excluded_from_further_analysis_by_authors"]:
            continue
        result = chaudhari_buffered_h2_mass_ppm(row["host"], row["temperature_K"],
                                               row["pressure_Pa"], buffer="Fe-FeO-H2O")
        predicted = result["molecular_h2_mass_ppm"]
        records.append({**row, "predicted_H2_mass_ppm": predicted,
                        "residual_ppm": predicted - row["H2_mass_ppm"],
                        "residual_over_observation_sigma": (predicted - row["H2_mass_ppm"]) / row["H2_mass_ppm_one_sigma"]})
    iron = kato1970_hydrogen_mass_ppm(1873.15)
    # This separate measured pressure check does not extend the public reference domain.
    reduced = iron / np.sqrt(11.)
    return {"source_data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "report_kind": "source_regression_residuals",
            "chaudhari_regression_residuals": records,
            "author_excluded_run_ids": [row["run_id"] for row in data["chaudhari_2025"]["observations"]
                                        if row["excluded_from_further_analysis_by_authors"]],
            "kato_pure_iron": {"temperature_K": 1873.15, "p_H2_Pa": ATM_PA,
                               "predicted_atomic_H_mass_ppm": iron, "observed_atomic_H_mass_ppm": 25.0,
                               "residual_ppm": iron - 25.0},
            "kato_low_partial_pressure_receipt": {
                "temperature_K": 1873.15, "p_H2_Pa": ATM_PA / 11.,
                "total_pressure_Pa": ATM_PA,
                "predicted_atomic_H_mass_ppm": reduced, "observed_mean_ppm": 7.34,
                "reported_sigma_ppm": .18, "residual_over_reported_sigma": (reduced - 7.34) / .18,
                "scope": "Separate measured condition; not a validated T/P rectangle."},
            "accepted_coupled_material_domain": None,
            "scientific_acceptance": {"M2_A": "pending", "M2_B": "pending"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reference_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
