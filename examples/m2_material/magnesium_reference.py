"""Reconstruct the measured MgO/Fe-Mg-O reference without a BSE alloy fit.

The published Wagner activity expansion is replayed as reported. It is not
installed as a new excess-Gibbs model or extended to high-O FeSiOH silently.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .oxygen_calibration import _finite_scalar


DATA_DIR = Path(__file__).with_name("steel_data")
DATA_PATH = DATA_DIR / "Itoh1997_Table2.csv"
JANAF_PATH = DATA_DIR / "JANAF_Mg-008.txt"
PROVENANCE_PATH = DATA_DIR / "magnesium_provenance.json"
R = 8.31446261815324
LN10 = np.log(10.)


def mgo_crystal_standard_rt(temperature_K):
    """JANAF MgO(cr), 1 bar, fixed 298.15-K elemental enthalpy gauge.

    Cubic Hermite interpolation uses G=Hf(298)+[H(T)-H(298)]-T*S(T)
    and endpoint slopes -S. The temperature-dependent formation G column
    changes the elemental reference phases and is deliberately not used.
    No extrapolation outside 1700-2400 K is provided.
    """
    t = _finite_scalar(temperature_K, "T_K", positive=True)
    if not 1700. <= t <= 2400.:
        raise ValueError("JANAF interpolation is restricted to 1700-2400 K.")
    rows = [list(map(float, line.split()[:5])) for line in JANAF_PATH.read_text().splitlines()[2:]
            if line.split() and 1700. <= float(line.split()[0]) <= 2400.]
    data = np.array(rows)
    grid, entropy = data[:, 0], data[:, 2]
    gibbs = 1000.*(-601.241+data[:, 4])-grid*entropy
    index = min(int(np.searchsorted(grid, t, side="right")-1), len(grid)-2)
    width = grid[index+1]-grid[index]
    z = (t-grid[index])/width
    h00, h10 = 2*z**3-3*z*z+1, z**3-2*z*z+z
    h01, h11 = -2*z**3+3*z*z, z**3-z*z
    g = h00*gibbs[index]-h10*width*entropy[index]+h01*gibbs[index+1]-h11*width*entropy[index+1]
    return float(g/(R*t))


def magnesium_henry_standard(temperature_K, *, oxygen_henry_standard_rt,
                             mgo_crystal_standard_rt, allow_temperature_extrapolation=False):
    """Convert MgO(cr)=Mg+O to a supplied common Henry mass-percent gauge.

    Both solutes have f->1 in pure liquid Fe. Supply O and crystalline MgO
    standards on the same RT and elemental gauge. Pressure dependence and
    finite FeSiOH activity are not inferred from this reaction standard.
    """
    t = _finite_scalar(temperature_K, "T_K", positive=True)
    o = _finite_scalar(oxygen_henry_standard_rt, "oxygen_henry_standard_rt")
    mgo = _finite_scalar(mgo_crystal_standard_rt, "mgo_crystal_standard_rt")
    if not isinstance(allow_temperature_extrapolation, bool):
        raise ValueError("allow_temperature_extrapolation must be bool.")
    inside = 1873. <= t <= 2023.
    if not inside and not allow_temperature_extrapolation:
        raise ValueError("Temperature extrapolation requires explicit opt-in.")
    log_k = -4.28-4700./t
    mu = mgo-o-LN10*log_k
    return {"reaction": "MgO(cr) = Mg(metal) + O(metal)", "T_K": t,
            "log10_K": log_k, "Mg_Henry_mass_percent_standard_rt": float(mu),
            "Mg_infinite_dilution_mole_fraction_standard_rt": float(mu+np.log(100.*24.305/55.845)),
            "oxygen_Henry_mass_percent_standard_rt": o, "MgO_crystal_standard_rt": mgo,
            "measured_temperatures_K": [1873., 2023.],
            "inside_measured_temperature_envelope": inside,
            "temperature_distance_K": max(1873.-t, t-2023., 0.),
            "pressure_dependence": "Uncalibrated; atmospheric laboratory reference, not a high-pressure law.",
            "finite_FeSiOH_activity_supplied": False,
            "standard_parameter_covariance": None,
            "composition_transfer_error_bound": None, "temperature_transfer_error_bound": None,
            "accepted_coupled_material_domain": None}


def itoh_activity_reference(temperature_K, magnesium_mass_percent, oxygen_mass_percent, *,
                            allow_extrapolation=False):
    """Evaluate the published finite-solute expansion as a reference only.

    Coordinate envelopes describe observed extrema, not a joint validated
    rectangle. Pure-Fe mass percentages exclude unmodeled Si/H solutes.
    The expansion is not supplied as an integrable five-component alloy G.
    """
    t, mg, o = [_finite_scalar(value, name, positive=True) for value, name in
                ((temperature_K, "T_K"), (magnesium_mass_percent, "Mg_mass_percent"),
                 (oxygen_mass_percent, "O_mass_percent"))]
    if mg+o >= 100. or not isinstance(allow_extrapolation, bool):
        raise ValueError("Require a positive Fe balance and boolean extrapolation flag.")
    limits = ((1873., 2023.), (.00002, .00307), (.00034, .00428))
    distances = [max(lo-v, v-hi, 0.) for v, (lo, hi) in zip((t, mg, o), limits)]
    # Unit conversion of a printed endpoint can differ by one floating ULP.
    distances = [0. if delta <= 8*np.finfo(float).eps*max(abs(v), abs(lo), abs(hi)) else delta
                 for delta, v, (lo, hi) in zip(distances, (t, mg, o), limits)]
    if any(distances) and not allow_extrapolation:
        raise ValueError("Extrapolation beyond observed coordinate extrema requires explicit opt-in.")
    e_omg, e_mgo = 630.-1.71e6/t, 958.-2.59e6/t
    r_mgo, r_omg = -1.90e6+4.22e9/t, 70500.-1.70e8/t
    r_omgo, r_mgmgo = -2.51e6+5.57e9/t, 2.14e5-5.16e8/t
    log_f_mg = e_mgo*o+r_mgo*o*o+r_mgmgo*mg*o
    log_f_o = (-1750./t+.76)*o+e_omg*mg+r_omg*mg*mg+r_omgo*mg*o
    return {"T_K": t, "Mg_mass_percent": mg, "O_mass_percent": o,
            "log10_f_Mg": log_f_mg, "log10_f_O": log_f_o,
            "log10_activity_product": float(np.log10(mg*o)+log_f_mg+log_f_o),
            "log10_K": -4.28-4700./t,
            "coordinate_distances_T_K_Mg_wt_percent_O_wt_percent": distances,
            "inside_observed_coordinate_envelope": not any(distances),
            "joint_composition_domain_established": False,
            "accepted_coupled_material_domain": None}


def reference_report():
    provenance = json.loads(PROVENANCE_PATH.read_text())
    for path in (DATA_PATH, JANAF_PATH):
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["data_sha256"][path.name]:
            raise ValueError("Mg reference data hash mismatch.")
    with DATA_PATH.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    results = []
    for row in rows:
        state = itoh_activity_reference(float(row["temperature_K"]),
            float(row["Mg_mass_percent_times_1e4"])/1e4, float(row["O_mass_percent_times_1e4"])/1e4)
        results.append({"run_id": int(row["run_id"]), **state,
                        "reaction_residual_log10_K": state["log10_activity_product"]-state["log10_K"]})
    summaries = {}
    for temperature in (1873., 2023.):
        residual = np.array([row["reaction_residual_log10_K"] for row in results if row["T_K"] == temperature])
        summaries[str(temperature)] = {"sample_count": len(residual),
            "rmse_log10_K": float(np.sqrt(np.mean(residual**2))),
            "max_absolute_residual_log10_K": float(np.max(np.abs(residual)))}
    return {"report_kind": "itoh1997_deoxidation_reference_v1", "observed_sample_count": len(rows),
            "summary": summaries, "observations": results,
            "refitted_parameter_count": 0,
            "interpretation": "Published second-order coefficients are replayed without refitting; substantial raw-point residuals remain. These residuals are not a confidence interval or transfer error bound. No high-O FeSiOH constitutive model is installed.",
            "provenance_sha256": hashlib.sha256(PROVENANCE_PATH.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "data_sha256": provenance["data_sha256"],
            "accepted_coupled_material_domain": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reference_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
