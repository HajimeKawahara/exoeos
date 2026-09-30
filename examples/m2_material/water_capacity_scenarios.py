"""Finite capacity scenarios covering recorded central reference residuals."""

import hashlib
import json
import math
from pathlib import Path

from .water_calibration import water_calibration_report

REFERENCE_PATH = Path(__file__).with_name("validation") / "20260928_water_reconstruction/pressure_zhang_duan.json"


def capacity_scenario_report():
    records = []
    for row in water_calibration_report()["observations"]:
        if row["used_in_author_fit"]:
            records.append({"sample": row["sample"], "study": row["study"],
                "role": "original_author_fit_point", "observed": row["observed_OH_mass_ppm"],
                "sigma": row["observed_sigma_ppm"], "predicted": row["predicted_OH_mass_ppm"],
                "units": "author_response_mass_ppm"})
    for row in json.loads(REFERENCE_PATH.read_text())["observations"]:
        records.append({"sample": row["sample"], "study": "Shishkina2010", "role": "independent_reference",
            "observed": row["KFT_H2O_wt_percent"], "sigma": row["KFT_sigma_wt_percent"],
            "predicted": row["predicted_H2O_equivalent_wt_percent"], "units": "H2O_equivalent_wt_percent"})
    for row in records:
        p, o, s = row["predicted"], row["observed"], row["sigma"]
        row["central_cover_factor"] = max(p/o, o/p)
        row["reported_concentration_1sigma_cover_factor"] = max(p/(o-s), (o+s)/p) if o > s else None
    factor = 2.265
    central = max(records, key=lambda r: r["central_cover_factor"])
    if central["central_cover_factor"] > factor:
        raise ValueError("The declared capacity interval does not cover the central reference residuals.")
    sigma = max((r for r in records if r["reported_concentration_1sigma_cover_factor"] is not None),
                key=lambda r: r["reported_concentration_1sigma_cover_factor"])
    offset = 2*math.log(factor)
    return {"record_id": "reference_spread_water_capacity_scenarios_v1",
        "scope": "Finite capacity scenarios covering retained central measurements; not a statistical confidence band or a bound on BSE extrapolation.",
        "factor": factor, "capacity_interval": [1/factor, factor],
        "standard_shift_key": "h2o_melts", "standard_shift_RT": [-offset, offset],
        "scenarios": {"lower_capacity": {"standard_offsets_rt": {"h2o_melts": offset}},
                      "upper_capacity": {"standard_offsets_rt": {"h2o_melts": -offset}}},
        "central_max": central, "reported_concentration_1sigma_max": sigma,
        "covered_observation_count": len(records), "original_author_fit_count": 74, "independent_KFT_count": 14,
        "observations": records, "reference_artifact_sha256": hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest(),
        "limitations": [
            "The central factor does not cover every reported 1sigma concentration interval.",
            "The separate 1sigma diagnostic varies only measured concentration; it does not propagate fugacity, fluid composition, temperature, density or common absorptivity errors.",
            "Fit observations are not independent validation; methods and units remain explicit.",
            "Applying the range at2173.15K/270bar is a declared finite model comparison, not a universal empirical envelope.",
            "No gas standard is changed; adding h*offset to one scalar retains host curvature."]}


if __name__ == "__main__":
    print(json.dumps(capacity_scenario_report(), indent=2))
