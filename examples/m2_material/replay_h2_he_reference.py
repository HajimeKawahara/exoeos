"""Replay the published He/H2 EOS without changing the M2 gas model.

Requires the optional, isolated reference dependency teqp==0.23.1.
The author-supplied mixture parameters are evaluated by teqp, not reimplemented.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


def replay(closure_path):
    import teqp

    source_path = Path(__file__).with_name("h2_he_reference_sources.json")
    source = json.loads(source_path.read_text())
    if teqp.__version__ != source["teqp_version"]:
        raise ValueError("Use the pinned reference teqp version.")
    model = teqp.make_model({"kind": "multifluid", "model": {
        "components": source["component_order"], "root": teqp.get_datapath(),
        "BIP": source["BIP"], "departure": source["departure"]}})
    z = np.array([1-source["table_6_hydrogen_fraction"], source["table_6_hydrogen_fraction"]])
    checks = []
    for row in source["table_6_pressure"]:
        t, rho = row["temperature_K"], row["density_mol_m3"]
        p = rho*model.get_R(z)*t*(1+model.get_Ar01(t, rho, z))/1e6
        error = p/row["pressure_MPa"]-1
        checks.append({**row, "calculated_pressure_MPa": p, "relative_error": error,
                       "within_tolerance": abs(error) <= source["implementation_relative_tolerance"]})
    raw = Path(closure_path).read_bytes()
    closure = json.loads(raw)
    roots = [root for run in closure["runs"] for root in run.get("roots", [])]
    if not closure["numerically_accepted"] or len(roots) != 1:
        raise ValueError("Supply a preserved, accepted closure with one root.")
    state = roots[0]
    parcel = state["basal_parcel"]
    amounts = np.array(parcel["gas_amounts_mol"])
    original = amounts[[parcel["gas_species"].index(name) for name in ("He1", "H2")]]/amounts.sum()
    z = original/original.sum()
    t, p = state["temperature_base_k"], state["pressure_base_pa"]
    rho_ideal = p/(model.get_R(z)*t)
    rho = brentq(lambda r: r*model.get_R(z)*t*(1+model.get_Ar01(t, r, z))-p,
                 rho_ideal/10, rho_ideal*10)
    virials = [{"temperature_K": temp, "helium_fraction": he,
                "apparent_cross_virial_cm3_mol": model.get_B12vir(temp, np.array([he, 1-he]))*1e6}
               for temp in (1000., 1500., 2000., t) for he in (.1, float(z[0]), .5, .9)]
    paths = [Path(__file__), source_path]
    paths.extend(Path(teqp.get_datapath())/"dev/fluids"/(name+".json") for name in source["component_order"])
    paths.extend(Path(teqp.__file__).parent.glob("*.so"))
    return {
        "assessment_id": "beckmueller2024_helium_hydrogen_replay_v1", "teqp_version": teqp.__version__,
        "source_doi": source["doi"], "implementation_checks": checks,
        "all_pressure_reference_checks_passed": all(row["within_tolerance"] for row in checks),
        "original_closure_sha256": hashlib.sha256(raw).hexdigest(),
        "conditioned_binary": {"temperature_K": t, "pressure_Pa": p,
            "original_He_H2_fractions": original.tolist(), "omitted_original_fraction": 1-float(original.sum()),
            "He_H2_fractions": z.tolist(), "density_mol_m3": rho,
            "compressibility": 1+model.get_Ar01(t, rho, z),
            "second_virial_compressibility_at_same_density": 1+model.get_B2vir(t, z)*rho,
            "density_virials_SI": model.get_Bnvir(4, t, z)},
        "apparent_cross_virials": virials,
        "file_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "adopted_in_m2_provider": False,
        "scope": "Author EOS replay at a conditioned binary composition and the full saved pressure. "
            "The apparent cross virial varies with composition under the reducing function; it is not "
            "silently inserted as a unique pair coefficient. The saved temperature exceeds the 2000 K "
            "cross-virial reference data. No wet-gas closure or empirical error bound is established.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output; previous replays are preserved.")
    report = replay(args.closure)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
