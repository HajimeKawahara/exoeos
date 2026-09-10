"""Regenerate external MELTS references with a pinned, separately installed runtime.

This script is not a native ExoEOS model. Each equilibrium runs in a fresh
process and temporary directory because the external C library has global
state and writes files. No backend download or installation is performed.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


REFERENCE = Path(__file__).with_name("melts_silicate_v1.json")
COMMIT = "9b92b1e0e6d1538a0f498572d6c6651d5d3aca7e"
SOURCE = "https://github.com/magmasource/alphaMELTS"
ARCHIVE = "alphamelts-py-2.3.2-ubuntu_22_04-x86_64.zip"
ARCHIVE_SHA256 = "97ec2cdb53cae69822a41b5639d8b93edf95b2c361bc15e64e53b192ea9e1425"
R = 8.3143  # J mol^-1 K^-1, the external model's value.
RUNTIME_HASHES = {
    "meltsdynamic.py": "c9b12a82ee2d8dfc317af33e571e441158e0d4422af5bcdc11cf2197dbb71ce8",
    "meltsengine.py": "05a41c1f842e6359fd44541638cb34389258a7539c47e5f205090434f6082af9",
    "meltsstatus.py": "4869341cdd18e8eac01a4fe612e14bc350ea0f8ecdec7fa498a7f0e8e65e99ee",
    "libalphamelts.so": "c218ef6f7ba5aef4b0760f3176530d1335457d6e2716823a1d0de3a5d13301e5",
    "libgsl.so.28": "62dcb82585a2970edf2054e8e7c7df1a1aa4c307c936e7ff95b1331a4ababb2e",
    "libgslcblas.so.0": "bb1d1273d2596ff756cbc881bc8a51837ed9cfbfa59ed55309ae42e3d9496948",
}
OXIDES = (
    "sio2", "tio2", "al2o3", "fe2o3", "cr2o3", "feo", "mno", "mgo",
    "nio", "coo", "cao", "na2o", "k2o", "p2o5", "h2o", "co2", "so3",
    "cl2o-1", "f2o-1",
)
# Rows are liquid components; columns in the fixture use OXIDES order.
COMPONENTS = {
    "sio2": {"sio2": 1}, "tio2": {"tio2": 1},
    "al2o3": {"al2o3": 1}, "fe2o3": {"fe2o3": 1},
    "mgcr2o4": {"mgo": 1, "cr2o3": 1},
    "fe2sio4": {"feo": 2, "sio2": 1},
    "mnsi0.5o2": {"mno": 1, "sio2": 0.5},
    "mg2sio4": {"mgo": 2, "sio2": 1},
    "nisi0.5o2": {"nio": 1, "sio2": 0.5},
    "cosi0.5o2": {"coo": 1, "sio2": 0.5},
    "casio3": {"cao": 1, "sio2": 1},
    "na2sio3": {"na2o": 1, "sio2": 1},
    "kalsio4": {"k2o": 0.5, "al2o3": 0.5, "sio2": 1},
    "ca3(po4)2": {"cao": 3, "p2o5": 1},
    "co2": {"co2": 1}, "so3": {"so3": 1},
    "cl2o-1": {"cl2o-1": 1}, "f2o-1": {"f2o-1": 1}, "h2o": {"h2o": 1},
}
# Original tutorial amounts, in grams: sum = 100.35, not assumed 100 wt%.
BULK_GRAMS = (
    48.68, 1.01, 17.64, 0.89, 0.03, 7.59, 0, 9.10, 0, 0,
    12.45, 2.65, 0.03, 0.08, 0.2, 0, 0, 0, 0,
)
CASES = {
    "morb_1473K_50MPa": (1473.15, 5e7),
    "morb_1423K_50MPa": (1423.15, 5e7),
    "morb_1473K_500MPa": (1473.15, 5e8),
}


def stoichiometry():
    return np.asarray([[row.get(oxide, 0) for oxide in OXIDES] for row in COMPONENTS.values()])


def check_runtime(runtime):
    for name, expected in RUNTIME_HASHES.items():
        if hashlib.sha256((runtime / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Runtime hash mismatch: {name}")


def checked_properties(values, present):
    """Keep unavailable endpoints null; require real finite present values."""
    result = []
    for value, available in zip(values, present):
        if available and not math.isfinite(value):
            raise ValueError("Nonfinite property for a present liquid component.")
        result.append(float(value) if available else None)
    return result


def evaluate_case(runtime, case):
    # Import only inside the external worker, never during ordinary tests.
    sys.path.insert(0, str(runtime))
    from meltsdynamic import MELTSdynamic

    model = MELTSdynamic(1)  # rhyolite-MELTS 1.0.2
    engine = model.engine
    assert model.endMemberFormulas["bulk"] == list(OXIDES)
    assert model.endMemberFormulas["liquid"] == list(COMPONENTS)
    temperature, pressure = CASES[case]
    engine.temperature = temperature - 273.15  # K -> Celsius
    engine.pressure = pressure / 1e5  # Pa -> bar (not ThermoEngine's MPa).
    engine.setBulkComposition(list(BULK_GRAMS))
    engine.setSystemProperties("Log fO2 Path", "None")
    engine.calcEquilibriumState(1, 0)  # Isothermal/isobaric, no fractionation.
    if engine.status.failed or engine.liquidNames != ["liquid1"]:
        raise RuntimeError(f"Unsupported equilibrium result: {engine.status.message}")
    state = {
        "id": case, "T_K": temperature, "P_Pa": pressure,
        "backend_T_C": engine.temperature, "backend_P_bar": engine.pressure,
        "solver_message": engine.status.message,
        "log10_fO2": engine.logfO2,
        "bulk_mass_g": engine.mass["bulk"],
        "phases": [{
            "name": name, "mass_g": engine.mass[name], "gibbs_J": engine.g[name],
            "oxide_wt_percent": list(engine.dispComposition[name]),
        } for name in engine.liquidNames + engine.solidNames],
    }
    liquid_phase = state["phases"][0]
    state["liquid_mass_fraction"] = liquid_phase["mass_g"] / state["bulk_mass_g"]
    engine.calcEndMemberProperties("liquid1", liquid_phase["oxide_wt_percent"])
    if engine.status.failed:
        raise RuntimeError("Liquid endmember calculation failed.")
    x = np.asarray(list(engine.X["liquid1"]))
    present = x > 0
    if np.any(x < 0) or not np.isclose(x.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("Unsupported liquid composition.")
    mu = np.asarray(list(engine.mu["liquid1"]))
    mu0 = np.asarray(list(engine.mu0["liquid1"]))
    activity = np.asarray(list(engine.activity["liquid1"]))
    oxide_weights = np.asarray(engine.status.molwts["bulk"])
    oxide_moles = liquid_phase["mass_g"] * np.asarray(liquid_phase["oxide_wt_percent"]) / 100 / oxide_weights
    matrix = stoichiometry()
    component_moles = np.linalg.solve(matrix.T, oxide_moles)
    active_oxides = oxide_moles > 0
    # Transform only the defined square subspace. Do not replace absent mu by zero.
    oxide_mu = np.linalg.solve(matrix[np.ix_(present, active_oxides)], mu[present])
    ln_a = (mu[present] - mu0[present]) / (R * temperature)
    ln_gamma = ln_a - np.log(x[present])
    total_moles = component_moles.sum()
    gex_RT = (
        (liquid_phase["gibbs_J"] / total_moles - x[present] @ mu0[present]) / (R * temperature)
        - x[present] @ np.log(x[present])
    )
    log_a_all = np.full(len(x), np.nan)
    log_gamma_all = log_a_all.copy()
    log_a_all[present], log_gamma_all[present] = ln_a, ln_gamma
    oxide_mu_all = np.full(len(OXIDES), np.nan)
    oxide_mu_all[active_oxides] = oxide_mu
    state["liquid"] = {
        "phase": "liquid1", "component_moles": component_moles.tolist(),
        "x": x.tolist(), "mu_J_mol": checked_properties(mu, present),
        "mu0_J_mol": checked_properties(mu0, present),
        "activity": checked_properties(activity, present),
        "ln_activity": checked_properties(log_a_all, present),
        "ln_gamma": checked_properties(log_gamma_all, present),
        "gex_RT": float(gex_RT),
        "oxide_mu_J_mol": checked_properties(oxide_mu_all, active_oxides),
    }
    return {
        "backend_version": model.version,
        "phase_database": engine.status.phases,
        "oxide_molar_masses_g_mol": oxide_weights.tolist(), "state": state,
    }


def generate(runtime):
    results = []
    for case in CASES:
        with tempfile.TemporaryDirectory(prefix="exoeos-melts-") as directory:
            output = Path(directory) / "state.json"
            env = dict(os.environ)
            env["LD_LIBRARY_PATH"] = str(runtime) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--runtime", str(runtime),
                 "--case", case, "--output", str(output)],
                cwd=directory, env=env, capture_output=True, text=True, timeout=60,
            )
            if process.returncode:
                raise RuntimeError(process.stdout[-4000:] + process.stderr[-4000:])
            log = process.stdout + process.stderr
            if "Iteration exceeded" in log or "convergence was acceptable" in log:
                raise RuntimeError(f"Unexpected solver fallback in {case}.")
            results.append(json.loads(output.read_text()))
    first = results[0]
    for result in results[1:]:
        for key in ("backend_version", "phase_database", "oxide_molar_masses_g_mol"):
            assert result[key] == first[key]
    return {
        "model_id": "alphamelts_2_3_2_rhyolite_melts_1_0_2_morb_v1",
        "status": "external_reference_only_no_native_silicate_model",
        "backend": {
            "model": "rhyolite-MELTS 1.0.2", "calculation_mode": 1,
            "version_string": first["backend_version"], "release": "v2.3.2",
            "source_commit": COMMIT, "source_url": f"{SOURCE}/tree/{COMMIT}",
            "archive_url": f"{SOURCE}/releases/download/v2.3.2/{ARCHIVE}",
            "archive_sha256": ARCHIVE_SHA256, "runtime_sha256": RUNTIME_HASHES,
            "license": "AGPL-3.0; backend is not redistributed",
            "build_provenance": "Release binary pinned by hash; thermodynamic C build commit is not independently established.",
        },
        "composition_source": {
            "url": f"{SOURCE}/blob/{COMMIT}/examples/tutorial/py/tutorial.py",
            "sha256": "9f44a4fdde489709a42efce93ffc88b814506e84a956c1cd5cd7fa813168cd28",
            "description": "Official MORB tutorial amounts, including 0.2 g H2O; original 100.35 g total retained.",
        },
        "R_J_mol_K": R, "oxide_order": list(OXIDES),
        "component_order": list(COMPONENTS),
        "oxide_molar_masses_g_mol": first["oxide_molar_masses_g_mol"],
        "component_oxide_stoichiometry": COMPONENTS,
        "bulk_oxide_mass_g": list(BULK_GRAMS),
        "phase_policy": {
            "database_phase_names": first["phase_database"],
            "explicitly_suppressed_phases": [], "selection": "Backend default phase policy; reported equilibrium assemblage.",
            "run_mode": 1, "output_flag": 0, "fractionation": False,
            "oxygen_buffer": "None; closed input oxygen, reported log10_fO2 is an output.",
        },
        "standard_state": {
            "basis": "MELTS liquid endmember mole fractions; pure endmember at the same T/P",
            "activity_field": "activity (not activity0)",
            "ln_gamma": "(mu - mu0)/(R*T) - ln(x)",
            "oxide_conversion": "n_oxide = nu.T @ n_component; mu_component = nu @ mu_oxide",
            "oxide_standard_warning": "Linear transformation of component mu0 is not a pure-oxide standard. No oxide activity coefficients are supplied.",
            "missing_values": "Absent components have x=0 and unavailable properties are null; no trace additions.",
            "consumer_warning": "Full mu already includes mixing. Common elemental/standard references with other phases must be established separately.",
        },
        "scope": "Calculated equation references in a hydrous basaltic host, not measured activities, a calibration box, or a coupled melt-metal equilibrium.",
        "reproduction_tolerances": {"rtol": 5e-9, "atol": 5e-9},
        "states": [result["state"] for result in results],
    }


def compare(actual, expected, path="reference"):
    """Compare structure, missing-value masks, metadata, and numeric outputs."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{path}[{index}]")
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=5e-9, abs_tol=5e-9), path
    else:
        assert actual == expected, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True, help="Extracted pinned Linux x86_64 Python release directory.")
    parser.add_argument("--output", type=Path, default=REFERENCE)
    parser.add_argument("--check", action="store_true", help="Regenerate and compare without writing the fixture.")
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    check_runtime(runtime)
    result = evaluate_case(runtime, args.case) if args.case else generate(runtime)
    if args.check:
        compare(result, json.loads(args.output.read_text()))
        print("MELTS silicate reference is reproducible.")
    else:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
