"""Optional supplied-composition MELTS properties; no equilibrium or JAX solver.

Run with --help. The source checkout's existing reference supplies the pinned
runtime hashes and chemical basis, never an interpolation of its three states.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np


REFERENCE_PATH = Path(__file__).resolve().parents[1] / "tests/reference/melts_silicate_v1.json"
REFERENCE = json.loads(REFERENCE_PATH.read_text())
COMPONENTS = REFERENCE["component_order"]
OXIDES = REFERENCE["oxide_order"]
NU = np.asarray([
    [REFERENCE["component_oxide_stoichiometry"][component].get(oxide, 0) for oxide in OXIDES]
    for component in COMPONENTS
])
OXIDE_MASSES = np.asarray(REFERENCE["oxide_molar_masses_g_mol"])
BACKEND_R = REFERENCE["R_J_mol_K"]
COMMON_R = 8.31446261815324  # J mol^-1 K^-1; caller can select its common R.
MODEL_ID = "alphamelts_2_3_2_rhyolite_melts_1_0_2_supplied_liquid_v1"
# Explicit elemental ledger; the formal halogen oxides have negative oxygen.
OXIDE_FORMULAS = [
    {"Si": 1, "O": 2}, {"Ti": 1, "O": 2}, {"Al": 2, "O": 3},
    {"Fe": 2, "O": 3}, {"Cr": 2, "O": 3}, {"Fe": 1, "O": 1},
    {"Mn": 1, "O": 1}, {"Mg": 1, "O": 1}, {"Ni": 1, "O": 1},
    {"Co": 1, "O": 1}, {"Ca": 1, "O": 1}, {"Na": 2, "O": 1},
    {"K": 2, "O": 1}, {"P": 2, "O": 5}, {"H": 2, "O": 1},
    {"C": 1, "O": 2}, {"S": 1, "O": 3}, {"Cl": 2, "O": -1},
    {"F": 2, "O": -1},
]
ELEMENTS = sorted(set().union(*OXIDE_FORMULAS))
OXIDE_ELEMENTS = np.asarray([[formula.get(element, 0) for element in ELEMENTS] for formula in OXIDE_FORMULAS])
FORMULA_MATRIX = NU @ OXIDE_ELEMENTS  # component rows, element columns


def check_runtime(runtime):
    """Reject a changed external wrapper/library before any backend import."""
    for name, expected in REFERENCE["backend"]["runtime_sha256"].items():
        if hashlib.sha256((runtime / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Runtime hash mismatch: {name}")


def validate_request(temperature, pressure, component_moles, common_R):
    n = np.asarray(component_moles, dtype=float)
    if n.shape != (len(COMPONENTS),) or not np.all(np.isfinite(n)) or np.any(n < 0):
        raise ValueError("Supply 19 finite nonnegative endmember amounts in component_order.")
    if not np.isfinite(n.sum()) or n.sum() <= 0:
        raise ValueError("The liquid must have a finite positive total amount.")
    if not np.isfinite(temperature) or temperature <= 0 or temperature == 273.15:
        raise ValueError("T_K must be positive and finite; the wrapper cannot accept exactly 0 Celsius.")
    if not np.isfinite(pressure) or pressure <= 0 or not np.isfinite(common_R) or common_R <= 0:
        raise ValueError("P_Pa and common_R_J_mol_K must be positive and finite.")
    grams = (NU.T @ n) * OXIDE_MASSES
    if not np.all(np.isfinite(grams)) or not np.isfinite(grams.sum()) or grams.sum() <= 0:
        raise ValueError("Converted oxide masses must be finite with a positive total.")
    return n, grams


def require_match(actual, expected, label, rtol=5e-9, atol=0):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError(f"Backend changed or failed to reproduce {label}.")


def nullable(values, present):
    array = np.asarray(values, dtype=float)
    if array.shape != present.shape or not np.all(np.isfinite(array[present])):
        raise ValueError("Unavailable or nonfinite property for a present component.")
    return [float(value) if exists else None for value, exists in zip(array, present)]


def source_provenance():
    """Record checkout identity and changed tracked content without importing ExoEOS."""
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    changed = subprocess.run(["git", "diff", "--name-only", "HEAD", "-z"], cwd=root, capture_output=True)
    hashes = {}
    if changed.returncode == 0:
        for name in changed.stdout.decode().split("\0"):
            if name:
                path = root / name
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {"exoeos_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "changed_tracked_file_sha256": hashes, "worker_command": sys.argv}


def _worker(runtime, request):
    """Only called in a new process and temporary working directory."""
    check_runtime(runtime)
    temperature, pressure = request["T_K"], request["P_Pa"]
    common_R = request["common_R_J_mol_K"]
    n, grams = validate_request(temperature, pressure, request["component_moles"], common_R)
    sys.path.insert(0, str(runtime))
    from meltsdynamic import MELTSdynamic

    model = MELTSdynamic(1)
    engine = model.engine
    if model.endMemberFormulas["bulk"] != OXIDES or model.endMemberFormulas["liquid"] != COMPONENTS:
        raise ValueError("Unexpected MELTS chemical basis.")
    require_match(engine.status.molwts["bulk"], OXIDE_MASSES, "oxide molar masses", rtol=0)
    engine.temperature, engine.pressure = temperature - 273.15, pressure / 1e5
    engine.setSystemProperties("Log fO2 Path", "None")
    if engine.status.failed:
        raise RuntimeError("Failed to disable imposed oxygen buffering.")
    engine.calcPhaseProperties("liquid", grams.tolist())
    if engine.status.failed:
        raise RuntimeError("MELTS phase property calculation failed.")
    mass, gibbs = float(engine.mass["liquid"]), float(engine.g["liquid"])
    if not np.isfinite(gibbs) or not np.isfinite(mass) or mass <= 0:
        raise ValueError("Nonfinite Gibbs energy or invalid liquid mass.")
    returned_grams = mass * np.asarray(engine.dispComposition["liquid"]) / 100
    require_match(returned_grams, grams, "phase oxide amounts")
    returned_n = np.linalg.solve(NU.T, returned_grams / OXIDE_MASSES)
    require_match(returned_n, n, "phase endmember amounts")
    engine.calcEndMemberProperties("liquid", grams.tolist())
    if engine.status.failed:
        raise RuntimeError("MELTS endmember property calculation failed.")
    endmember_wt = np.asarray(engine.dispComposition["liquid"])
    require_match(endmember_wt, 100 * grams / grams.sum(), "endmember oxide composition")
    x = np.asarray(engine.X["liquid"])
    require_match(x, n / n.sum(), "endmember mole fractions")
    present = n > 0
    mu, mu0, activity = [np.asarray(getattr(engine, field)["liquid"]) for field in ("mu", "mu0", "activity")]
    if np.any(activity[present] <= 0):
        raise ValueError("Nonpositive activity for a present component.")
    ln_a, ln_gamma = np.full(n.shape, np.nan), np.full(n.shape, np.nan)
    ln_a[present] = (mu[present] - mu0[present]) / (BACKEND_R * temperature)
    ln_gamma[present] = ln_a[present] - np.log(x[present])
    require_match(ln_a[present], np.log(activity[present]), "activity identity", atol=1e-12)
    require_match(n[present] @ mu[present], gibbs, "Euler relation", atol=1e-5)
    # Only a square, full-rank present subspace defines oxide potentials here.
    oxide_n = NU.T @ n
    active_oxides = oxide_n > 0
    subspace = NU[np.ix_(present, active_oxides)]
    oxide_mu = [None] * len(OXIDES)
    oxide_status = "unavailable: present component/oxide subspace is not square and full rank"
    if subspace.shape[0] == subspace.shape[1] and np.linalg.matrix_rank(subspace) == subspace.shape[0]:
        values = np.full(len(OXIDES), np.nan)
        values[active_oxides] = np.linalg.solve(subspace, mu[present])
        oxide_mu = nullable(values, active_oxides)
        oxide_status = "full potentials on the present square subspace; no oxide standards or activities"
    return {
        "model_id": MODEL_ID, "status": "ok_supplied_liquid_properties",
        "T_K": temperature, "P_Pa": pressure,
        "backend_T_C": engine.temperature, "backend_P_bar": engine.pressure,
        "component_order": COMPONENTS, "component_moles": n.tolist(),
        "returned_component_moles": returned_n.tolist(), "x": x.tolist(),
        "oxide_order": OXIDES, "oxide_mass_g": grams.tolist(),
        "returned_oxide_mass_g": returned_grams.tolist(), "mass_g": mass, "gibbs_J": gibbs,
        "mu_J_mol": nullable(mu, present), "mu0_J_mol": nullable(mu0, present),
        "mu_RT": nullable(mu / (common_R * temperature), present),
        "mu0_RT": nullable(mu0 / (common_R * temperature), present),
        "activity": nullable(activity, present), "ln_activity": nullable(ln_a, present),
        "ln_gamma": nullable(ln_gamma, present),
        "ln_activity_common_R": nullable(ln_a * BACKEND_R / common_R, present),
        "ln_gamma_common_R": nullable(ln_a * BACKEND_R / common_R - np.log(np.where(present, x, 1)), present),
        "oxide_mu_J_mol": oxide_mu,
        "oxide_potential_status": oxide_status,
        "basis": {
            "amount_unit": "mol of named MELTS liquid endmember", "energy_unit": "J",
            "component_oxide_matrix": NU.tolist(), "element_order": ELEMENTS,
            "component_element_matrix": FORMULA_MATRIX.tolist(),
            "element_moles": (FORMULA_MATRIX.T @ n).tolist(),
            "standard": "pure liquid endmember at supplied T_K/P_Pa; no separate pressure correction",
            "backend_R_J_mol_K": BACKEND_R, "common_R_J_mol_K": common_R,
            "dimensionless_potential": "mu_RT = mu_J_mol / (common_R_J_mol_K * T_K)",
            "activity_convention": "activity, ln_activity, ln_gamma use backend_R; ln_activity_common_R = (mu - mu0)/(common_R*T_K); ln_gamma_common_R = ln_activity_common_R - ln(x)",
            "mixing": "Full potentials already include ideal and excess mixing; do not add mixing again.",
        },
        "phase_policy": {
            "oxygen_buffer": "None", "equilibrated": False, "stability_checked": False,
            "calibration_domain": "Not established by this property evaluator; three MORB fixtures and local perturbations are equation references only.",
            "cross_phase_standards": "Not aligned to alloy/gas; oxide basis conversion does not establish pure-oxide standards.",
        },
        "provenance": {
            **source_provenance(),
            "backend": REFERENCE["backend"], "backend_version": model.version,
            "backend_message": engine.status.message, "runtime_directory": str(runtime),
            "worker_pid": os.getpid(), "python_executable": sys.executable,
            "python_version": platform.python_version(), "platform": platform.platform(),
            "numpy_version": np.__version__, "tinynumpy_version": importlib.metadata.version("tinynumpy"),
            "dtype": str(n.dtype),
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest(),
            "methods": ["calcPhaseProperties(liquid, oxide_grams)", "calcEndMemberProperties(liquid, oxide_grams)"],
        },
    }


def evaluate_liquid(temperature, pressure, component_moles, *, runtime, common_R=COMMON_R, python_executable=None):
    """Return JSON-compatible properties from a fresh pinned external worker.

    Inputs are K, Pa, and mol in COMPONENTS order. Ordinary imports require only
    NumPy; python_executable may select a separate environment with tinynumpy.
    Absent endmember potentials are None. Failure raises without returning a state.
    """
    n, _ = validate_request(temperature, pressure, component_moles, common_R)
    runtime = Path(runtime).resolve()
    check_runtime(runtime)
    request = {"T_K": float(temperature), "P_Pa": float(pressure), "component_moles": n.tolist(), "common_R_J_mol_K": float(common_R)}
    with tempfile.TemporaryDirectory(prefix="exoeos-melts-liquid-") as directory:
        request_path, output = Path(directory) / "request.json", Path(directory) / "result.json"
        request_path.write_text(json.dumps(request, allow_nan=False))
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(runtime) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        process = subprocess.run(
            [str(python_executable or sys.executable), str(Path(__file__).resolve()),
             "--runtime", str(runtime), "--input", str(request_path), "--output", str(output), "--worker"],
            cwd=directory, env=env, capture_output=True, text=True, timeout=60,
        )
        if process.returncode:
            raise RuntimeError(process.stdout[-4000:] + process.stderr[-4000:])
        log = process.stdout + process.stderr
        if "Iteration exceeded" in log or "convergence was acceptable" in log:
            raise RuntimeError("MELTS reported an unexpected solver fallback: " + log[-4000:])
        result = json.loads(output.read_text())
        result["provenance"]["backend_log"] = log
        result["provenance"]["caller_command"] = sys.argv
        return result


def validate_backend(runtime, python_executable=None):
    """Re-evaluate saved liquids, scale amounts, and differentiate actual G."""
    records = []
    evaluation_count = 0

    def evaluate(state, amounts):
        nonlocal evaluation_count
        evaluation_count += 1
        return evaluate_liquid(state["T_K"], state["P_Pa"], amounts, runtime=runtime, python_executable=python_executable)

    def derivatives(state, n, result, indices):
        errors = []
        for index in indices:
            # Differentiate independent endmember amounts, keeping all others fixed.
            step = n[index] * 1e-3
            plus, minus = n.copy(), n.copy()
            plus[index] += step
            minus[index] -= step
            fd = (evaluate(state, plus)["gibbs_J"] - evaluate(state, minus)["gibbs_J"]) / (2 * step)
            mu = result["mu_J_mol"][index]
            require_match(fd, mu, "dG/dn = mu", rtol=0, atol=0.02)
            errors.append(abs(fd - mu))
        return float(max(errors))

    for state in REFERENCE["states"]:
        n = np.asarray(state["liquid"]["component_moles"])
        result = evaluate(state, n)
        present = n > 0
        for field in ("mu_J_mol", "mu0_J_mol", "activity", "x"):
            require_match(np.asarray(result[field], dtype=float)[present], np.asarray(state["liquid"][field], dtype=float)[present], "saved " + field)
        require_match(result["gibbs_J"], state["phases"][0]["gibbs_J"], "saved liquid Gibbs energy")
        scaled = evaluate(state, 2.5 * n)
        require_match(scaled["gibbs_J"], 2.5 * result["gibbs_J"], "extensive Gibbs scaling")
        require_match(np.asarray(scaled["mu_J_mol"], float)[present], np.asarray(result["mu_J_mol"], float)[present], "intensive potential scaling")
        records.append({"id": state["id"], "max_fd_error_J_mol": derivatives(state, n, result, np.flatnonzero(present)), "result": result})
    # Element-isolated Fe, Si, O perturbations preserve every background budget.
    state = REFERENCE["states"][0]
    base = np.asarray(state["liquid"]["component_moles"])
    indices = [COMPONENTS.index(name) for name in ("sio2", "fe2o3", "fe2sio4")]
    elemental = [ELEMENTS.index(name) for name in ("Fe", "Si", "O")]
    matrix = FORMULA_MATRIX[np.ix_(indices, elemental)].T
    for element, unit in zip(("Fe", "Si", "O"), np.eye(3)):
        direction = np.linalg.solve(matrix, unit)
        magnitude = 0.02 * np.min(base[indices][direction != 0] / np.abs(direction[direction != 0]))
        for sign in (-1, 1):
            n = base.copy()
            n[indices] += sign * magnitude * direction
            result = evaluate(state, n)
            records.append({"id": f"{element}_{sign:+d}", "element_delta_mol": sign * magnitude, "max_fd_error_J_mol": derivatives(state, n, result, indices), "result": result})
    return {"status": "passed", "model_id": MODEL_ID, "evaluation_count": evaluation_count,
            "tolerances": {"property_rtol": 5e-9, "fd_rtol": 0, "fd_atol_J_mol": 0.02, "fd_relative_step": 1e-3},
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path, help="Pinned extracted Linux x86_64 alphaMELTS runtime.")
    parser.add_argument("--input", type=Path, help="JSON with T_K, P_Pa, component_moles; optional common_R_J_mol_K.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python", help="Separate Python interpreter with numpy and tinynumpy installed.")
    parser.add_argument("--validate", action="store_true", help="Run saved-liquid, Fe/Si/O, amount-scaling, and finite-difference checks.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.validate:
        if args.input or args.worker:
            parser.error("--validate cannot be combined with --input or --worker")
        result = validate_backend(args.runtime, args.python)
    else:
        if not args.input:
            parser.error("--input is required unless --validate is selected")
        request = json.loads(args.input.read_text())
        request.setdefault("common_R_J_mol_K", COMMON_R)
        result = _worker(args.runtime.resolve(), request) if args.worker else evaluate_liquid(
            request["T_K"], request["P_Pa"], request["component_moles"], runtime=args.runtime,
            common_R=request["common_R_J_mol_K"], python_executable=args.python,
        )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
