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
MODEL_IDS = {1: MODEL_ID, 4: "alphamelts_2_3_2_rhyolite_melts_1_2_0_supplied_liquid_v1"}
BACKEND_MODELS = {1: "rhyolite-MELTS 1.0.2", 4: "rhyolite-MELTS 1.2.0"}
UNSUPPORTED_COMPONENTS = ("so3", "cl2o-1", "f2o-1")
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


def validate_request(temperature, pressure, component_moles, common_R, calculation_mode=1):
    if type(calculation_mode) is not int or calculation_mode not in MODEL_IDS:
        raise ValueError("calculation_mode must be 1 or 4; carbon requires mode 4.")
    n = np.asarray(component_moles, dtype=float)
    if n.shape != (len(COMPONENTS),) or not np.all(np.isfinite(n)) or np.any(n < 0):
        raise ValueError("Supply 19 finite nonnegative endmember amounts in component_order.")
    if not np.isfinite(n.sum()) or n.sum() <= 0:
        raise ValueError("The liquid must have a finite positive total amount.")
    unsupported = UNSUPPORTED_COMPONENTS + (("co2",) if calculation_mode == 1 else ())
    for component in unsupported:
        if n[COMPONENTS.index(component)] > 0:
            raise ValueError(f"Unsupported positive component {component} in calculation_mode {calculation_mode}.")
    if not np.isfinite(temperature) or temperature <= 0 or temperature == 273.15:
        raise ValueError("T_K must be positive and finite; the wrapper cannot accept exactly 0 Celsius.")
    if not np.isfinite(pressure) or pressure <= 0 or not np.isfinite(common_R) or common_R <= 0:
        raise ValueError("P_Pa and common_R_J_mol_K must be positive and finite.")
    grams = (NU.T @ n) * OXIDE_MASSES
    if not np.all(np.isfinite(grams)) or not np.isfinite(grams.sum()) or grams.sum() <= 0:
        raise ValueError("Converted oxide masses must be finite with a positive total.")
    return n, grams


def independent_fractions(species_x, calculation_mode):
    """Undo CaSiO3 + CO2 = CaCO3 + SiO2 in the reported species amounts."""
    species_x = np.asarray(species_x, dtype=float)
    expected_size = len(COMPONENTS) + (calculation_mode == 4)
    if species_x.shape != (expected_size,) or not np.all(np.isfinite(species_x)) or np.any(species_x < 0):
        raise ValueError("Invalid backend liquid species fractions.")
    x = species_x[:len(COMPONENTS)].copy()
    if calculation_mode == 4:
        carbonate = species_x[-1]
        x[COMPONENTS.index("sio2")] -= carbonate
        x[COMPONENTS.index("casio3")] += carbonate
        x[COMPONENTS.index("co2")] += carbonate
    return x


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


def molar_candidate_properties(model, name, oxide_mass_g=None):
    """Evaluate a solid on its native endmember basis, preserving oxide amounts.

    The oxide-to-endmember converter in some native phases changes the input
    composition. Unit endmember properties define the inverse basis without
    parsing display formulas or changing the pinned native model. The native
    molar API normalizes its input to one mole; restore the requested amount.
    """
    engine = model.engine
    count = len(engine.status.molwts[name])
    width = len(engine.status.molwts["bulk"])
    columns = []
    for index in range(count):
        engine.molarComposition = np.eye(width)[index]
        engine.calcMolarProperties(name)
        if engine.status.failed:
            raise RuntimeError(f"MELTS molar basis evaluation failed: {name}")
        columns.append(float(engine.mass[name]) * np.asarray(engine.dispComposition[name]) / 100.)
    matrix = np.asarray(columns, dtype=float).T
    if (matrix.shape != (width, count) or not np.all(np.isfinite(matrix))
            or np.linalg.matrix_rank(matrix) != count):
        raise ValueError("Invalid or singular native endmember oxide basis.")
    result = {"phase": name, "native_endmember_oxide_mass_g_per_mol": matrix.tolist(),
              "composition_basis_method": "Native unit endmembers through calcMolarProperties."}
    if oxide_mass_g is None:
        return result
    oxide = np.asarray(oxide_mass_g, dtype=float)
    if oxide.shape != (width,) or not np.all(np.isfinite(oxide)) or oxide.sum() <= 0:
        raise ValueError("Candidate oxide amounts must be finite with positive mass.")
    amounts, _, _, _ = np.linalg.lstsq(matrix, oxide, rcond=None)
    require_match(matrix @ amounts, oxide, "candidate endmember reconstruction", atol=1e-9)
    total = float(amounts.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Candidate endmember total must be positive.")
    engine.molarComposition = np.r_[amounts / total, np.zeros(width - count)]
    engine.calcMolarProperties(name)
    if engine.status.failed:
        raise RuntimeError(f"MELTS molar candidate evaluation failed: {name}")
    mass, gibbs = total * float(engine.mass[name]), total * float(engine.g[name])
    returned = mass * np.asarray(engine.dispComposition[name], dtype=float) / 100.
    if not np.isfinite(gibbs) or not np.isfinite(mass) or mass <= 0 or not np.all(np.isfinite(returned)):
        raise ValueError("Nonfinite native molar candidate properties.")
    require_match(returned, oxide, "candidate oxide amounts", atol=1e-9)
    result.update(oxide_mass_g=oxide.tolist(), native_endmember_moles=amounts.tolist(),
                  gibbs_J=gibbs, mass_g=mass, returned_oxide_mass_g=returned.tolist(),
                  status="ok_candidate_properties", reason=None)
    return result


def saturation_properties(model, grams):
    """Return native candidate properties without equilibrating the liquid.

    Affinities retain the backend's convention. Candidate energies instead
    use an explicit 100 g oxide basis, including signed redox oxide amounts.
    Failed candidates stay in the catalog and never imply phase absence.
    """
    engine = model.engine
    engine.setBulkComposition(grams.tolist())
    phases = engine.calcSaturationState()
    if engine.status.failed:
        raise RuntimeError("MELTS supplied-liquid saturation calculation failed.")
    require_match(engine.bulkComposition, grams, "saturation input oxide amounts", rtol=0)
    catalog = [name for name in model.systemNames if name not in {"bulk", "oxygen", "liquid"}]
    # Phase-property calls overwrite display compositions, so capture all
    # incipient compositions before evaluating any candidate.
    native = [(name, engine.affinity.get(name),
               np.asarray(engine.dispComposition[name], dtype=float).copy()
               if name in phases and name in engine.dispComposition else None)
              for name in catalog]
    candidates = []
    for name, affinity, oxide in native:
        row = {"phase": name, "native_affinity_J": None,
               "status": "unavailable", "reason": "No finite native saturation estimate."}
        if affinity is not None and np.isfinite(affinity):
            row["native_affinity_J"] = float(affinity)
        if row["native_affinity_J"] is not None and oxide is not None and np.all(np.isfinite(oxide)):
            if oxide.shape != grams.shape or not np.isclose(oxide.sum(), 100., rtol=1e-10):
                row["reason"] = "Invalid native incipient oxide basis."
            else:
                row["oxide_mass_g"] = oxide.tolist()
                engine.calcPhaseProperties(name, oxide.tolist())
                if engine.status.failed:
                    raise RuntimeError(f"MELTS candidate property calculation failed: {name}")
                mass, gibbs = float(engine.mass[name]), float(engine.g[name])
                returned = mass * np.asarray(engine.dispComposition[name], dtype=float) / 100
                if np.isfinite(mass) and np.isfinite(gibbs) and np.all(np.isfinite(returned)):
                    row.update(gibbs_J=gibbs, mass_g=mass, returned_oxide_mass_g=returned.tolist())
                    if mass > 0 and np.allclose(returned, oxide, rtol=5e-9, atol=1e-9):
                        row.update(status="ok_candidate_properties", reason=None)
                    else:
                        row["reason"] = "Candidate property call changed the requested oxide amounts."
                else:
                    row["reason"] = "Nonfinite candidate properties."
                if row["status"] == "unavailable":
                    row["oxide_conversion_diagnostic"] = {
                        key: row.get(key) for key in ("reason", "mass_g", "gibbs_J", "returned_oxide_mass_g")}
                    try:
                        row.update(molar_candidate_properties(model, name, oxide))
                    except ValueError as error:
                        row["reason"] = str(error)
        candidates.append(row)
    require_match(engine.bulkComposition, grams, "unchanged saturation input oxide amounts", rtol=0)
    return {
        "status": "native_saturation_candidates_only", "candidate_order": catalog,
        "candidates": candidates, "equilibrated": False, "global_minimum_certified": False,
        "affinity_convention": "Native MELTS affinity in J; smaller means closer to saturation. Candidate Gibbs energies use the independently recorded oxide-mass basis.",
        "composition_search": "Native incipient estimates; no global minimization certificate.",
        "excluded_system_entries": ["bulk", "oxygen", "liquid"],
        "scope": "Native supplied liquid only; external solutes, alloy models and gas standards are not included.",
    }


def _worker(runtime, request):
    """Only called in a new process and temporary working directory."""
    temperature, pressure = request["T_K"], request["P_Pa"]
    common_R = request["common_R_J_mol_K"]
    calculation_mode = request.get("calculation_mode", 1)
    n, grams = validate_request(temperature, pressure, request["component_moles"], common_R, calculation_mode)
    check_runtime(runtime)
    sys.path.insert(0, str(runtime))
    from meltsdynamic import MELTSdynamic

    model = MELTSdynamic(calculation_mode)
    engine = model.engine
    species_order = COMPONENTS + (["caco3"] if calculation_mode == 4 else [])
    if model.endMemberFormulas["bulk"] != OXIDES or model.endMemberFormulas["liquid"] != species_order:
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
    species_x = np.asarray(engine.X["liquid"])
    x = independent_fractions(species_x, calculation_mode)
    require_match(x, n / n.sum(), "endmember mole fractions")
    present = n > 0
    species_properties = [np.asarray(getattr(engine, field)["liquid"]) for field in ("mu", "mu0", "activity")]
    if any(values.shape != species_x.shape for values in species_properties):
        raise ValueError("Unexpected backend species property dimensions.")
    species_mu = species_properties[0]
    if calculation_mode == 4 and species_x[-1] > 0:
        expected_mu = species_mu[COMPONENTS.index("casio3")] + species_mu[COMPONENTS.index("co2")] - species_mu[COMPONENTS.index("sio2")]
        require_match(species_mu[-1], expected_mu, "carbonate reaction potential", atol=1e-5)
    # The backend calculates independent-component potentials before reporting
    # carbonate species fractions; the extra potential follows the reaction.
    mu, mu0, activity = [values[:len(COMPONENTS)] for values in species_properties]
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
    result = {
        "model_id": MODEL_IDS[calculation_mode], "status": "ok_supplied_liquid_properties",
        "T_K": temperature, "P_Pa": pressure,
        "backend_T_C": engine.temperature, "backend_P_bar": engine.pressure,
        "component_order": COMPONENTS, "component_moles": n.tolist(),
        "returned_component_moles": returned_n.tolist(), "x": x.tolist(),
        "backend_species": {"order": species_order, "x": species_x.tolist(),
                            "reaction": "casio3 + co2 = caco3 + sio2" if calculation_mode == 4 else None},
        "oxide_order": OXIDES, "oxide_mass_g": grams.tolist(),
        "oxide_molar_masses_g_mol": OXIDE_MASSES.tolist(),
        "returned_oxide_mass_g": returned_grams.tolist(), "mass_g": mass, "gibbs_J": gibbs,
        "gibbs_RT": gibbs / (common_R * temperature),
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
            "amount_unit": "mol of named independent MELTS liquid component", "energy_unit": "J",
            "component_oxide_matrix": NU.tolist(), "element_order": ELEMENTS,
            "component_element_matrix": FORMULA_MATRIX.tolist(),
            "element_moles": (FORMULA_MATRIX.T @ n).tolist(),
            "standard": "pure liquid endmember at supplied T_K/P_Pa; no separate pressure correction",
            "backend_R_J_mol_K": BACKEND_R, "common_R_J_mol_K": common_R,
            "dimensionless_potential": "mu_RT = mu_J_mol / (common_R_J_mol_K * T_K)",
            "activity_convention": "activity, ln_activity, ln_gamma use backend_R; ln_activity_common_R = (mu - mu0)/(common_R*T_K); ln_gamma_common_R = ln_activity_common_R - ln(x)",
            "mole_fraction_convention": "x and ln_gamma use independent component amounts; backend_species.x reports internal species, including carbonate in mode 4.",
            "mixing": "Full potentials already include ideal and excess mixing; do not add mixing again.",
        },
        "phase_policy": {
            "oxygen_buffer": "None", "equilibrated": False, "stability_checked": False,
            "calibration_domain": "Not established by this property evaluator; three MORB fixtures and local perturbations are equation references only.",
            "cross_phase_standards": "Not aligned to alloy/gas; oxide basis conversion does not establish pure-oxide standards.",
        },
        "capabilities": {
            "carbon": "co2 component with internal carbonate speciation" if calculation_mode == 4 else "unsupported; select calculation_mode 4",
            "nitrogen": "unsupported; no nitrogen component in the backend basis",
            "sulfur": "unsupported; so3 is an unparameterized placeholder",
            "unsupported_positive_components": list(UNSUPPORTED_COMPONENTS) + (["co2"] if calculation_mode == 1 else []),
        },
        "provenance": {
            **source_provenance(),
            "backend": {**REFERENCE["backend"], "model": BACKEND_MODELS[calculation_mode], "calculation_mode": calculation_mode}, "backend_version": model.version,
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
    if request.get("include_saturation", False):
        result["saturation"] = saturation_properties(model, grams)
        result["provenance"]["methods"].extend([
            "calcSaturationState(supplied_liquid_oxide_grams)",
            "calcPhaseProperties(candidate, incipient_oxide_grams)",
            "calcMolarProperties(candidate, reconstructed_native_endmembers) when oxide conversion fails",
        ])
    if "candidate_compositions" in request:
        rows = []
        for candidate in request["candidate_compositions"]:
            name = candidate["phase"]
            if name not in model.systemNames or name in {"bulk", "oxygen", "liquid"}:
                raise ValueError("Supply a native solid candidate phase.")
            try:
                rows.append(molar_candidate_properties(model, name, candidate.get("oxide_mass_g")))
            except ValueError as error:
                rows.append({"phase": name, "status": "unavailable", "reason": str(error)})
        result["candidate_evaluations"] = rows
        result["provenance"]["methods"].append("calcMolarProperties(candidate, native_endmember_basis)")
    return result


def evaluate_liquid(temperature, pressure, component_moles, *, runtime, common_R=COMMON_R, python_executable=None, calculation_mode=1, include_saturation=False, candidate_compositions=None):
    """Return JSON-compatible properties from a fresh pinned external worker.

    Inputs are K, Pa, and mol in COMPONENTS order. Ordinary imports require only
    NumPy; python_executable may select a separate environment with tinynumpy.
    Mode 4 enables CO2 and internal carbonate speciation; mode 1 is the default.
    Absent component potentials are None. Failure raises without returning a state.
    Optional native saturation candidates do not certify equilibrium or stability.
    """
    n, _ = validate_request(temperature, pressure, component_moles, common_R, calculation_mode)
    runtime = Path(runtime).resolve()
    check_runtime(runtime)
    if type(include_saturation) is not bool:
        raise ValueError("include_saturation must be a boolean.")
    request = {"T_K": float(temperature), "P_Pa": float(pressure), "component_moles": n.tolist(), "common_R_J_mol_K": float(common_R), "calculation_mode": calculation_mode, "include_saturation": include_saturation}
    if candidate_compositions is not None:
        if (not isinstance(candidate_compositions, list)
                or any(not isinstance(row, dict) or not isinstance(row.get("phase"), str)
                       or set(row) - {"phase", "oxide_mass_g"} for row in candidate_compositions)):
            raise ValueError("Candidate compositions require phase and optional oxide_mass_g.")
        request["candidate_compositions"] = candidate_compositions
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
    parser.add_argument("--input", type=Path, help="JSON with T_K, P_Pa, component_moles; optional common_R_J_mol_K and calculation_mode (1 or 4).")
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
            calculation_mode=request.get("calculation_mode", 1),
            include_saturation=request.get("include_saturation", False),
            candidate_compositions=request.get("candidate_compositions"),
        )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
