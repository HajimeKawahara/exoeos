"""Generate dry CMFAS references using the hash-pinned MELTS runtime.

No ExoEOS/JAX thermodynamic model or example solver is imported. Records
native supplied properties, independently solved restricted equilibria,
and separately labelled full MELTS phase selection for the same bulk.
Requires SciPy, tinynumpy and the separately installed pinned runtime.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
import melts_liquid_evaluator as native  # noqa: E402
from generate_magma_reference import REVISION, SOURCE_HASHES  # noqa: E402

LIQUID_INDICES = [0, 7, 5, 10, 2]
LIQUID_ORDER = ["SiO2", "Mg2SiO4", "Fe2SiO4", "CaSiO3", "Al2O3"]
OXIDES = {"SiO2": 1., "MgO": .55, "FeO": .15, "CaO": .25, "Al2O3": .12}
# Construct each solid in the liquid basis independently from its formula.
COLUMNS = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                    [.5, .5, 0, 1, 0], [.5, 0, .5, 1, 0],
                    [1, 0, 0, 1, 1], [.5, .5, 0, 0, 0], [1, 0, 0, 0, 0]]).T


def worker(runtime, request):
    from scipy.optimize import brentq, minimize, root

    native.check_runtime(runtime)
    sys.path.insert(0, str(runtime))
    from meltsdynamic import MELTSdynamic

    engine = MELTSdynamic(1).engine
    engine.setSystemProperties("Log fO2 Path", "None")
    if engine.status.failed:
        raise RuntimeError("Could not disable oxygen buffering.")
    T, P = request["T_K"], request["P_Pa"]
    engine.temperature, engine.pressure = T - 273.15, P / 1e5

    def properties(phase, amounts):
        n = np.asarray(amounts, float)
        if phase == "liquid":
            indices = LIQUID_INDICES
            raw = np.zeros(19)
            raw[indices] = n
            grams = native.NU.T @ raw * native.OXIDE_MASSES
            engine.calcPhaseProperties(phase, grams.tolist())
            factor = 1.
        else:
            indices, count = {"olivine": ([5, 1], 6), "clinopyroxene": ([0, 2], 7),
                              "plagioclase": ([1], 3), "orthopyroxene": ([1], 7),
                              "quartz": ([0], 1), "tridymite": ([0], 1),
                              "cristobalite": ([0], 1)}[phase]
            raw = np.zeros(count)
            raw[indices] = n
            engine.molarComposition = np.r_[raw / n.sum(), np.zeros(19-count)]
            engine.calcMolarProperties(phase)
            factor = n.sum()
            columns = {"olivine": COLUMNS[:, :2], "clinopyroxene": COLUMNS[:, 2:4],
                       "plagioclase": COLUMNS[:, 4:5], "orthopyroxene": 2*COLUMNS[:, 5:6],
                       "quartz": COLUMNS[:, 6:], "tridymite": COLUMNS[:, 6:],
                       "cristobalite": COLUMNS[:, 6:]}[phase]
            grams = native.NU[LIQUID_INDICES].T @ (columns @ n) * native.OXIDE_MASSES
        if engine.status.failed:
            raise RuntimeError(f"Native {phase} properties failed.")
        g = factor * float(engine.g[phase])
        if not np.isfinite(g):
            raise FloatingPointError(f"Pinned MELTS returned nonfinite {phase} G at {n.tolist()}.")
        returned = factor * float(engine.mass[phase]) * np.asarray(engine.dispComposition[phase]) / 100
        native.require_match(returned, grams, "unchanged oxide amounts", atol=1e-9)
        if phase not in ("liquid", "olivine", "clinopyroxene"):
            return g, None, None
        engine.calcEndMemberProperties(phase, grams.tolist())
        if engine.status.failed:
            raise RuntimeError(f"Native {phase} chemical potentials failed.")
        native.require_match(engine.X[phase], raw/raw.sum(), "native fractions", atol=1e-10)
        mu = np.asarray(engine.mu[phase], float)[indices]
        g0 = np.asarray(engine.mu0[phase], float)[indices]
        present = n > 0
        native.require_match(n[present] @ mu[present], g,
                             f"{phase} Euler identity at {n}: {n[present] @ mu[present]} vs {g}", atol=4e-5)
        return g, mu, g0

    bulk = np.array([.4, .275, .075, .25, .12])
    if request.get("full_equilibrium"):
        # A separate worker prevents supplied-property calls from changing the solver state.
        oxide_moles = {key.lower(): value for key, value in OXIDES.items()}
        grams = np.array([oxide_moles.get(oxide.lower(), 0.) for oxide in native.OXIDES]) * native.OXIDE_MASSES
        engine.setBulkComposition(grams.tolist())
        engine.calcEquilibriumState(1, 0)
        if engine.status.failed:
            raise RuntimeError(f"Full MELTS equilibrium failed: {engine.status.message}")
        phases = [{"name": name, "mass_g": float(engine.mass[name]),
                   "oxide_wt_percent": list(engine.dispComposition[name])}
                  for name in engine.liquidNames + engine.solidNames]
        recovered = sum(row["mass_g"] * np.asarray(row["oxide_wt_percent"]) / 100 for row in phases)
        native.require_match(recovered, grams, "closed full MELTS oxide balance", atol=2e-6)
        return {"full_equilibrium": {"T_K": T, "P_Pa": P, "initial_oxide_moles": OXIDES,
                                    "phases": phases, "solver_message": engine.status.message}}

    gl, ml, standards = properties("liquid", bulk)
    pure = np.array([properties("plagioclase", [1.])[0],
                     properties("orthopyroxene", [1.])[0]/2,
                     min(properties(name, [1.])[0] for name in ("quartz", "tridymite", "cristobalite"))])
    rows = []
    for n in ([1., 0.], [.8, .2], [.2, .8], [0., 1.], [1.4, .6]):
        try:
            g, mu, g0 = properties("clinopyroxene", n)
        except FloatingPointError as error:
            if min(n) != 0:
                raise
            rows.append({"n_mol": n, "status": "unavailable_nonfinite_native_endpoint",
                         "reason": str(error)})
            continue
        rows.append({"n_mol": n, "g_J": g, "mu_J_mol": native.nullable(mu, np.array(n) > 0),
                     "standard_J_mol": g0.tolist()})
    data = {"properties": {"T_K": T, "P_Pa": P, "liquid_mol": bulk.tolist(),
                            "liquid_g_J": gl, "liquid_mu_J_mol": ml.tolist(),
                            "liquid_standard_J_mol": standards.tolist(),
                            "anorthite_g_J_mol": pure[0], "clinopyroxene": rows}}
    if not request.get("restricted_equilibrium"):
        return data

    def unpack(z):
        s = np.r_[z[0]*np.array([1-z[5], z[5]]), z[1]*np.array([1-z[6], z[6]]), z[2:5]]
        jac = np.zeros((7, 7))
        jac[:2, 0], jac[2:4, 1] = [1-z[5], z[5]], [1-z[6], z[6]]
        jac[4:, 2:5] = np.eye(3)
        jac[:2, 5], jac[2:4, 6] = [-z[0], z[0]], [-z[1], z[1]]
        return s, jac

    baseline = bulk @ standards
    def objective(z):
        s, jac = unpack(z)
        liquid = bulk - COLUMNS @ s
        gl, ml, _ = properties("liquid", liquid)
        go, mo, _ = properties("olivine", [1-z[5], z[5]])
        gc, mc, _ = properties("clinopyroxene", [1-z[6], z[6]])
        g = gl + z[0]*go + z[1]*gc + z[2:5] @ pure
        grad = jac.T @ (np.r_[mo, mc, pure] - COLUMNS.T @ ml)
        return (g-baseline)/(native.BACKEND_R*T), grad/(native.BACKEND_R*T)

    def gap(phase, mu, cols):
        potential = COLUMNS[:, cols].T @ mu
        def slope(y):
            _, m, _ = properties(phase, [1-y, y])
            return m[1]-m[0]-potential[1]+potential[0]
        y = brentq(slope, 1e-10, 1-1e-10, xtol=2e-14)
        return properties(phase, [1-y, y])[0] - np.dot([1-y, y], potential), y

    y = [gap("olivine", ml, [0, 1])[1], gap("clinopyroxene", ml, [2, 3])[1]]
    constraint = {"type": "ineq", "fun": lambda z: bulk-COLUMNS @ unpack(z)[0]-1e-9,
                  "jac": lambda z: -COLUMNS @ unpack(z)[1]}
    # Native property evaluation rejects trial states with negative liquid amounts.
    # SLSQP may probe infeasible states, so provide a smooth linear penalty there.
    def feasible_objective(z):
        c = constraint["fun"](z)
        if min(c) <= 0:
            negative = np.minimum(c, 0.)
            return 1e5 + 1e8 * (negative @ negative), 2e8 * constraint["jac"](z).T @ negative
        return objective(z)

    solution = minimize(feasible_objective, np.r_[np.zeros(5), y], jac=True, method="SLSQP",
                        bounds=[(0., 2.)]*5 + [(1e-8, 1-1e-8)]*2,
                        constraints=constraint, options={"ftol": 1e-12, "maxiter": 400})
    if not solution.success:
        raise RuntimeError(solution.message)
    z = solution.x.copy()
    active = z[:5] > 1e-7
    z[:5][~active] = 0.
    indices = np.r_[np.flatnonzero(active), [5] if active[0] else [], [6] if active[1] else []].astype(int)
    if len(indices):
        def residual(v):
            trial = z.copy()
            trial[indices] = v
            return objective(trial)[1][indices]
        polished = root(residual, z[indices], tol=1e-10)
        if max(abs(polished.fun)) > 1e-8:
            raise RuntimeError("Native coexistence solve failed.")
        z[indices] = polished.x
    s, _ = unpack(z)
    liquid = bulk - COLUMNS @ s
    _, ml, _ = properties("liquid", liquid)
    gaps = np.r_[gap("olivine", ml, [0, 1])[0], gap("clinopyroxene", ml, [2, 3])[0],
                 pure-COLUMNS[:, 4:].T @ ml]
    if min(gaps) < -2e-4 or (any(active) and max(abs(gaps[active])) > 2e-4):
        raise RuntimeError(f"Native solid stability failed: {gaps}")
    if min(liquid) <= 0 or min(s) < 0:
        raise RuntimeError("Native equilibrium left its physical domain.")
    data["restricted_equilibrium"] = {
        "T_K": T, "P_Pa": P, "initial_oxide_moles": OXIDES, "liquid_mol": liquid.tolist(),
        "solid_mol": s.tolist(), "phase_mol": z[:5].tolist(),
        "g_J": objective(z)[0]*native.BACKEND_R*T+baseline,
        "solid_insertion_J_mol": gaps.tolist()}
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    if args.worker:
        data = worker(runtime, json.loads(args.worker.read_text()))
        data["worker_versions"] = {"python_version": sys.version.split()[0],
                                   "scipy_version": version("scipy"),
                                   "tinynumpy_version": version("tinynumpy")}
    else:
        native.check_runtime(runtime)
        data = {"backend": native.REFERENCE["backend"], "liquid_order": LIQUID_ORDER,
                "phase_order": ["olivine", "clinopyroxene", "anorthite", "orthopyroxene", "silica"],
                "method": "Native supplied properties; independent SLSQP and chemical-potential roots for restricted equilibria; separate full MELTS equilibrium workers. No JAX imports.",
                "iron_valence_restricted": 2, "oxygen_buffer": None,
                "full_equilibrium_scope": "All native phases and their full solution models; closed oxygen, no imposed oxygen buffer. Not the restricted JAX candidate set.",
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_revision": REVISION,
                "source_sha256": {**SOURCE_HASHES,
                    "includes/param_struct_data_v34.h": "ad78f9ac9a59c030aeee240893b00efc034ea81aed4d551abaed770215af7059",
                    "sources/clinopyroxene.c": "0a50a5171ec75f77fc439bebcd1b96f70962859eaf3c0f532d072d847fa585a0"},
                "evaluator_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                "generator_python_version": sys.version.split()[0],
                "properties": [], "restricted_equilibria": [], "full_equilibria": [],
                "unavailable_full_equilibria": []}
        requests = [{"T_K": T, "P_Pa": P, "restricted_equilibrium": P == 1e5}
                    for P in (1e5, 5e7, 5e8)
                    for T in (1400., 1425., 1450., 1475., 1500., 1600., 1800., 2000.)]
        requests += [{"T_K": T, "P_Pa": 1e5, "full_equilibrium": True}
                     for T in (1400., 1425., 1450., 1475., 1500., 1600., 1800., 2000.)]
        for request in requests:
            with tempfile.TemporaryDirectory(prefix="exoeos-dry-melts-") as tmp:
                tmp = Path(tmp)
                inp, out = tmp / "request.json", tmp / "result.json"
                inp.write_text(json.dumps(request))
                env = dict(os.environ)
                env["LD_LIBRARY_PATH"] = str(runtime) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
                result = subprocess.run(
                    [args.python, str(Path(__file__).resolve()), "--runtime", str(runtime),
                     "--worker", str(inp), "--output", str(out)],
                    cwd=tmp, env=env, capture_output=True, text=True, timeout=60)
                log = result.stdout + result.stderr
                warnings = [message for message in ("Iteration exceeded", "convergence was acceptable")
                            if message in log]
                if result.returncode or warnings:
                    if request.get("full_equilibrium"):
                        data["unavailable_full_equilibria"].append({
                            **request, "returncode": result.returncode,
                            "reason": "; ".join(warnings) if warnings else log[-2000:],
                            "worker_log_sha256": hashlib.sha256(log.encode()).hexdigest()})
                        print(f"Rejected unconverged full MELTS state: {request}", flush=True)
                        continue
                    raise RuntimeError(log)
                row = json.loads(out.read_text())
                data.update(row["worker_versions"])
                for singular, plural in [("properties", "properties"),
                                         ("restricted_equilibrium", "restricted_equilibria"),
                                         ("full_equilibrium", "full_equilibria")]:
                    if singular in row:
                        data[plural].append(row[singular])
                print(f"Saved {request}", flush=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
