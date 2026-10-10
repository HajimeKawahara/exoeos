"""Independent pinned MELTS properties and Fe/Mg partitioning references.

No ExoEOS/JAX model or example equilibrium solver is imported. Each T/P
state runs in a fresh process. SciPy solves native chemical-potential roots;
this does not invoke MELTS full phase selection. Requires the pinned runtime,
SciPy and tinynumpy (for the external Python wrapper).
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


def worker(runtime, request):
    from scipy.optimize import brentq, least_squares
    from scipy.special import expit

    native.check_runtime(runtime)
    sys.path.insert(0, str(runtime))
    from meltsdynamic import MELTSdynamic

    model = MELTSdynamic(1)
    engine = model.engine
    engine.setSystemProperties("Log fO2 Path", "None")
    if engine.status.failed:
        raise RuntimeError("Could not disable oxygen buffering.")
    native.require_match(engine.status.molwts["bulk"], native.OXIDE_MASSES, "oxide masses", rtol=0)
    T, P = request["T_K"], request["P_Pa"]
    engine.temperature, engine.pressure = T - 273.15, P / 1e5

    def properties(phase, n):
        n = np.asarray(n, float)
        oxide = np.zeros(19)
        indices = [0, 7, 5] if phase == "liquid" else [5, 1]
        if phase == "liquid":
            oxide[[0, 7, 5]] = [sum(n), 2*n[1], 2*n[2]]
            native_n = np.zeros(19)
            native_n[indices] = n
        else:
            oxide[[0, 7, 5]] = [sum(n), 2*n[0], 2*n[1]]
            native_n = np.zeros(6)
            native_n[indices] = n
        grams = oxide * native.OXIDE_MASSES
        if phase == "liquid":
            engine.calcPhaseProperties(phase, grams.tolist())
            scale = 1.
        else:
            engine.molarComposition = np.r_[native_n/n.sum(), np.zeros(13)]
            engine.calcMolarProperties(phase)
            scale = n.sum()
        if engine.status.failed:
            raise RuntimeError(f"{phase} G failed")
        g = scale * float(engine.g[phase])
        returned = scale * float(engine.mass[phase]) * np.asarray(engine.dispComposition[phase]) / 100
        native.require_match(returned, grams, "unchanged phase oxide inventory", atol=1e-10)
        engine.calcEndMemberProperties(phase, grams.tolist())
        if engine.status.failed:
            raise RuntimeError(f"{phase} mu failed")
        native.require_match(engine.X[phase], native_n/n.sum(), "native component fractions", atol=1e-12)
        mu = np.array(engine.mu[phase], float)[indices]
        standard = np.array(engine.mu0[phase], float)[indices]
        present = n > 0
        native.require_match(n[present] @ mu[present], g, "Euler G", atol=3e-5)
        return g, mu, standard

    rows = []
    for y in (0., .2, .8, 1.):
        liquid_n, solid_n = [.25, .6, .15], [1-y, y]
        gl, ml, standards = properties("liquid", liquid_n)
        gs, ms, _ = properties("olivine", solid_n)
        rows.append({"T_K": T, "P_Pa": P, "liquid_n_mol": liquid_n,
                     "olivine_n_mol": solid_n, "liquid_g_J": gl, "olivine_g_J": gs,
                     "liquid_standard_J_mol": standards.tolist(),
                     "liquid_mu_J_mol": ml.tolist(),
                     "olivine_mu_J_mol": native.nullable(ms, np.array(solid_n) > 0)})

    equilibria = []
    for M, F in ((1.2, .3), (.6, .9)):
        bulk = np.array([1-(M+F)/2, M/2, F/2])
        gl, ml, _ = properties("liquid", bulk)

        def exchange(y):
            ms = properties("olivine", [1-y, y])[1]
            return (ms[1]-ms[0]) - (ml[2]-ml[1])

        y = brentq(exchange, 1e-8, 1-1e-8, xtol=1e-14)
        gs = properties("olivine", [1-y, y])[0]
        gap = gs - np.array([1-y, y]) @ ml[1:]
        solid, liquid = np.zeros(2), bulk.copy()
        residual = None
        if gap < 0:
            def coexistence(z):
                s = bulk[1:] * expit(z)
                remaining = np.r_[bulk[0], bulk[1:] - s]
                return properties("olivine", s)[1] - properties("liquid", remaining)[1][1:]

            result = least_squares(coexistence, [-.5, -2.],
                                   xtol=1e-13, ftol=1e-13, gtol=1e-9, max_nfev=100)
            if not result.success or np.max(np.abs(result.fun)) > 1e-6:
                raise RuntimeError(f"Native coexistence solve failed: {result}")
            solid = bulk[1:] * expit(result.x)
            liquid = np.r_[bulk[0], bulk[1:] - solid]
            residual = result.fun.tolist()
        energy = properties("liquid", liquid)[0]
        if solid.sum() > 0:
            energy += properties("olivine", solid)[0]
        equilibria.append({
            "T_K": T, "P_Pa": P, "initial_oxide_moles": {"MgO": M, "FeO": F, "SiO2": 1.},
            "amounts_mol": np.r_[liquid, solid].tolist(), "g_J": energy,
            "liquid_fe_fraction": float(liquid[2]/liquid[1:].sum()),
            "olivine_fe_fraction": float(solid[1]/solid.sum()) if solid.sum() else None,
            "Kd_Fe_Mg": float((solid[1]/solid[0])/(liquid[2]/liquid[1])) if solid.sum() else None,
            "initial_insertion_J_mol": gap, "incipient_olivine_fe_fraction": y,
            "coexistence_residual_J_mol": residual,
        })
    return {"properties": rows, "equilibria": equilibria,
            "python_version": sys.version, "tinynumpy_version": version("tinynumpy"),
            "scipy_version": version("scipy")}


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
    else:
        native.check_runtime(runtime)
        data = {"reference_kind": "pinned_MELTS_FeII_liquid_olivine",
                "backend": native.REFERENCE["backend"],
                "method": "Fresh worker per T/P; native G/mu, brentq incipient olivine and independent least_squares coexistence roots. No JAX or full MELTS phase selection.",
                "liquid_order": ["SiO2", "Mg2SiO4", "Fe2SiO4"],
                "olivine_order": ["Mg2SiO4", "Fe2SiO4"],
                "amount_order": ["liquid Q", "liquid F", "liquid A", "solid Fo", "solid Fa"],
                "iron_valence": 2, "oxygen_buffer": None,
                "evaluator_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "properties": [], "equilibria": []}
        for P in (1e5, 5e7, 5e8):
            for T in (1600., 1800., 1900., 2000., 2200.):
                with tempfile.TemporaryDirectory(prefix="exoeos-olivine-") as directory:
                    root = Path(directory)
                    request, output = root / "request.json", root / "result.json"
                    request.write_text(json.dumps({"T_K": T, "P_Pa": P}))
                    env = dict(os.environ)
                    env["LD_LIBRARY_PATH"] = str(runtime) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
                    result = subprocess.run(
                        [args.python, str(Path(__file__).resolve()), "--runtime", str(runtime),
                         "--worker", str(request), "--output", str(output)],
                        env=env, cwd=root, capture_output=True, text=True, timeout=60)
                    log = result.stdout + result.stderr
                    if result.returncode or "Iteration exceeded" in log or "convergence was acceptable" in log:
                        raise RuntimeError(log)
                    row = json.loads(output.read_text())
                    for key in ("properties", "equilibria"):
                        data[key].extend(row[key])
                    for key in ("python_version", "tinynumpy_version", "scipy_version"):
                        data[key] = row[key]
                print(f"Saved {T:g} K, {P/1e5:g} bar", flush=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
