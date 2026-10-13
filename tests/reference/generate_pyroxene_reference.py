"""Generate independent Ca--Mg--Fe properties with the pinned MELTS runtime.

No ExoEOS/JAX thermodynamic functions are imported. Requires tinynumpy and
the separately installed alphaMELTS runtime; ordinary tests use the JSON.
"""

import argparse
import hashlib
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

COMPOSITIONS = [[.8, 1., .2], [.05, 1.5, .45], [.2, .3, 1.5], [.4, 2.8, .8],
                [0., 1.6, .4], [.3, 1.7, 0.], [0., 2., 0.], [0., 0., 2.],
                [1., .8, .2], [1., 1., 0.], [1., 0., 1.]]
# Columns convert physical [CaSiO3, MgSiO3, FeSiO3] to native [Di, En, Hd].
NATIVE_MAP = np.array([[1., 0., -1.], [-.5, .5, .5], [0., 0., 1.]])
LOW_CA_OXIDES = {"SiO2": 1., "MgO": .9, "FeO": .1, "CaO": .04, "Al2O3": .02}


def assemblage_root(engine, request):
    """Independent native chemical-potential root for liquid + olivine + pyroxene.

    The assemblage is specified, not selected by MELTS. Its stability against
    the other admitted solids is checked separately by the JAX example.
    """
    from scipy.optimize import root
    from scipy.special import expit, logit

    oxides, phase = request["oxides"], request["phase"]
    q, m, f, c, a = [oxides[key] for key in ("SiO2", "MgO", "FeO", "CaO", "Al2O3")]
    bulk = np.array([q-(m+f)/2-c, m/2, f/2, c, a])
    px_map = np.array([[0., .5, .5], [0., .5, 0.], [0., 0., .5], [1., 0., 0.], [0., 0., 0.]])
    liquid_indices = [0, 7, 5, 10, 2]

    def properties(name, n):
        if name == "liquid":
            oxide = native.NU[liquid_indices].T@n
        else:
            oxide = np.zeros(19)
            if name == "olivine":
                oxide[[0, 5, 7]] = [sum(n), 2*n[1], 2*n[0]]
            else:
                oxide[[0, 5, 7, 10]] = [sum(n), n[2], n[1], n[0]]
        grams = oxide*native.OXIDE_MASSES
        engine.calcPhaseProperties(name, grams.tolist())
        if engine.status.failed:
            raise RuntimeError(f"Native {name} G failed in assemblage root.")
        g = float(engine.g[name])
        engine.calcEndMemberProperties(name, grams.tolist())
        if engine.status.failed:
            raise RuntimeError(f"Native {name} mu failed in assemblage root.")
        raw = np.asarray(engine.mu[name])
        mu = raw[liquid_indices] if name == "liquid" else raw[[5, 1]] if name == "olivine" else NATIVE_MAP.T@raw[:3]
        if not np.all(np.isfinite(mu)) or not np.isfinite(g):
            raise FloatingPointError("Nonfinite native root properties.")
        return g, mu

    def evaluate(z):
        no, npy, yo, c, y = z
        ol = np.array([1-yo, yo])
        px = np.array([c/2, (1-c/2)*(1-y), (1-c/2)*y])
        liquid = bulk-no*np.r_[0., ol, 0., 0.]-npy*px_map@px
        if min(liquid) <= 0 or min(z[:2]) <= 0 or min(z[2:]) <= 0 or max(z[2:]) >= 1:
            raise ValueError("Root trial left the physical interior.")
        gl, ml = properties("liquid", liquid)
        go, mo = properties("olivine", ol)
        gp, mp = properties(phase, px)
        do, dp = mo-ml[1:3], mp-px_map.T@ml
        residual = np.array([ol@do, px@dp, do[1]-do[0],
                              np.array([.5, -.5*(1-y), -.5*y])@dp,
                              np.array([0., -(1-c/2), 1-c/2])@dp])
        return residual, liquid, no*ol, npy*px, gl+no*go+npy*gp

    guess = ([.18, .06, .3, .7, .13] if phase == "clinopyroxene"
             else [.1, .6, .15, .02, .08])

    def transform(values, inverse=False):
        z = np.array(values) if inverse else expit(values)
        yo, c, y = z[2:]
        ol = np.r_[0., 1-yo, yo, 0., 0.]
        maximum_ol = min(bulk[1:3]/ol[1:3])
        no = z[0] if inverse else maximum_ol*z[0]
        px = px_map@np.array([c/2, (1-c/2)*(1-y), (1-c/2)*y])
        maximum_px = min(((bulk-no*ol)/np.where(px > 0, px, np.inf))[:4])
        return (logit(np.r_[no/maximum_ol, z[1]/maximum_px, z[2:]]) if inverse
                else np.r_[no, maximum_px*z[1], z[2:]])

    result = root(lambda u: evaluate(transform(u))[0]/(8.3143*request["T_K"]),
                  transform(guess, inverse=True), tol=1e-10, options={"factor": .1})
    residual, liquid, ol, px, g = evaluate(transform(result.x))
    if max(abs(residual)) > 2e-6:
        raise RuntimeError(f"Unresolved independent native root: {residual}")
    solids = np.r_[ol, px if phase == "clinopyroxene" else np.zeros(3), 0.,
                    px if phase == "orthopyroxene" else np.zeros(3), 0.]
    return {"T_K": request["T_K"], "P_Pa": request["P_Pa"], "initial_oxide_moles": oxides,
            "assemblage": ["liquid", "olivine", phase], "liquid_mol": liquid.tolist(),
            "solid_mol": solids.tolist(), "g_J": g,
            "max_coexistence_residual_J_mol": float(max(abs(residual)))}


def worker(runtime, T, P, request):
    native.check_runtime(runtime)
    sys.path.insert(0, str(runtime))
    from meltsdynamic import MELTSdynamic

    engine = MELTSdynamic(1).engine
    engine.temperature, engine.pressure = T-273.15, P/1e5
    if "phase" in request:
        return {"restricted_equilibria": [assemblage_root(engine, request)]}
    rows = []
    for phase in ("clinopyroxene", "orthopyroxene"):
        for amounts in COMPOSITIONS:
            n = np.array(amounts)
            oxide = np.zeros(19)
            oxide[[0, 5, 7, 10]] = [sum(n), n[2], n[1], n[0]]
            grams = oxide*native.OXIDE_MASSES
            engine.calcPhaseProperties(phase, grams.tolist())
            if engine.status.failed:
                raise RuntimeError(f"Failed native {phase} properties.")
            row = {"phase": phase, "T_K": T, "P_Pa": P, "n_mol": amounts}
            g = float(engine.g[phase])
            if not np.isfinite(g):
                rows.append({**row, "status": "unavailable_nonfinite_native_endpoint"})
                continue
            returned = float(engine.mass[phase])*np.array(engine.dispComposition[phase])/100
            native.require_match(returned, grams, "unchanged pyroxene oxides", atol=1e-9)
            row["g_J"] = g
            # The pinned runtime has singular ordering derivatives at exact edges.
            # Only interior S, Cp, V and mu are independent derivative references.
            if min(n) > 0 and n[0] < n[1]+n[2]:
                row.update(s_J_K=float(engine.s[phase]), cp_J_K=float(engine.cp[phase]),
                           v_m3=float(engine.v[phase])*1e-6)
                engine.calcEndMemberProperties(phase, grams.tolist())
                if engine.status.failed:
                    raise RuntimeError(f"Failed native {phase} chemical potentials.")
                expected = np.r_[NATIVE_MAP@n, np.zeros(4)]/(sum(n)/2)
                native.require_match(engine.X[phase], expected, "signed native fractions", atol=1e-12)
                mu = NATIVE_MAP.T @ np.asarray(engine.mu[phase])[:3]
                native.require_match(n@mu, g, "pyroxene Euler identity", atol=1e-7)
                row["mu_J_mol"] = mu.tolist()
                row["native_fractions"] = expected.tolist()
            rows.append(row)
    return {"properties": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    if args.worker:
        request = json.loads(args.worker.read_text())
        data = worker(runtime, request["T_K"], request["P_Pa"], request)
    else:
        native.check_runtime(runtime)
        previous = json.loads((ROOT / "tests/reference/dry_magma_melts_v1.json").read_text())
        data = {
            "backend": native.REFERENCE["backend"],
            "source_revision": previous["source_revision"],
            "source_sha256": {**previous["source_sha256"], "sources/orthopyroxene.c":
                              "cddc4cc48d3a829724a61589a33e96f0b909189ece1a045c6c64d775754596f7"},
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "component_order": ["CaSiO3", "MgSiO3", "FeSiO3"],
            "method": "Actual supplied MELTS properties and independent roots for specified liquid/olivine/pyroxene assemblages; no JAX imports or full MELTS phase selection.",
            "edge_derivatives": "Not retained: native ordering Cp is singular at exact composition edges.",
            "properties": [], "restricted_equilibria": [],
        }
        requests = [{"T_K": T, "P_Pa": P} for T in (1200., 1600., 2200.) for P in (1e5, 5e7, 5e8)]
        requests += [{"T_K": 1475., "P_Pa": 1e5, "phase": "clinopyroxene",
                      "oxides": {"SiO2": 1., "MgO": .55, "FeO": .15, "CaO": .25, "Al2O3": .12}}]
        requests += [{"T_K": T, "P_Pa": 1e5, "phase": "orthopyroxene", "oxides": LOW_CA_OXIDES}
                     for T in (1800., 1700., 1600.)]
        for request_data in requests:
            T, P = request_data["T_K"], request_data["P_Pa"]
            with tempfile.TemporaryDirectory(prefix="exoeos-pyroxene-") as directory:
                directory = Path(directory)
                request, output = directory/"request.json", directory/"output.json"
                request.write_text(json.dumps(request_data))
                env = dict(os.environ)
                env["LD_LIBRARY_PATH"] = str(runtime)+os.pathsep+env.get("LD_LIBRARY_PATH", "")
                result = subprocess.run([
                    args.python, str(Path(__file__).resolve()), "--runtime", str(runtime),
                    "--worker", str(request), "--output", str(output)],
                    cwd=directory, env=env, capture_output=True, text=True, timeout=60)
                log = result.stdout+result.stderr
                if result.returncode or "Iteration exceeded" in log:
                    raise RuntimeError(log)
                for key, rows in json.loads(output.read_text()).items():
                    data[key].extend(rows)
                print(f"Saved {request_data}", flush=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
