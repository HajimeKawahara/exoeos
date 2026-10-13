"""Generate Fe2O3 quadrature or hash-pinned MELTS liquid reference data.

No ExoEOS/JAX thermodynamic code is imported. Without --runtime, integrate
the published heat capacity and volume numerically. With --runtime, save
supplied liquid G, chemical potentials and native thermodynamic properties.
The runtime and tinynumpy are optional, separate regeneration dependencies.
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

from generate_magma_reference import REVISION, SOURCE_HASHES, integral


ROOT = Path(__file__).resolve().parents[2]
ORDER = ["SiO2", "Mg2SiO4", "Fe2SiO4", "CaSiO3", "Al2O3", "Fe2O3"]
INDICES = [0, 7, 5, 10, 2, 3]
TEMPERATURES = (1200., 1479., 1481., 1600., 1895., 2200.)
PRESSURES = (1e5, 5e7, 5e8)
SOURCE = {
    key: SOURCE_HASHES[key] for key in ("sources/gibbs.c", "includes/liq_struct_data.h")
}
SOURCE["includes/param_struct_data_v34.h"] = "ad78f9ac9a59c030aeee240893b00efc034ea81aed4d551abaed770215af7059"


def source_reference():
    """Independent numerical integrals, including the solid lambda anomaly."""
    tr, tc, tm = 298.15, 955., 1895.
    def cp(t):
        return 146.86 - 5.5768e6/t**2 + 5.2563e8/t**3

    def anomaly(t):
        return t*(-7.403e-2 + 27.921e-5*t)**2
    hf = (-822000. + integral(cp, tr, tm) + integral(anomaly, tr, tc)
          + 1287. + tm*60.41)
    sf = (87.40 + integral(lambda t: cp(t)/t, tr, tm)
          + integral(lambda t: anomaly(t)/t, tr, tc) + 1287./tc + 60.41)
    rows = []
    for T in (298.15, 955., *TEMPERATURES):
        for P in PRESSURES:
            dt, dp = T-1673., P/1e5-1.
            def v(p):
                return 4.213 + 9.09e-4*dt + (-2.53e-5+3.1e-8*dt)*(p-1) + 4.4e-10*(p-1)**2/2

            def vt(p):
                return 9.09e-4 + 3.1e-8*(p-1)
            h = hf + integral(lambda t: np.full_like(t, 240.9), tm, T)
            s = sf + integral(lambda t: 240.9/t, tm, T)
            g = h-T*s + integral(v, 1., P/1e5)
            s -= integral(vt, 1., P/1e5)
            rows.append({"T_K": T, "P_Pa": P, "g_J_mol": g, "s_J_mol_K": s,
                         "cp_J_mol_K": 240.9, "v_m3_mol": v(P/1e5)/1e5,
                         "dvdt_m3_mol_K": vt(P/1e5)/1e5,
                         "dvdp_m3_mol_Pa": (-2.53e-5+3.1e-8*dt+4.4e-10*dp)/1e10})
    return {"reference_kind": "published_source_quadrature",
            "method": "96-point Gauss-Legendre integration of Cp, Cp/T, V and dV/dT; solid lambda integral ends at 955 K, with latent heat 1287 J/mol. No JAX or native runtime.",
            "component": "Fe2O3", "states": rows}


def worker(runtime, request):
    sys.path.insert(0, str(ROOT / "examples"))
    import melts_liquid_evaluator as native

    native.check_runtime(runtime)
    sys.path.insert(0, str(runtime))
    from meltsdynamic import MELTSdynamic

    engine = MELTSdynamic(1).engine
    engine.setSystemProperties("Log fO2 Path", "None")
    if engine.status.failed:
        raise RuntimeError("Could not disable oxygen buffering.")
    T, P = request["T_K"], request["P_Pa"]
    engine.temperature, engine.pressure = T-273.15, P/1e5
    rows = []
    for amounts in request["compositions"]:
        n = np.array(amounts)
        raw = np.zeros(19)
        raw[INDICES] = n
        grams = (native.NU.T @ raw)*native.OXIDE_MASSES
        engine.calcPhaseProperties("liquid", grams.tolist())
        if engine.status.failed:
            raise RuntimeError("Native liquid properties failed.")
        returned = float(engine.mass["liquid"])*np.asarray(engine.dispComposition["liquid"])/100
        native.require_match(returned, grams, "unchanged oxide amounts", rtol=0, atol=1e-9)
        row = {"T_K": T, "P_Pa": P, "n_mol": amounts,
               "g_J": float(engine.g["liquid"]), "s_J_K": float(engine.s["liquid"]),
               "cp_J_K": float(engine.cp["liquid"]), "v_m3": float(engine.v["liquid"])*1e-6,
               "dvdt_m3_K": float(engine.dvdt["liquid"])*1e-6,
               "dvdp_m3_Pa": float(engine.dvdp["liquid"])*1e-11}
        engine.calcEndMemberProperties("liquid", grams.tolist())
        if engine.status.failed:
            raise RuntimeError("Native liquid chemical potentials failed.")
        fractions = np.asarray(engine.X["liquid"])
        native.require_match(fractions, raw/raw.sum(), "unchanged native fractions", rtol=0, atol=1e-10)
        mu = np.asarray(engine.mu["liquid"])[INDICES]
        present = n > 0
        native.require_match(n[present]@mu[present], row["g_J"], "Euler identity", rtol=0, atol=5e-5)
        row.update(mu_J_mol=native.nullable(mu, present),
                   standard_J_mol=native.nullable(np.asarray(engine.mu0["liquid"])[INDICES], present),
                   max_oxide_roundtrip_error_g=float(max(abs(returned-grams))),
                   returned_component_fractions=fractions[INDICES].tolist())
        rows.append(row)
    return rows


def backend_reference(runtime, python):
    sys.path.insert(0, str(ROOT / "examples"))
    import melts_liquid_evaluator as native

    native.check_runtime(runtime)
    compositions = [[.4, .275, .075, .25, .12, .025], [.2, .3, .2, .1, .15, .1],
                    [.4, .6, .4, .2, .3, .2], [.4, .275, .075, .25, .12, 1e-6],
                    [.4, .275, .075, .25, .12, 0.], [0., 0., 0., 0., 0., 1.]]
    requests = [{"T_K": T, "P_Pa": P, "compositions": compositions}
                for T in TEMPERATURES for P in PRESSURES]
    # Isolate all five new interactions on binary composition faces.
    pairs = []
    for i in range(5):
        n = np.zeros(6)
        n[i], n[5] = .7, .3
        pairs.append(n.tolist())
    requests.extend({"T_K": 1700., "P_Pa": 5e7, "compositions": [pair]} for pair in pairs)
    rows, unavailable = [], []
    for request_data in requests:
        with tempfile.TemporaryDirectory(prefix="exoeos-ferric-") as directory:
            directory = Path(directory)
            request, output = directory/"request.json", directory/"output.json"
            request.write_text(json.dumps(request_data))
            env = dict(os.environ)
            env["LD_LIBRARY_PATH"] = str(runtime)+os.pathsep+env.get("LD_LIBRARY_PATH", "")
            result = subprocess.run([
                python, str(Path(__file__).resolve()), "--runtime", str(runtime),
                "--worker", str(request), "--output", str(output)],
                cwd=directory, env=env, capture_output=True, text=True, timeout=60)
            log = result.stdout+result.stderr
            if "SiO2 has negative mole fraction" in log and request_data["compositions"] in [[pair] for pair in pairs if pair[0] == 0.]:
                unavailable.append({**request_data, "reason": "Pinned MELTS oxide conversion reports negative SiO2 at this exact composition edge; no perturbed replacement is used."})
                continue
            if result.returncode or "calcPhaseProperties failed" in log:
                raise RuntimeError(log)
            rows.extend(json.loads(output.read_text()))
            print(f"Saved T={request_data['T_K']} K, P={request_data['P_Pa']} Pa", flush=True)
    return {"reference_kind": "pinned_supplied_liquid_properties",
            "backend": native.REFERENCE["backend"], "component_order": ORDER,
            "method": "Fresh process per T/P; calcPhaseProperties and calcEndMemberProperties; oxygen buffering disabled, oxide amounts and fractions checked unchanged. No equilibrium solve or ExoEOS/JAX imports.",
            "units": "Native V and dV/dT in cm3 converted by 1e-6; native dV/dP in cm3/bar converted by 1e-11. Energies and heat capacities are extensive for supplied mol amounts.",
            "python_version": sys.version, "numpy_version": np.__version__,
            "tinynumpy_version": version("tinynumpy"), "states": rows,
            "unavailable_cases": unavailable}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        data = worker(args.runtime.resolve(), json.loads(args.worker.read_text()))
    else:
        data = backend_reference(args.runtime.resolve(), args.python) if args.runtime else source_reference()
        data.update(source_revision=REVISION, source_sha256=SOURCE,
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
