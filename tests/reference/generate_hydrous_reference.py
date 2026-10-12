"""Save independent compiled-MAGMA standards/pairs and pinned MELTS liquids.

No ExoEOS or JAX import. Source mode requires a C compiler and the five
hash-pinned source files; runtime mode requires the existing pinned binary
and tinynumpy. Ordinary tests read the saved JSON only.
"""

import argparse
import ctypes
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

from generate_dry11_reference import (
    ORDER as DRY_ORDER, OXIDES as DRY_OXIDES, INDICES as DRY_INDICES,
    SOURCE as DRY_SOURCE, REVISION, native_module, compositions,
)


ORDER, OXIDES, INDICES = DRY_ORDER+["H2O"], DRY_OXIDES+["H2O"], DRY_INDICES+[18]
SOURCE = {**DRY_SOURCE,
          "sources/water.c": "5ee8b5d59eb813c50fd66d2a9b47b317a0cde327c6ac924967d2521f770a8b08",
          "sources/liquid_v34.c": "584ac446d03abf64ca7406768ad84975e1af346d159bc743c9adafd9441e91f6"}
FIELDS = ("g_J", "s_J_K", "cp_J_K", "v_m3", "dvdt_m3_K", "dvdp_m3_Pa")


def source_reference(source):
    for name, digest in SOURCE.items():
        if hashlib.sha256((source/Path(name).name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source hash mismatch: {name}")
    header = (source/"param_struct_data_v34.h").read_text().split("};", 1)[0]
    pairs = []
    for a, b, h, s, v in re.findall(r'"W\(([^,]+),(.+)\)"\s*,\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)', header):
        if a.strip() == "H2O" and b.strip() in DRY_ORDER:
            pairs.append({"component": b.strip(), "H_J_mol": float(h),
                          "S_J_mol_K": float(s), "V_J_mol_bar": float(v)})
    assert len(pairs) == 11
    # Compile the actual upstream whaar plus the H2O branch of gibbs.c.
    # Only the include shim and zero tabulated H/S references are supplied.
    branch = (source/"gibbs.c").read_text().split("double a = -33676.0", 1)[1]
    branch = ("double a = -33676.0"+branch.split("/* special case - no EOS option */", 1)[0]).rsplit("}", 1)[0]
    branch = branch.replace("phase->h", "0.0").replace("phase->s", "0.0")
    shim = """#include <math.h>
#include <float.h>
#include <stdio.h>
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
#define ABS(a) fabs(a)
#define MODE__MELTS 1
#define MODE_xMELTS 2
#define MODE__MELTSandCO2 3
#define MODE__MELTSandCO2_H2O 4
static int calculationMode = MODE__MELTS;
"""
    wrapper = """
#define CUBE(x) ((x)*(x)*(x))
void standard(double t, double p, double *out) {
  const double r = 8.3143;
  double gl, hl, sl, cpl, dcpldt, vl, dvldt, dvldp, d2vldt2, d2vldp2, d2vldtdp;
""" + branch + """
  out[0]=gl; out[1]=sl; out[2]=cpl;
  out[3]=vl/1e5; out[4]=dvldt/1e5; out[5]=dvldp/1e10;
}
"""
    rows = []
    with tempfile.TemporaryDirectory(prefix="hydrous-source-") as directory:
        directory = Path(directory)
        (directory/"silmin.h").write_text(shim)
        (directory/"reference.c").write_text((source/"water.c").read_text()+wrapper)
        command = ["cc", "-shared", "-fPIC", "-O2", str(directory/"reference.c"), "-lm", "-o", str(directory/"reference.so")]
        subprocess.run(command, check=True, capture_output=True, text=True)
        library = ctypes.CDLL(str(directory/"reference.so"))
        library.standard.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        library.standard.restype = None
        for t in (700., 1000., 1200., 1479., 1480., 1481., 1800., 2200., 2500.):
            for p in (1e5, 5e7, 5e8):
                out = (ctypes.c_double*6)()
                library.standard(t, p/1e5, out)
                rows.append({"T_K": t, "P_Pa": p, **dict(zip(FIELDS, out))})
    return {"reference_kind": "compiled_upstream_H2O_standard",
            "method": "Unmodified hash-checked water.c plus extracted MELTS H2O branch of gibbs.c; zero tabulated H/S. Pair coefficients parsed from header. No ExoEOS/JAX imports.",
            "compiler": subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
            "states": rows, "pairs": pairs}


def worker(runtime, request):
    native = native_module()
    native.check_runtime(runtime)
    sys.path.insert(0, str(runtime))
    from meltsdynamic import MELTSdynamic
    engine = MELTSdynamic(1).engine
    engine.setSystemProperties("Log fO2 Path", "None")
    if engine.status.failed:
        raise RuntimeError("Could not disable oxygen buffering.")
    t, p = request["T_K"], request["P_Pa"]
    engine.temperature, engine.pressure = t-273.15, p/1e5
    rows = []
    for sample in request["samples"]:
        n = np.array(sample["n_mol"])
        raw = np.zeros(19)
        raw[INDICES] = n
        grams = (native.NU.T@raw)*native.OXIDE_MASSES
        engine.calcPhaseProperties("liquid", grams.tolist())
        if engine.status.failed or engine.mass is None:
            raise RuntimeError("Native liquid properties failed.")
        returned = float(engine.mass["liquid"])*np.asarray(engine.dispComposition["liquid"])/100
        native.require_match(returned, grams, "unchanged oxide inventory", rtol=2e-11, atol=1e-10)
        values = [float(getattr(engine, key)["liquid"])*scale
                  for key, scale in zip(("g", "s", "cp", "v", "dvdt", "dvdp"), (1, 1, 1, 1e-6, 1e-6, 1e-11))]
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Nonfinite native liquid properties.")
        engine.calcEndMemberProperties("liquid", grams.tolist())
        if engine.status.failed:
            raise RuntimeError("Native liquid chemical potentials failed.")
        fractions = np.asarray(engine.X["liquid"])
        native.require_match(fractions, raw/raw.sum(), "unchanged fractions", rtol=0, atol=1e-10)
        mu, mu0 = np.asarray(engine.mu["liquid"])[INDICES], np.asarray(engine.mu0["liquid"])[INDICES]
        present = n > 0
        native.require_match(n[present]@mu[present], values[0], "Euler identity", rtol=0, atol=5e-5*max(1., n.sum()))
        rows.append({**sample, "T_K": t, "P_Pa": p, **dict(zip(FIELDS, values)),
                     "mu_J_mol": native.nullable(mu, present),
                     "standard_J_mol": native.nullable(mu0, present),
                     "returned_component_fractions": fractions[INDICES].tolist()})
    return rows


def backend_reference(runtime, python):
    native = native_module()
    native.check_runtime(runtime)
    recipes = []
    for dry in compositions(native)[:3]:
        for weight in (.1, 2., 5.):
            n = np.r_[np.array(dry["n_mol"])*(100-weight)/100, weight/native.OXIDE_MASSES[14]]
            recipes.append({"name": f"{dry['name']}_{weight:g}wt", "water_wt_percent": weight, "n_mol": n.tolist()})
    requests = [{"T_K": t, "P_Pa": p, "samples": recipes}
                for t in (1200., 1479., 1481., 1800., 2200.) for p in (1e5, 5e7, 5e8)]
    rng = np.random.default_rng(20261012)
    samples = [{"name": f"random_{i}", "n_mol": n.tolist()}
               for i, n in enumerate(rng.dirichlet(np.ones(12), size=5))]
    samples.append({"name": "scaled_basalt", "n_mol": (7*np.array(recipes[4]["n_mol"])).tolist()})
    requests.append({"T_K": 1700., "P_Pa": 5e7, "samples": samples})
    for i in range(11):
        n = np.zeros(12)
        n[i], n[-1] = .7, .3
        requests.append({"T_K": 1700., "P_Pa": 5e7,
                         "samples": [{"name": f"water_pair_{i}", "n_mol": n.tolist()}]})
    base = np.array(recipes[4]["n_mol"])
    for water in (0., 1e-8, .5):
        n = base.copy()
        n[-1] = water
        requests.append({"T_K": 1700., "P_Pa": 5e7,
                         "samples": [{"name": f"water_limit_{water:g}", "n_mol": n.tolist()}]})
    requests.append({"T_K": 1700., "P_Pa": 5e7,
                     "samples": [{"name": "pure_water", "n_mol": [0.]*11+[1.]}]})
    rows, unavailable = [], []
    for request in requests:
        with tempfile.TemporaryDirectory(prefix="hydrous-native-") as directory:
            directory = Path(directory)
            source, output = directory/"request.json", directory/"output.json"
            source.write_text(json.dumps(request))
            env = dict(os.environ)
            env["LD_LIBRARY_PATH"] = str(runtime)+os.pathsep+env.get("LD_LIBRARY_PATH", "")
            result = subprocess.run([python, str(Path(__file__).resolve()), "--runtime", str(runtime),
                                     "--worker", str(source), "--output", str(output)],
                                    cwd=directory, env=env, capture_output=True, text=True, timeout=60)
            log = result.stdout+result.stderr
            negative = re.search(r"Liquid component (\S+) has negative mole fraction", log)
            boundary = request["samples"][0]["name"]
            if negative and boundary.startswith("water_pair_"):
                unavailable.append({**request, "reason": f"Native oxide conversion gives negative {negative[1]}; no perturbation used."})
                continue
            if boundary == "pure_water" and "Nonfinite native liquid properties" in log:
                unavailable.append({**request, "reason": "Native mixing evaluates 0*log(0) at pure water; nonfinite G/S. The compiled standard is saved separately."})
                continue
            if result.returncode or not output.exists():
                raise RuntimeError(log)
            rows.extend(json.loads(output.read_text()))
    return {"reference_kind": "pinned_supplied_hydrous_liquid_properties", "backend": native.REFERENCE["backend"],
            "method": "Fresh process per T/P or boundary; supplied properties, oxygen buffering disabled. Oxide inventory, fractions and Euler identity verified; no equilibrium solve.",
            "composition_note": "100 g complete liquid: dry BSE or synthetic basalt/rhyolite plus 0.1, 2, 5 mass percent H2O. Fixed Fe(II)/Fe(III).",
            "python_version": sys.version, "tinynumpy_version": version("tinynumpy"),
            "recipes": recipes, "states": rows, "unavailable_cases": unavailable}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        result = worker(args.runtime.resolve(), json.loads(args.worker.read_text()))
    else:
        if not args.runtime and not args.source:
            parser.error("supply --source or --runtime")
        result = backend_reference(args.runtime.resolve(), args.python) if args.runtime else source_reference(args.source)
        result.update(component_order=ORDER, oxide_order=OXIDES, source_revision=REVISION, source_sha256=SOURCE,
                      generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
