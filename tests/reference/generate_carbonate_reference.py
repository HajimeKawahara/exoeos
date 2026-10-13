"""Save independent compiled-source, ZD09 and pinned mode-4 references.

Regeneration uses NumPy/SciPy, a C compiler and optionally alphaMELTS.
No JAX or ExoEOS thermodynamic implementation is imported. Normal tests
read the saved JSON offline. Download the listed sources at REVISION first.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from generate_dry11_reference import ORDER as DRY_ORDER, INDICES as DRY_INDICES, native_module
from generate_magma_fluid_reference import COEFFICIENTS, R, RH, first, second


REVISION = "705a0fb315e5054d18275a580562f6121c8e458c"
SOURCE = {
    "gibbs.c": "2eceff9f315210ebdcf9dfc9bacd77f625cf04dbdc5084104c093caba34d4166",
    "water.c": "5ee8b5d59eb813c50fd66d2a9b47b317a0cde327c6ac924967d2521f770a8b08",
    "fluidPhase.c": "2aa11edc637c2aadc7722c8a606ce6e7ccde87a59add5c7f8d9abefdb8048cc0",
    "liquid_CO2_H2O.c": "70fcc1b74785a45708c88c634f41a27590eb3619b493775e2d753470e36ad081",
    "param_struct_data_CO2_H2O.h": "05b650b1c6b978e9269f1401848243948e5d79c27d7de87cbf6a3931b121e42e",
    "liq_struct_data.h": "206b298d054d9110f28dd544dab1240aa252bb39650bcdf21e032fee0e7d95b4",
}
ORDER, INDICES = DRY_ORDER+["H2O", "CO2"], DRY_INDICES+[18, 14]
FIELDS = ("g_J", "s_J_K", "cp_J_K", "v_m3", "dvdt_m3_K", "dvdp_m3_Pa")


def zd_state(t, p, x):
    """Published ZD09 mixing and Z; independent root and pressure integral."""
    epsilon = x@np.array([[510., .85*np.sqrt(510*235)], [.85*np.sqrt(510*235), 235.]])@x
    diameter = x@np.array([[2.88, 1.02*(2.88+3.79)/2], [1.02*(2.88+3.79)/2, 3.79]])@x
    scale, tr = (diameter/3.691)**3/1000, 154*t/epsilon
    b, c, d, e = [a[0]+a[1]/tr**2+a[2]/tr**3 for a in COEFFICIENTS[:12].reshape(4, 3)]
    f, beta, gamma = COEFFICIENTS[12:]

    def zm1(q):
        return b*q+c*q*q+d*q**4+e*q**5+f/tr**3*q*q*(beta+gamma*q*q)*np.exp(-gamma*q*q)

    def residual(q):
        return q*(1+zm1(q))*R*t/scale-p

    grid = np.r_[0., np.geomspace(1e-9, 128., 1500)]
    i = np.flatnonzero([residual(q) >= 0 for q in grid])[0]
    q = brentq(residual, grid[i-1], grid[i], xtol=1e-15)
    ar = quad(lambda s: zm1(s)/s if s else b, 0., q, epsabs=1e-12)[0]
    return q/scale, ar+zm1(q)-np.log1p(zm1(q))


def source_reference(source):
    for name, digest in SOURCE.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Pinned source hash mismatch: {name}")
    header = (source/"param_struct_data_CO2_H2O.h").read_text()
    pairs = []
    for a, b, h, s, v in re.findall(r'"W\(([^,]+),(.+)\)"\s*,\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)', header):
        a, b = a.strip(), b.strip()
        if a in ORDER+["CaCO3"] and b in ORDER+["CaCO3"]:
            pairs.append({"components": [a, b], "H_J_mol": float(h), "S_J_mol_K": float(s), "V_J_mol_bar": float(v)})
    assert len(pairs) == 91
    gibbs = (source/"gibbs.c").read_text()
    water = gibbs.split('} else if( (strcmp(name, "H2O") == 0) && ((calculationMode == MODE__MELTSandCO2_H2O)', 1)[1]
    water = water[water.index('double a ='):].split('/* special case - no EOS option */', 1)[0]
    water = water.replace('phase->h', '0.0').replace('phase->s', '0.0')
    co2 = gibbs.split('double hCO2       =', 1)[1].split('/* special case - no EOS option */', 1)[0]
    co2 = 'double hCO2       ='+co2
    shim = """#include <math.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
#define ABS(a) fabs(a)
#define MODE__MELTS 1
#define MODE_xMELTS 2
#define MODE__MELTSandCO2 3
#define MODE__MELTSandCO2_H2O 4
static int calculationMode = MODE__MELTSandCO2_H2O;
"""
    declarations = """
  const double r=8.3143, trl=1673.0, pr=1.0;
  double gl,hl,sl,cpl,dcpldt,vl,dvldt,dvldp,d2vldt2,d2vldp2,d2vldtdp;
"""
    output = "out[0]=gl;out[1]=sl;out[2]=cpl;out[3]=vl/1e5;out[4]=dvldt/1e5;out[5]=dvldp/1e10;"
    # The original source's private pure-fluid functions remain unmodified.
    fluid = (source/"fluidPhase.c").read_text().split('\nint\ntestFlu', 1)[0]
    fluid += '\nvoid co2_ideal(double t,double p,double *out) { double h,s,cp,dcp; idealGasCO2(t,&cp,&s,&h,&dcp); out[0]=h-t*s;out[1]=s;out[2]=cp; }\n'
    fluid += 'void co2_standard(double t,double p,double *out) {'+declarations+co2+output+'}\n'
    water_source = (source/"water.c").read_text()
    water_source += '#define CUBE(x) ((x)*(x)*(x))\n#define QUARTIC(x) ((x)*(x)*(x)*(x))\n#define QUINTIC(x) ((x)*(x)*(x)*(x)*(x))\n'
    water_source += 'void water_standard(double t,double p,double *out) {'+declarations+water+output+'}\n'
    water_source += 'void water_gas(double t,double p,double *out) { double h,dcp,v,vt,vp,vtt,vtp,vpp; whaar(p,t,out,&h,out+1,out+2,&dcp,&v,&vt,&vp,&vtt,&vtp,&vpp); }\n'
    rows, fluid_rows = [], []
    with tempfile.TemporaryDirectory(prefix="carbonate-source-") as directory:
        directory = Path(directory)
        (directory/"silmin.h").write_text(shim)
        (directory/"fluid.c").write_text(fluid)
        (directory/"water.c").write_text(water_source)
        subprocess.run(["cc", "-shared", "-fPIC", "-O2", str(directory/"fluid.c"), str(directory/"water.c"), "-lm", "-o", str(directory/"ref.so")], check=True)
        library = ctypes.CDLL(str(directory/"ref.so"))

        def call(name, t, p=1.):
            fun = getattr(library, name)
            fun.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
            fun.restype = None
            out = (ctypes.c_double*6)()
            fun(t, p, out)
            return np.array(out)

        for t in (700., 1000., 1200., 1479., 1480., 1481., 1700., 2200., 2500.):
            for p in (1e5, 5e7, 5e8):
                for component, name in (("H2O", "water_standard"), ("CO2", "co2_standard")):
                    rows.append({"component": component, "T_K": t, "P_Pa": p, **dict(zip(FIELDS, call(name, t, p/1e5)))})
            rows.append({"component": "CO2_gas", "T_K": t, "P_Pa": 1e5, **dict(zip(FIELDS[:3], call("co2_ideal", t)[:3]))})
        for t in (1200., 1700., 2200.):
            water_limits = [call("water_gas", t, p)[:3]+[-RH*t*np.log(p), RH*np.log(p), 0.] for p in (.001, .0005)]
            standards = np.array([2*water_limits[1]-water_limits[0], call("co2_ideal", t)[:3]])
            for p in (1e5, 5e7, 5e8):
                for xw in (0., .1, .5, .9, 1.):
                    x = np.array([xw, 1-xw])
                    ideal = sum(xi*np.log(xi) for xi in x if xi)
                    def correction(ti):
                        return R*ti*(ideal+np.log(p/1e5)+zd_state(ti, p, x)[1])
                    def volume(ti, pi):
                        return 1/zd_state(ti, pi, x)[0]
                    thermal = x@standards+[correction(t), -first(correction, t, .5), -t*second(correction, t, .5)]
                    values = [*thermal, volume(t, p), first(lambda ti: volume(ti, p), t, .5), first(lambda pi: volume(t, pi), p, p*1e-4)]
                    mu = []
                    for i in range(2):
                        if x[i] == 0:
                            mu.append(None)
                            continue
                        def energy(ni):
                            n = x.copy(); n[i] = ni
                            y = n/n.sum()
                            mixing = sum(ni*np.log(yi) for ni, yi in zip(n, y) if ni)
                            return n@standards[:, 0]+R*t*(mixing+n.sum()*(np.log(p/1e5)+zd_state(t, p, y)[1]))
                        mu.append(first(energy, x[i], x[i]*1e-3))
                    fluid_rows.append({"T_K": t, "P_Pa": p, "n_mol": x.tolist(), "mu_J_mol": mu, **dict(zip(FIELDS, values))})
    return {"reference_kind": "compiled_mode4_standards_and_independent_ZD09_quadrature", "pairs": pairs, "standards": rows, "fluid_states": fluid_rows,
            "method": "Unmodified hash-checked pure-fluid C and extracted gibbs.c branches. ZD09 Eq.8/Tables3-4 mixing in NumPy, SciPy brentq and integral of (Z-1)/density. Five-point derivatives, 0.5 K/0.0001 P/0.001 n. Haar gas standards extrapolate corrected 0.001/0.0005 bar values. No JAX/ExoEOS imports."}


def worker(runtime, request):
    native = native_module()
    native.check_runtime(runtime)
    sys.path.insert(0, str(runtime))
    from meltsdynamic import MELTSdynamic
    model = MELTSdynamic(4)
    engine = model.engine
    assert model.endMemberFormulas["liquid"] == native.COMPONENTS+["caco3"]
    engine.setSystemProperties("Log fO2 Path", "None")
    if engine.status.failed:
        raise RuntimeError("Could not disable oxygen buffering.")
    t, p = request["T_K"], request["P_Pa"]
    engine.temperature, engine.pressure = t-273.15, p/1e5
    rows = []
    for sample in request["samples"]:
        n = np.array(sample["n_mol"])
        raw = np.zeros(19); raw[INDICES] = n
        grams = (native.NU.T@raw)*native.OXIDE_MASSES
        engine.calcPhaseProperties("liquid", grams.tolist())
        if engine.status.failed:
            raise RuntimeError("Native liquid properties failed.")
        returned = float(engine.mass["liquid"])*np.asarray(engine.dispComposition["liquid"])/100
        native.require_match(returned, grams, "unchanged oxide inventory", rtol=2e-11, atol=1e-10)
        values = [float(getattr(engine, k)["liquid"])*s for k, s in zip(("g", "s", "cp", "v", "dvdt", "dvdp"), (1, 1, 1, 1e-6, 1e-6, 1e-11))]
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Nonfinite native properties.")
        engine.calcEndMemberProperties("liquid", grams.tolist())
        if engine.status.failed:
            raise RuntimeError("Native chemical potentials failed.")
        species = np.asarray(engine.X["liquid"])
        native.require_match(native.independent_fractions(species, 4), raw/raw.sum(), "independent fractions", atol=1e-10)
        mu, mu0 = (np.asarray(getattr(engine, field)["liquid"])[INDICES] for field in ("mu", "mu0"))
        present = n > 0
        native.require_match(n[present]@mu[present], values[0], "Euler identity", rtol=0, atol=5e-5*max(1., n.sum()))
        rows.append({**sample, "T_K": t, "P_Pa": p, **dict(zip(FIELDS, values)), "mu_J_mol": native.nullable(mu, present),
                     "standard_J_mol": native.nullable(mu0, present), "species_mol": (species[INDICES+[19]]*n.sum()).tolist()})
    return rows


def backend_reference(runtime, python):
    native = native_module(); native.check_runtime(runtime)
    # Positive, oxide-representable controls with contrasting Ca/alkali hosts.
    base = np.array([.5,.15,.05,.15,.1,.01,.01,.002,.02,.01,.002,.05,.001])
    recipes = [{"name": name, "n_mol": n.tolist()} for name, n in (
        ("basalt", base), ("alkali_rich", base*np.array([1,.2,.2,.2,1,1,1,1,5,3,1,2,3])),
        ("carbon_rich", np.r_[base[:12], .1]))]
    requests = [{"T_K": t, "P_Pa": p, "samples": recipes} for t in (1200., 1479., 1481., 1700., 2200.) for p in (1e5, 5e7, 5e8)]
    rng = np.random.default_rng(20261013)
    samples = [{"name": f"random_{i}", "n_mol": n.tolist()} for i, n in enumerate(rng.dirichlet(np.ones(13), size=4))]
    for name, n in (("scaled", 7*base), ("zero_carbon", np.r_[base[:12], 0.]), ("dry", np.r_[base[:11], 0., 0.]),
                    ("trace_carbon", np.r_[base[:12], 1e-9])):
        samples.append({"name": name, "n_mol": n.tolist()})
    requests.append({"T_K": 1700., "P_Pa": 5e7, "samples": samples})
    requests.append({"T_K": 1700., "P_Pa": 5e7, "samples": [{"name": "zero_calcium",
                     "n_mol": (base*np.array([1,1,1,0,1,1,1,1,1,1,1,1,1])).tolist()}]})
    rows, unavailable = [], []
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for request in requests:
            source, output = directory/"request.json", directory/"result.json"
            source.write_text(json.dumps(request))
            env = dict(os.environ); env["LD_LIBRARY_PATH"] = str(runtime)+os.pathsep+env.get("LD_LIBRARY_PATH", "")
            result = subprocess.run([python, str(Path(__file__).resolve()), "--runtime", str(runtime), "--worker", str(source), "--output", str(output)], cwd=directory, env=env, capture_output=True, text=True, timeout=60)
            if request["samples"][0]["name"] == "zero_calcium" and ("negative mole fraction" in result.stdout or "SIGABRT" in result.stdout+result.stderr):
                unavailable.append({**request, "reason": "Native oxide conversion produces negative roundoff CaSiO3 and aborts; input is not perturbed."})
                continue
            if result.returncode or not output.exists():
                raise RuntimeError(result.stdout+result.stderr)
            rows.extend(json.loads(output.read_text()))
    backend = {**native.REFERENCE["backend"], "model": "rhyolite-MELTS 1.2.0", "calculation_mode": 4}
    return {"reference_kind": "pinned_supplied_mode4_liquid", "backend": backend, "calculation_mode": 4,
            "model": "rhyolite-MELTS 1.2.0", "recipes": recipes, "states": rows, "unavailable_cases": unavailable,
            "method": "Fresh mode-4 process per T/P; supplied compositions, oxygen buffering disabled. Returned oxide inventories, independent fractions and Euler identity checked. No phase equilibration."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.worker:
        result = worker(args.runtime.resolve(), json.loads(args.worker.read_text()))
    else:
        if not args.source and not args.runtime:
            parser.error("supply --source or --runtime")
        result = source_reference(args.source) if args.source else backend_reference(args.runtime.resolve(), args.python)
        result.update(component_order=ORDER, source_revision=REVISION, source_sha256=SOURCE,
                      generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
