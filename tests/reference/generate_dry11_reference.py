"""Generate independent dry eleven-component source or pinned MELTS fixtures.

No ExoEOS/JAX thermodynamic code is imported. Source mode numerically
integrates Cp, Cp/T and V and extracts all 55 pairs from the pinned header.
Runtime mode checks supplied properties without equilibrating iron or phases.
"""

import argparse
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

from generate_ferric_reference import SOURCE
from generate_magma_reference import REVISION, integral


ROOT = Path(__file__).resolve().parents[2]
ORDER = ["SiO2", "Mg2SiO4", "Fe2SiO4", "CaSiO3", "Al2O3", "Fe2O3",
         "TiO2", "MgCr2O4", "Na2SiO3", "KAlSiO4", "Ca3(PO4)2"]
OXIDES = ["SiO2", "MgO", "FeO", "CaO", "Al2O3", "Fe2O3", "TiO2", "Cr2O3", "Na2O", "K2O", "P2O5"]
INDICES = [0, 7, 5, 10, 2, 3, 1, 4, 11, 12, 13]
TEMPERATURES = (1200., 1479., 1480., 1481., 1800., 2200.)
PRESSURES = (1e5, 5e7, 5e8)
# Href, Sref, Berman Cp[4], Tm, Sf, CpL, Kress V[5], transition[Tc,Hc,l1,l2].
STANDARDS = [
    (-944750., 50.460, (77.84, 0., -33.678e5, 40.294e7),
     1870., 35.824, 109.2, (2.316, 7.246e-4, -2.310e-5, 0., 5e-10), None),
    (-1783640., 106.02, (201.981, -551.9, -57.844e5, 57.729e7),
     2673.15, 73.22, 335.1, (5.358, 11.71e-4, -2.26e-5, 1.8e-8, 4.67e-10), None),
    (-373190.*4.184, 27.21*4.184, (234.77, -2218.9, 0., 13.530e7),
     1361., 38.34, 180.2, (5.568, 7.41e-4, -4.29e-5, -5.3e-8, 8.4e-10), None),
    (-2111813.55, 133.9653, (186., 0., -131.067e5, 213.893e7),
     2023.15, 24.5, 217., (6.8375, 7.265e-4, -6.395e-5, -4.6e-8, 12.1e-10),
     (800.15, 1154., -7.096454e-2, 21.682e-5)),
    (-4097169., 235.978, (402.997, -2808.35, 0., -32.6230e7),
     1943.15, 35.690, 574.67, (10.7382, 0., 0., 0., 0.),
     (1373., 14059., 2.5427e-2, 19.255e-5)),
]


def source_reference(source):
    for name, digest in SOURCE.items():
        if hashlib.sha256((source/Path(name).name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source hash mismatch: {name}")
    header = (source/"param_struct_data_v34.h").read_text().split("};", 1)[0]
    pairs = []
    for a, b, h, s, v in re.findall(r'"W\(([^,]+),(.+)\)"\s*,\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)', header):
        a, b = a.strip(), b.strip()
        if a in ORDER and b in ORDER:
            pairs.append({"components": [a, b], "H_J_mol": float(h),
                          "S_J_mol_K": float(s), "V_J_mol_bar": float(v)})
    assert len(pairs) == 55
    rows = []
    temperatures = sorted(set((298.15, 800.15, 1373., *TEMPERATURES, *(s[3] for s in STANDARDS))))
    for name, (hr, sr, coefficients, tm, sf, cpl, eos, transition) in zip(ORDER[6:], STANDARDS):
        k0, k1, k2, k3 = coefficients
        def cp(t):
            return k0+k1/np.sqrt(t)+k2/t**2+k3/t**3
        hf = hr+integral(cp, 298.15, tm)+tm*sf
        ss = sr+integral(lambda t: cp(t)/t, 298.15, tm)+sf
        if transition:
            tc, latent, l1, l2 = transition
            hf += integral(lambda t: t*(l1+l2*t)**2, 298.15, tc)+latent
            ss += integral(lambda t: (l1+l2*t)**2, 298.15, tc)+latent/tc
        v0, vt, vp, vtp, vpp = eos
        for T in temperatures:
            for P in PRESSURES:
                dt, dp = T-1673., P/1e5-1.
                def volume(p):
                    return v0+vt*dt+(vp+vtp*dt)*(p-1)+vpp*(p-1)**2/2

                def volume_t(p):
                    return vt+vtp*(p-1)
                h = hf+integral(lambda t: np.full_like(t, cpl), tm, T)
                s = ss+integral(lambda t: cpl/t, tm, T)
                rows.append({"component": name, "T_K": T, "P_Pa": P,
                             "g_J_mol": h-T*s+integral(volume, 1., P/1e5),
                             "s_J_mol_K": s-integral(volume_t, 1., P/1e5),
                             "cp_J_mol_K": cpl, "v_m3_mol": volume(P/1e5)/1e5,
                             "dvdt_m3_mol_K": volume_t(P/1e5)/1e5,
                             "dvdp_m3_mol_Pa": (vp+vtp*dt+vpp*dp)/1e10})
    return {"reference_kind": "published_source_quadrature",
            "method": "96-point Gauss-Legendre Cp, Cp/T, V and dV/dT integrals; solid transitions included before fusion. All 55 pairs parsed from hash-checked header. No ExoEOS/JAX or runtime.",
            "pairs": pairs, "states": rows}


def native_module():
    sys.path.insert(0, str(ROOT/"examples"))
    import melts_liquid_evaluator
    return melts_liquid_evaluator


def compositions(native):
    oxide_indices = [native.OXIDES.index(name.lower()) for name in OXIDES]
    transform = native.NU[np.ix_(INDICES, oxide_indices)].T
    bse = json.loads((ROOT/"examples/m2_material/bse_liquid_validation.json").read_text())["input"]["ledger"]
    weights = dict(zip(bse["oxide_order"], bse["oxide_mass_fractions"]))
    # Non-BSE controls are declared synthetic oxide recipes, not natural samples.
    recipes = {
        "BSE": [100*weights[name] for name in OXIDES],
        "basalt_like": [49., 8., 8., 11., 15., 2., 2., .1, 3., 1., .9],
        "rhyolite_like": [73., .5, 1.5, 1.5, 14., .5, .3, .01, 4., 4.5, .19],
        "alkali_rich": [52., 4., 5., 6., 18., 2., 1., .1, 6., 5., .9],
    }
    rows = []
    for name, weights in recipes.items():
        grams = np.array(weights)/sum(weights)*100
        oxides = grams/native.OXIDE_MASSES[oxide_indices]
        n = np.linalg.solve(transform, oxides)
        assert np.all(n >= 0)
        rows.append({"name": name, "oxide_g_per_100g": grams.tolist(),
                     "oxide_mol": oxides.tolist(), "n_mol": n.tolist()})
    return rows


def worker(runtime, request):
    native = native_module()
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
    for sample in request["samples"]:
        n = np.array(sample["n_mol"])
        raw = np.zeros(19)
        raw[INDICES] = n
        grams = (native.NU.T@raw)*native.OXIDE_MASSES
        engine.calcPhaseProperties("liquid", grams.tolist())
        if engine.status.failed or engine.mass is None:
            raise RuntimeError("Native liquid properties failed.")
        returned = float(engine.mass["liquid"])*np.asarray(engine.dispComposition["liquid"])/100
        native.require_match(returned, grams,
                             f"unchanged oxides ({sample['name']}; max error {max(abs(returned-grams)):.3g} g)",
                             rtol=2e-11, atol=1e-10)
        row = {**sample, "T_K": T, "P_Pa": P,
               "g_J": float(engine.g["liquid"]), "s_J_K": float(engine.s["liquid"]),
               "cp_J_K": float(engine.cp["liquid"]), "v_m3": float(engine.v["liquid"])*1e-6,
               "dvdt_m3_K": float(engine.dvdt["liquid"])*1e-6,
               "dvdp_m3_Pa": float(engine.dvdp["liquid"])*1e-11}
        engine.calcEndMemberProperties("liquid", grams.tolist())
        if engine.status.failed:
            raise RuntimeError("Native liquid chemical potentials failed.")
        fractions = np.asarray(engine.X["liquid"])
        native.require_match(fractions, raw/raw.sum(), "unchanged fractions", rtol=0, atol=1e-10)
        present = n > 0
        mu = np.asarray(engine.mu["liquid"])[INDICES]
        native.require_match(n[present]@mu[present], row["g_J"], "Euler identity",
                             rtol=0, atol=5e-5*max(1., n.sum()))
        row.update(mu_J_mol=native.nullable(mu, present),
                   standard_J_mol=native.nullable(np.asarray(engine.mu0["liquid"])[INDICES], present),
                   max_oxide_roundtrip_error_g=float(max(abs(returned-grams))),
                   returned_component_fractions=fractions[INDICES].tolist())
        rows.append(row)
    return rows


def backend_reference(runtime, python):
    native = native_module()
    native.check_runtime(runtime)
    recipes = compositions(native)
    requests = [{"T_K": T, "P_Pa": P, "samples": recipes} for T in TEMPERATURES for P in PRESSURES]
    rng = np.random.default_rng(20261012)
    random = [{"name": f"random_{i}", "n_mol": n.tolist()}
              for i, n in enumerate(rng.dirichlet(np.ones(11), size=8))]
    random.append({"name": "scaled_basalt", "n_mol": (7*np.array(recipes[1]["n_mol"])).tolist()})
    requests.append({"T_K": 1700., "P_Pa": 5e7, "samples": random})
    # Keep exact boundaries, recording native conversion failures without floors.
    for i in range(11):
        for j in range(i, 11):
            n = np.zeros(11)
            n[i], n[j] = (.7, .3) if i != j else (1., 1.)
            requests.append({"T_K": 1700., "P_Pa": 5e7,
                             "samples": [{"name": f"edge_{i}_{j}", "n_mol": n.tolist()}]})
    rows, unavailable = [], []
    for request_data in requests:
        with tempfile.TemporaryDirectory(prefix="exoeos-dry11-") as directory:
            directory = Path(directory)
            request, output = directory/"request.json", directory/"output.json"
            request.write_text(json.dumps(request_data))
            env = dict(os.environ)
            env["LD_LIBRARY_PATH"] = str(runtime)+os.pathsep+env.get("LD_LIBRARY_PATH", "")
            result = subprocess.run([python, str(Path(__file__).resolve()), "--runtime", str(runtime),
                                     "--worker", str(request), "--output", str(output)],
                                    cwd=directory, env=env, capture_output=True, text=True, timeout=60)
            log = result.stdout+result.stderr
            negative = re.search(r"Liquid component (\S+) has negative mole fraction", log)
            if negative and request_data["samples"][0]["name"].startswith("edge_"):
                unavailable.append({**request_data, "reason": f"Pinned native oxide conversion reports negative {negative[1]} on the exact boundary; no perturbation used."})
                continue
            if result.returncode or "calcPhaseProperties failed" in log or not output.exists():
                raise RuntimeError(log)
            rows.extend(json.loads(output.read_text()))
    return {"reference_kind": "pinned_supplied_liquid_properties", "backend": native.REFERENCE["backend"],
            "method": "Fresh process per T/P or exact edge; native supplied properties and potentials with oxygen buffering disabled. Oxide amounts, fractions and Euler identity checked. No ExoEOS/JAX imports or equilibrium solve.",
            "composition_note": "BSE: existing McDonough-Sun Table 4 recipe, all iron supplied as FeO. Other named recipes are synthetic composition controls, normalized to 100 g.",
            "oxide_order": OXIDES, "recipes": recipes, "random_seed": 20261012,
            "python_version": sys.version, "numpy_version": np.__version__, "tinynumpy_version": version("tinynumpy"),
            "states": rows, "unavailable_cases": unavailable}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Directory of the three pinned source files (source mode).")
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        data = worker(args.runtime.resolve(), json.loads(args.worker.read_text()))
    else:
        if not args.runtime and not args.source:
            parser.error("supply --source or --runtime")
        data = backend_reference(args.runtime.resolve(), args.python) if args.runtime else source_reference(args.source)
        data.update(component_order=ORDER, source_revision=REVISION, source_sha256=SOURCE,
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
