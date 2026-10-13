"""Generate independent Mg orthopyroxene source or pinned MELTS references.

No JAX/ExoEOS thermodynamic function is imported. Source mode integrates
Cp, Cp/T and V and evaluates the previously transcribed pyroxene site
polynomials at the exact Mg endpoint. Backend mode solves the restricted
liquid + forsterite path by bisection of actual MELTS chemical potentials;
enstatite is an insertion diagnostic, not an equilibrated phase.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from generate_magma_reference import REVISION, SOURCE_HASHES, integral


ROOT = Path(__file__).resolve().parents[2]
STATES = [(t, p) for p in (1e5, 5e7, 5e8) for t in (1400., 1800., 2000., 2200.)]
COOLING_T = [2200., 2100., 2050., 2000., 1950., 1925., 1915., 1910., 1905., 1900., 1850., 1800.]


def source_reference():
    path = ROOT / "examples/m2_solid_mixing/parameters.json"
    model = json.loads(path.read_text())["models"]["orthopyroxene"]
    endpoint = np.array([0., 0., 0., 0., 0., 1., 0., 0.])
    reference = model["pure_reference_models"][1]

    def polynomial(terms, x):
        return sum(c * np.prod(x**np.array(powers)) for c, powers in terms)

    # All occupied sites are pure. The Mg pure-reference curve is constant.
    assert all(polynomial(site["polynomial"], endpoint) in (0., 1.)
               for site in model["entropy_sites"])
    assert not reference["entropy_sites"]
    assert all(not any(powers) for key in ("H_polynomial", "S_polynomial", "V_polynomial")
               for _, powers in reference[key])
    dh, ds, dv = [polynomial(model[key], endpoint) - polynomial(reference[key], np.array([0.]))
                  for key in ("H_polynomial", "S_polynomial", "V_polynomial")]
    def cp(t):
        return 333.16 - 2401.2 / np.sqrt(t) - 4.5412e6 / t**2 + 5.583e8 / t**3

    states = []
    for T, P in STATES:
        dt, dp = T - 298.15, P / 1e5 - 1.
        def volume(p):
            return 6.3279 * (1 + 24.656e-6 * dt + 74.670e-10 * dt**2
                             - .749e-6 * (p - 1) + .447e-12 * (p - 1)**2) + dv

        h = -3086083. + integral(cp, 298.15, T) + dh
        s = 135.164 + integral(lambda t: cp(t) / t, 298.15, T) + ds
        g = (h - T * s + integral(volume, 1., P / 1e5)) / 2
        s = (s - 6.3279 * (24.656e-6 + 2 * 74.670e-10 * dt) * dp) / 2
        heat = (cp(T) - T * 6.3279 * 2 * 74.670e-10 * dp) / 2
        states.append({"T_K": T, "P_Pa": P, "g_J_mol": g, "s_J_mol_K": s,
                       "cp_J_mol_K": heat, "v_m3_mol": volume(P / 1e5) / 2e5})
    return {"reference_kind": "published_source_quadrature_and_site_endpoint",
            "source_url": f"https://github.com/magmasource/MAGMA/tree/{REVISION}",
            "source_sha256": {"includes/sol_struct_data.h": SOURCE_HASHES["includes/sol_struct_data.h"],
                              "sources/orthopyroxene.c": model["source_sha256"]},
            "site_parameter_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "method": "96-point Cp, Cp/T, V quadrature; orthopyroxene site endpoint minus monoclinic pure reference. Not a compiled source run.",
            "formula": "MgSiO3", "native_formula": "Mg2Si2O6",
            "endpoint_correction_per_native_mol": {"H_J_mol": dh, "S_J_mol_K": ds, "V_J_mol_bar": dv},
            "states": states}


def backend_reference(runtime, python):
    sys.path.insert(0, str(ROOT / "examples"))
    import melts_liquid_evaluator as native

    candidates = [{"phase": "orthopyroxene", "endmember_moles": [0., 1., 0., 0., 0., 0., 0.]},
                  {"phase": "olivine", "endmember_moles": [0., 0., 0., 0., 0., 1.]}]

    def evaluate(T, P, xi=0., solids=False):
        n = np.zeros(19)
        n[[0, 7]] = [.25, .75 - xi]
        return native.evaluate_liquid(T, P, n, runtime=runtime, python_executable=python,
                                      candidate_compositions=candidates if solids else None)

    def solid_energies(state):
        en, fo = state["candidate_evaluations"]
        for row, sio2, mgo in ((en, 2., 2.), (fo, 1., 2.)):
            if row["status"] != "ok_candidate_properties":
                raise RuntimeError(row)
            expected = np.zeros(19)
            expected[[0, 7]] = [sio2, mgo]
            np.testing.assert_allclose(np.asarray(row["oxide_mass_g"]) / native.OXIDE_MASSES,
                                       expected, rtol=0, atol=1e-11)
        return en["gibbs_J"], fo["gibbs_J"]

    states = []
    for T, P in STATES:
        state = evaluate(T, P, solids=True)
        g_en, _ = solid_energies(state)
        states.append({"T_K": T, "P_Pa": P, "orthopyroxene_g_J_per_mol_Mg2Si2O6": g_en})

    cooling = []
    for T in COOLING_T:
        state = evaluate(T, 1e5, solids=True)
        g_en, g_fo = solid_energies(state)
        xi = 0.
        if g_fo - state["mu_J_mol"][7] < 0:
            lower, upper = 0., .75
            # The pure-F disappearance boundary has infinite positive slope;
            # evaluate only positive liquid interiors, without composition floors.
            for _ in range(40):
                xi = (lower + upper) / 2
                trial = evaluate(T, 1e5, xi)
                if g_fo - trial["mu_J_mol"][7] < 0:
                    lower = xi
                else:
                    upper = xi
            xi = (lower + upper) / 2
            state = evaluate(T, 1e5, xi)
        mu = [state["mu_J_mol"][i] for i in (0, 7)]
        cooling.append({"T_K": T, "P_Pa": 1e5, "forsterite_mol": xi,
                        "liquid_n_mol": [.25, .75 - xi], "liquid_mu_J_mol": mu,
                        "orthopyroxene_g_J_per_mol_Mg2Si2O6": g_en,
                        "insertion_J_mol_MgSiO3": (g_en - sum(mu)) / 2,
                        "forsterite_slope_J_mol": g_fo - mu[1]})
        print(f"{T:g} K: {xi:.9f} mol Fo, En insertion {cooling[-1]['insertion_J_mol_MgSiO3']:.6f} J/mol", flush=True)
    return {"reference_kind": "pinned_binary_properties_and_restricted_cooling",
            "backend": native.REFERENCE["backend"],
            "method": "Fresh MELTS workers; pure orthopyroxene/olivine via calcMolarProperties. Restricted cooling bisects actual liquid chemical potentials; no MELTS phase-assemblage solver or JAX model used.",
            "evaluator_sha256": state["provenance"]["evaluator_sha256"],
            "python_version": state["provenance"]["python_version"],
            "tinynumpy_version": state["provenance"]["tinynumpy_version"],
            "native_enstatite_endmember_moles": candidates[0]["endmember_moles"],
            "native_formula": "Mg2Si2O6", "reported_insertion_formula": "MgSiO3",
            "initial_oxide_moles": {"MgO": 1.5, "SiO2": 1.},
            "states": states, "cooling": cooling}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = backend_reference(args.runtime, args.python) if args.runtime else source_reference()
    result["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
