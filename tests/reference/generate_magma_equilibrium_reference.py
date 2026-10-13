"""Solve the restricted binary equilibrium with actual pinned MELTS properties.

Uses a supporting-line test for absent liquid and native chemical-potential
roots for liquid/solid coexistence. No JAX model or example solver is imported.
This is an independent restricted equilibrium, not MELTS full phase selection.
Requires SciPy, the pinned runtime and the worker's tinynumpy dependency.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq, root


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
import melts_liquid_evaluator as native  # noqa: E402


def generate(runtime, python):
    candidates = [{"phase": "orthopyroxene", "endmember_moles": [0., 1., 0., 0., 0., 0., 0.]},
                  {"phase": "olivine", "endmember_moles": [0., 0., 0., 0., 0., 1.]}]
    provenance = {}

    def evaluate(T, n, solids=False):
        amounts = np.zeros(19)
        amounts[[0, 7]] = n
        state = native.evaluate_liquid(T, 1e5, amounts, runtime=runtime,
                                       python_executable=python,
                                       candidate_compositions=candidates if solids else None)
        provenance.update(state["provenance"])
        mu = np.array(state["mu_J_mol"])[[0, 7]].astype(float)
        if not solids:
            return state["gibbs_J"], mu
        en, fo = state["candidate_evaluations"]
        for row, oxide_moles in [(en, [2., 2.]), (fo, [1., 2.])]:
            if row["status"] != "ok_candidate_properties":
                raise RuntimeError(row)
            expected = np.zeros(19)
            expected[[0, 7]] = oxide_moles
            np.testing.assert_allclose(np.asarray(row["oxide_mass_g"]) / native.OXIDE_MASSES,
                                       expected, rtol=0, atol=1e-11)
        return state["gibbs_J"], mu, fo["gibbs_J"], en["gibbs_J"] / 2

    def invariant_residual(tx):
        T, x = tx
        _, mu, gf, ge = evaluate(T, [1-x, x], solids=True)
        return np.array([gf - mu[1], ge - mu.mean()])

    solution = root(invariant_residual, [1909., .4], tol=1e-10)
    if not solution.success or np.max(np.abs(solution.fun)) > 1e-6:
        raise RuntimeError(f"Invariant solve failed: {solution}")
    Ti, xi = solution.x.tolist()
    invariant = {"T_K": Ti, "liquid_x_F": xi, "residual_J_mol": solution.fun.tolist()}
    print(f"MELTS invariant: {Ti:.9f} K, liquid x_F={xi:.12f}", flush=True)

    states = []
    for M, temperatures in [(1.5, [2100., 2000., 1925., 1910., 1908.5, 1900., 1800.]),
                            (.9, [1950., 1925., 1910., 1900., 1850., 1800.])]:
        a, b = M/2, 1 - M/2
        for T in temperatures:
            gl, mu, gf, ge = evaluate(T, [b, a], solids=True)
            potentials = np.array([2*ge - gf, gf])

            def tangent_slope(x):
                _, m = evaluate(T, [1-x, x])
                return m[1] - m[0] - (potentials[1] - potentials[0])

            # These reference states have roots inside this bracket. Keep native
            # oxide conversion away from poorly resolved pure-component limits.
            x = brentq(tangent_slope, .05, .95, xtol=1e-13)
            probe_g, _ = evaluate(T, [1-x, x])
            gap = probe_g - np.dot([1-x, x], potentials)
            if a >= b and gap >= 0:
                amounts = np.array([0., 0., a-b, 2*b])
                energy = (a-b)*gf + 2*b*ge
                slopes = None
            else:
                def fo_slope(q):
                    return gf - evaluate(T, [b, a-q])[1][1]

                q = brentq(fo_slope, 0., .99*a, xtol=1e-13) if gf < mu[1] else 0.
                gl, mu = evaluate(T, [b, a-q])
                amounts = np.array([b, a-q, q, 0.])
                if ge - mu.mean() < -1e-6:
                    def en_slope(e):
                        return ge - evaluate(T, [b-e/2, a-e/2])[1].mean()

                    e = brentq(en_slope, 0., 1.98*min(a, b), xtol=1e-13)
                    amounts = np.array([b-e/2, a-e/2, 0., e])
                    gl, mu = evaluate(T, amounts[:2])
                slopes = [gf - mu[1], ge - mu.mean()]
                if min(slopes) < -1e-6:
                    raise RuntimeError(f"Unstable reference at {T}: {slopes}")
                energy = gl + amounts[2]*gf + amounts[3]*ge
            states.append({"T_K": T, "P_Pa": 1e5, "initial_oxide_moles": {"MgO": M, "SiO2": 1.},
                           "amounts_mol": amounts.tolist(), "g_J": energy,
                           "solid_insertion_J_mol": slopes,
                           "minimum_liquid_gap_J_mol_Si": gap,
                           "tangent_liquid_x_F": x, "tangent_liquid_g_J_mol_Si": probe_g,
                           "forsterite_g_J_mol": gf, "enstatite_g_J_mol_MgSiO3": ge})
            print(f"M/S={M:g}, {T:g} K: [Q, F, Fo, En]={amounts}", flush=True)
    return {"reference_kind": "pinned_binary_restricted_three_phase_equilibrium",
            "backend": native.REFERENCE["backend"],
            "method": "Fresh MELTS property workers; independent supporting-line stability and SciPy chemical-potential roots. No JAX thermodynamic model or MELTS full phase-assemblage solver.",
            "evaluator_sha256": provenance["evaluator_sha256"],
            "python_version": provenance["python_version"],
            "tinynumpy_version": provenance["tinynumpy_version"],
            "amount_order": ["liquid SiO2", "liquid Mg2SiO4", "solid Mg2SiO4", "solid MgSiO3"],
            "native_solid_candidates": candidates,
            "native_enstatite_formula": "Mg2Si2O6", "reported_enstatite_formula": "MgSiO3",
            "invariant": invariant, "states": states}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.runtime, args.python)
    result["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
