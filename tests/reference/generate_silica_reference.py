"""Record pinned MELTS silica properties and independent binary equilibria.

No ExoEOS/JAX thermodynamic function or example selector is imported.
Equilibria use native supporting chemical potentials and SciPy roots,
including a global supporting-line test when liquid is absent. They are
restricted to the declared phases, not MELTS full assemblage calculations.
"""

import argparse
from functools import lru_cache
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq, root

from generate_magma_reference import REVISION, SOURCE_HASHES


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
import melts_liquid_evaluator as native  # noqa: E402

PHASES = ["forsterite", "enstatite", "quartz", "tridymite", "cristobalite"]
X_SOLID = np.array([1., .5, 0., 0., 0.])
CANDIDATES = [
    {"phase": "olivine", "endmember_moles": [0., 0., 0., 0., 0., 1.]},
    {"phase": "orthopyroxene", "endmember_moles": [0., 1., 0., 0., 0., 0., 0.]},
] + [{"phase": phase, "endmember_moles": [1.]} for phase in PHASES[2:]]


def generate(runtime, python):
    provenance = {}

    @lru_cache(maxsize=None)
    def evaluate(T, P, x=.5):
        n = np.zeros(19)
        n[[0, 7]] = [1-x, x]
        s = native.evaluate_liquid(T, P, n, runtime=runtime, python_executable=python,
                                  candidate_compositions=CANDIDATES)
        provenance.update(s["provenance"])
        solids = []
        for row, x_s, factor in zip(s["candidate_evaluations"], X_SOLID, [1., 2., 1., 1., 1.]):
            if row["status"] != "ok_candidate_properties":
                raise RuntimeError(row)
            oxide = np.asarray(row["oxide_mass_g"]) / native.OXIDE_MASSES / factor
            target = np.zeros(19)
            target[[0, 7]] = [1., 2*x_s]
            np.testing.assert_allclose(oxide, target, atol=1e-11, rtol=0)
            solids.append(row["gibbs_J"]/factor)
        return float(s["gibbs_J"]), np.array(s["mu_J_mol"])[[0, 7]].astype(float), np.array(solids)

    standards = []
    for P in [1e5, 5e7, 5e8]:
        switches = [383., 535.+.048*(P/1e5-1), 848.+.0237*(P/1e5-1)]
        for T in sorted([300., 600., 1000., 1400., 1800., 2200., 2600.]
                        + [t+d for t in switches for d in [-.01, 0., .01]]):
            standards.append({"T_K": T, "P_Pa": P,
                              "g_J_mol": evaluate(T, P)[2][2:].tolist()})
        print(f"Silica standards at {P:g} Pa complete", flush=True)

    derivatives = []
    for T, P in [(300., 1e5), (500., 5e7), (800., 5e8), (1400., 5e8)]:
        dt, dp = .01, 100.
        entropy = -(evaluate(T+dt, P)[2][2:]-evaluate(T-dt, P)[2][2:])/(2*dt)
        volume = (evaluate(T, P+dp)[2][2:]-evaluate(T, P-dp)[2][2:])/(2*dp)
        derivatives.append({"T_K": T, "P_Pa": P, "s_J_mol_K": entropy.tolist(),
                            "v_m3_mol": volume.tolist(), "difference_steps": [dt, dp]})

    # Construct supporting lines independently of the edge-amount solver.
    def solve(T, M):
        P, bulk = 1e5, M/2
        g, mu, gs = evaluate(T, P, bulk)
        columns = np.array([1-X_SOLID, X_SOLID])
        if np.min(gs-columns.T@mu) >= -1e-6:
            return np.r_[1-bulk, bulk, np.zeros(5)], g
        # An all-solid pair must support every solid AND the entire liquid.
        for i, j in combinations(range(5), 2):
            if X_SOLID[i] == X_SOLID[j] or not min(X_SOLID[i], X_SOLID[j]) <= bulk <= max(X_SOLID[i], X_SOLID[j]):
                continue
            lam = np.linalg.solve(columns[:, [i, j]].T, gs[[i, j]])
            if np.min(gs-columns.T@lam) < -1e-6:
                continue
            def slope(x):
                m = evaluate(T, P, x)[1]
                return m[1]-m[0]-lam[1]+lam[0]
            x = brentq(slope, .001, .995, xtol=1e-13)
            gap = evaluate(T, P, x)[0]-np.dot([1-x, x], lam)
            if gap >= -3e-5:  # Known native liquid G/mu roundoff convention.
                n = np.zeros(7)
                n[[i+2, j+2]] = np.linalg.solve(columns[:, [i, j]], [1-bulk, bulk])
                return n, n[2:]@gs
        for i, xs in enumerate(X_SOLID):
            if bulk == xs:
                continue
            def residual(x):
                return columns[:, i]@evaluate(T, P, x)[1]-gs[i]
            lower, upper = (.001, bulk) if bulk < xs else (bulk, .995)
            if residual(lower)*residual(upper) > 0:
                continue
            x = brentq(residual, lower, upper, xtol=1e-13)
            gl, mu, _ = evaluate(T, P, x)
            if np.min(gs-columns.T@mu) < -1e-6:
                continue
            liquid = (bulk-xs)/(x-xs)
            n = np.r_[liquid*np.array([1-x, x]), np.zeros(5)]
            n[i+2] = 1-liquid
            return n, liquid*gl+n[2:]@gs
        raise RuntimeError(f"No certified equilibrium at T={T}, MgO={M}")

    states = []
    for M, temperatures in [(.1, [2600., 2200., 1900., 1815., 1800., 800., 500.]),
                            (.5, [2000., 1850., 1830., 1815., 1800.]),
                            (.9, [2000., 1925., 1900., 1850., 1815., 1800.]),
                            (1.5, [2100., 2000., 1900.])]:
        for T in temperatures:
            n, g = solve(T, M)
            states.append({"T_K": T, "P_Pa": 1e5, "MgO_mol": M,
                           "amounts_mol": n.tolist(), "g_J": float(g),
                           "liquid_x_F": float(n[1]/n[:2].sum()) if n[:2].sum() > 0 else None})
            print(f"M/S={M:g}, {T:g} K: {n}", flush=True)

    def invariant_residual(tx):
        T, x = tx
        _, mu, gs = evaluate(T, 1e5, x)
        return [gs[1]-mu.mean(), gs[3]-mu[0]]
    inv = root(invariant_residual, [1815., .2], tol=1e-10)
    if not inv.success or np.max(np.abs(inv.fun)) > 1e-6:
        raise RuntimeError(inv)
    transition = brentq(lambda t: np.diff(evaluate(t, 1e5)[2][2:4])[0], 500., 600., xtol=1e-9)
    return {"reference_kind": "pinned_binary_properties_and_supporting_line_equilibria",
            "backend": native.REFERENCE["backend"],
            "method": "Fresh native property workers and independent SciPy supporting-line roots; no JAX model or full MELTS assemblage solver.",
            "source_revision": REVISION,
            "source_sha256": {p: SOURCE_HASHES[p] for p in ["sources/gibbs.c", "includes/sol_struct_data.h"]},
            "silica_offsets_J_mol": [-1291., -2625., -450.],
            "offset_provenance": "Quartz: public RHYOLITE_ADJUSTMENTS. Tridymite/cristobalite: constant differences transcribed from the hash-pinned runtime, not present in the pinned public source. Build identity remains unestablished.",
            "evaluator_sha256": provenance["evaluator_sha256"],
            "python_version": provenance["python_version"], "tinynumpy_version": provenance["tinynumpy_version"],
            "amount_order": ["liquid SiO2", "liquid Mg2SiO4"]+PHASES,
            "native_solid_candidates": CANDIDATES, "standards": standards, "derivatives": derivatives,
            "states": states, "enstatite_tridymite_invariant": {"T_K": float(inv.x[0]),
                "liquid_x_F": float(inv.x[1]), "residual_J_mol": inv.fun.tolist()},
            "quartz_tridymite_transition_T_K": transition}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.runtime, args.python)
    result["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
