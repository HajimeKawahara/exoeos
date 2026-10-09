"""Generate independent source quadrature or optional pinned MELTS fixtures.

No ExoEOS/JAX model is imported. Default: integrate the published Cp and V
numerically with NumPy. With --runtime: evaluate the supplied binary liquid
and pure olivine endmember in fresh workers. These are separate references;
neither source transcription nor finite comparisons establish build identity.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REVISION = "705a0fb315e5054d18275a580562f6121c8e458c"
SOURCE_HASHES = {
    "sources/gibbs.c": "2eceff9f315210ebdcf9dfc9bacd77f625cf04dbdc5084104c093caba34d4166",
    "includes/liq_struct_data.h": "206b298d054d9110f28dd544dab1240aa252bb39650bcdf21e032fee0e7d95b4",
    "includes/sol_struct_data.h": "ebc47d7b66e7cd9bebf36d2a058270360f70fe65872f4f799f9c400f058e0033",
}
STATES = [(298.15, 1e5), (1400., 5e7), (1479., 1e5), (1480., 1e5),
          (1481., 1e5), (2000., 1e5), (2050., 1e5), (2100., 1e5),
          (2163., 1e5), (2000., 5e7), (2000., 5e8)]
NODES, WEIGHTS = np.polynomial.legendre.leggauss(96)


def integral(function, lower, upper):
    x = lower + (NODES + 1) * (upper - lower) / 2
    return float((upper - lower) / 2 * np.dot(WEIGHTS, function(x)))


def thermal(T):
    def cp_f(t):
        return 238.64 - 2001.3 / np.sqrt(t) - 1.1624e8 / t**3

    def cp_q(t):
        return 127.2 - 10.777e-3 * t + 4.3127e5 / t**2 - 1463.8 / np.sqrt(t)

    def hs(cp, t, h0, s0):
        return (h0 + integral(cp, 298.15, t),
                s0 + integral(lambda u: cp(u) / u, 298.15, t))

    hq, sq = hs(cp_q, min(T, 1480.), -901554., 48.475)
    if T >= 1480.:
        hq += 81.373 * (T - 1480.)
        sq += 81.373 * np.log(T / 1480.)
    hf, sf = hs(cp_f, 2163., -2174420., 94.010)
    hf += 2163. * 57.2 + 271. * (T - 2163.)
    sf += 57.2 + 271. * np.log(T / 2163.)
    hs_, ss = hs(cp_f, T, -2174420., 94.010)
    return (np.array([hq, hf, hs_]), np.array([sq, sf, ss]),
            np.array([81.373 if T >= 1480. else cp_q(T), 271., cp_f(T)]))


def source_reference():
    states = []
    for T, P in STATES:
        h, s, cp = thermal(T)
        # Volume in J/bar/mol; integrate pressure in bar, then return SI.
        dt = T - 1673.
        volumes = [
            lambda p: 2.690 + (-1.89e-5 + 1.3e-8 * dt) * (p - 1) + 3.6e-10 * (p - 1)**2 / 2,
            lambda p: 4.980 + 5.24e-4 * dt + (-1.35e-5 - 1.3e-8 * dt) * (p - 1) + 4.14e-10 * (p - 1)**2 / 2,
            lambda p: 4.366 * (1 - .791e-6 * (p - 1) + 1.351e-12 * (p - 1)**2
                              + 29.464e-6 * (T - 298.15) + 88.633e-10 * (T - 298.15)**2),
        ]
        dvdt = [lambda p: 1.3e-8 * (p - 1),
                lambda p: 5.24e-4 - 1.3e-8 * (p - 1),
                lambda p: np.full_like(p, 4.366 * (29.464e-6 + 2 * 88.633e-10 * (T - 298.15)))]
        g = h - T * s + np.array([integral(v, 1., P / 1e5) for v in volumes])
        s -= np.array([integral(v, 1., P / 1e5) for v in dvdt])
        cp[2] -= T * 4.366 * 2 * 88.633e-10 * (P / 1e5 - 1.)
        states.append({"T_K": T, "P_Pa": P, "g_J_mol": g.tolist(),
                       "s_J_mol_K": s.tolist(), "cp_J_mol_K": cp.tolist(),
                       "v_m3_mol": [float(v(P / 1e5) / 1e5) for v in volumes]})
    return {"reference_kind": "published_source_quadrature",
            "method": "96-point Gauss-Legendre integration of Cp, Cp/T and V; no JAX or compiled MAGMA evaluation.",
            "source_url": f"https://github.com/magmasource/MAGMA/tree/{REVISION}",
            "source_sha256": SOURCE_HASHES,
            "phase_order": ["SiO2_liquid", "Mg2SiO4_liquid", "forsterite_solid"],
            "states": states}


def backend_reference(runtime, python):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    import melts_liquid_evaluator as native

    cases = [(t, p, [.25, .75]) for t, p in STATES if t != 298.15]
    cases += [(2000., 1e5, n) for n in ([1., 0.], [0., 1.], [.75, .25], [.01, .99])]
    states, unavailable = [], []
    for T, P, amounts in cases:
        n = np.zeros(19)
        n[[0, 7]] = amounts
        try:
            state = native.evaluate_liquid(
                T, P, n, runtime=runtime, python_executable=python,
                candidate_compositions=[{"phase": "olivine", "endmember_moles": [0., 0., 0., 0., 0., 1.]}],
            )
        except RuntimeError as error:
            # The oxide conversion can round the pure F endpoint outside
            # the backend domain. Preserve the failure instead of flooring Q.
            if amounts != [0., 1.] or "SiO2 has negative mole fraction" not in str(error):
                raise
            unavailable.append({"T_K": T, "P_Pa": P, "n_mol": amounts,
                                "reason": "Pinned worker oxide conversion reports negative SiO2 at the pure Mg2SiO4 liquid endpoint."})
            continue
        solid = state["candidate_evaluations"][0]
        if solid["status"] != "ok_candidate_properties":
            raise RuntimeError(solid)
        # Confirm that the native last olivine endmember is pure Mg2SiO4.
        expected_oxides = np.zeros(19)
        expected_oxides[[0, 7]] = [1., 2.]
        np.testing.assert_allclose(np.asarray(solid["oxide_mass_g"]) / native.OXIDE_MASSES,
                                   expected_oxides, atol=1e-12)
        states.append({"T_K": T, "P_Pa": P, "n_mol": amounts,
                       "liquid_standard_J_mol": [state["mu0_J_mol"][i] for i in (0, 7)],
                       "liquid_gibbs_J": state["gibbs_J"],
                       "liquid_mu_J_mol": [state["mu_J_mol"][i] for i in (0, 7)],
                       "forsterite_gibbs_J_mol": solid["gibbs_J"]})
    return {"reference_kind": "pinned_binary_supplied_properties",
            "backend": native.REFERENCE["backend"],
            "method": "Fresh calcPhaseProperties/calcEndMemberProperties liquid worker; pure Mg2SiO4 via calcMolarProperties(olivine). No equilibrium solve.",
            "evaluator_sha256": state["provenance"]["evaluator_sha256"],
            "python_version": state["provenance"]["python_version"],
            "tinynumpy_version": state["provenance"]["tinynumpy_version"],
            "component_order": ["SiO2", "Mg2SiO4"], "states": states,
            "unavailable_cases": unavailable}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, help="Use the separately installed pinned binary.")
    parser.add_argument("--python", default=sys.executable, help="External worker Python.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = backend_reference(args.runtime, args.python) if args.runtime else source_reference()
    result["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
