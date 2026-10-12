"""Generate supplied dry-solid properties using hash-pinned alphaMELTS.

No ExoEOS or JAX thermodynamic code is imported. Each T/P uses a fresh
runtime process. Endmember oxide ledgers are the previously audited source
transcriptions. Native endpoint derivative failures are preserved explicitly.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = ROOT/'examples/m2_solid_mixing/parameters.json'
PHASES = ('plagioclase', 'alkali-feldspar', 'spinel', 'rhm-oxide',
          'clinopyroxene', 'orthopyroxene')
SAMPLES = {
    'plagioclase': [[.4, .5, .1], [.1, .8, .1], [.6, .1, .3]],
    'alkali-feldspar': [[.4, .1, .5], [.8, .1, .1], [.1, .1, .8]],
    'spinel': [[.1, .2, .2, .3, .2], [.05, .05, .65, .05, .2], [.1, .1, .1, .6, .1]],
    'rhm-oxide': [[.2, .2, .5, .1], [.05, .05, .88, .02], [.05, .8, .1, .05]],
    'clinopyroxene': [[.45, .1, .1, .05, .05, .1, .15], [.1, .55, .15, .02, .02, .06, .1],
                      [-.1, .7, .2, .03, .02, .1, .05]],
    'orthopyroxene': [[.45, .1, .1, .05, .05, .1, .15], [.1, .55, .15, .02, .02, .06, .1],
                      [-.1, .7, .2, .03, .02, .1, .05]],
}
LIMIT_SAMPLES = {
    phase: [[1.] + [epsilon]*6 for epsilon in (1e-3, 1e-6)]
    for phase in ('clinopyroxene', 'orthopyroxene')
}
LIMIT_SAMPLES['rhm-oxide'] = [[1., epsilon, epsilon, epsilon] for epsilon in (1e-9, 1e-12)]


def worker(runtime, T, P):
    sys.path.insert(0, str(ROOT/'examples'))
    import melts_liquid_evaluator as native
    native.check_runtime(runtime)
    sys.path.insert(0, str(runtime))
    from meltsdynamic import MELTSdynamic
    engine = MELTSdynamic(1).engine
    engine.setSystemProperties('Log fO2 Path', 'None')
    assert not engine.status.failed
    engine.temperature, engine.pressure = T-273.15, P/1e5
    data = json.loads(PARAMETERS.read_text())
    assert data['oxide_order'] == list(native.OXIDES)
    rows, endpoints, interior_limits = [], [], []
    for phase in PHASES:
        model = data['models'][phase]
        indices = model['native_endmember_indices']
        matrix = np.array(model['endmember_oxide_moles']).T
        for sample in SAMPLES[phase]:
            n = np.array(sample)
            raw = np.zeros(19)
            raw[indices] = n/n.sum()
            engine.molarComposition = raw
            engine.calcMolarProperties(phase)
            assert not engine.status.failed
            factor = n.sum()
            row = {'phase': phase, 'T_K': T, 'P_Pa': P, 'n_mol': sample,
                   'g_J': factor*float(engine.g[phase]), 's_J_K': factor*float(engine.s[phase]),
                   'cp_J_K': factor*float(engine.cp[phase]), 'v_m3': factor*float(engine.v[phase])*1e-6}
            grams = matrix@n*native.OXIDE_MASSES
            returned = factor*float(engine.mass[phase])*np.array(engine.dispComposition[phase])/100
            native.require_match(returned, grams, 'solid oxide ledger', atol=1e-9)
            engine.calcEndMemberProperties(phase, grams.tolist())
            assert not engine.status.failed
            native.require_match(np.array(engine.X[phase])[indices], n/n.sum(), 'native fractions', atol=1e-11)
            mu = np.array(engine.mu[phase])[indices]
            row['mu_J_mol'] = mu.tolist()
            row['standard_J_mol'] = np.array(engine.mu0[phase])[indices].tolist()
            native.require_match(n@mu, row['g_J'], 'Euler identity', atol=1e-6)
            assert all(np.all(np.isfinite(v)) for k, v in row.items() if k != 'phase')
            rows.append(row)
        if P == 5e7 and T in (1200., 1600., 2200.):
            for n in np.eye(len(indices)):
                raw = np.zeros(19)
                raw[indices] = n
                engine.molarComposition = raw
                engine.calcMolarProperties(phase)
                assert not engine.status.failed
                row = {'phase': phase, 'T_K': T, 'P_Pa': P, 'n_mol': n.tolist(),
                       'derivative_reference': phase not in ('spinel', 'clinopyroxene', 'orthopyroxene')}
                for key, field, factor in [('g_J', 'g', 1.), ('s_J_K', 's', 1.),
                                            ('cp_J_K', 'cp', 1.), ('v_m3', 'v', 1e-6)]:
                    value = factor*float(getattr(engine, field)[phase])
                    row[key] = value if np.isfinite(value) else None
                row['status'] = 'finite' if all(row[k] is not None for k in ('g_J', 's_J_K', 'cp_J_K', 'v_m3')) else 'native_nonfinite_endpoint_properties'
                endpoints.append(row)
    for phase in ('rutile', 'corundum'):
        engine.molarComposition = np.r_[1., np.zeros(18)]
        engine.calcMolarProperties(phase)
        assert not engine.status.failed
        rows.append({'phase': phase, 'T_K': T, 'P_Pa': P, 'g_J': float(engine.g[phase]),
                     's_J_K': float(engine.s[phase]), 'cp_J_K': float(engine.cp[phase]),
                     'v_m3': float(engine.v[phase])*1e-6})
    if (T, P) == (1600., 5e7):
        # Run after the historical endpoint diagnostics: native endpoint
        # ordering derivatives can depend on the previously supplied state.
        for phase, samples in LIMIT_SAMPLES.items():
            indices = data['models'][phase]['native_endmember_indices']
            for sample in samples:
                n = np.array(sample)
                raw = np.zeros(19)
                raw[indices] = n/n.sum()
                engine.molarComposition = raw
                engine.calcMolarProperties(phase)
                assert not engine.status.failed
                row = {'phase': phase, 'T_K': T, 'P_Pa': P, 'n_mol': sample,
                       **{key: n.sum()*float(getattr(engine, field)[phase])*scale
                          for key, field, scale in [('g_J', 'g', 1.), ('s_J_K', 's', 1.),
                                                   ('cp_J_K', 'cp', 1.), ('v_m3', 'v', 1e-6)]}}
                assert all(np.all(np.isfinite(v)) for k, v in row.items() if k != 'phase')
                # Tiny species use direct molar input, without an oxide
                # round trip to define their singular insertion potentials.
                interior_limits.append(row)
    return {'properties': rows, 'endpoints': endpoints, 'interior_limits': interior_limits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', nargs=2, type=float)
    args = parser.parse_args()
    if args.worker:
        args.output.write_text(json.dumps(worker(args.runtime.resolve(), *args.worker), indent=2)+'\n')
        return
    previous = json.loads((ROOT/'tests/reference/dry11_melts_v1.json').read_text())
    data = {'backend': previous['backend'], 'method': 'Supplied native molar properties and chemical potentials; no phase selection or JAX imports.',
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'parameter_sha256': hashlib.sha256(PARAMETERS.read_bytes()).hexdigest(),
            'endpoint_policy': 'Native spinel/pyroxene endpoint ordering derivatives can be singular or finite but inconsistent (including negative Cp near -1e13 J/K). Preserve them as diagnostics; use finite endpoint G as a reference and independently test fixed-face derivatives against continuous pure limits and finite differences.',
            'properties': [], 'endpoints': [], 'interior_limits': []}
    with tempfile.TemporaryDirectory() as folder:
        states = [(t, p) for t in (1200., 1600., 2200.) for p in (1e5, 5e7, 5e8)]
        states += [(1435., 5e7), (1437., 5e7), (1730., 5e7), (1740., 5e7)]
        for T, P in states:
            output = Path(folder)/'worker.json'
            run = subprocess.run([sys.executable, __file__, '--runtime', str(args.runtime.resolve()),
                                  '--output', str(output), '--worker', str(T), str(P)], capture_output=True, text=True)
            if run.returncode:
                raise RuntimeError(run.stderr)
            result = json.loads(output.read_text())
            if T in (1730., 1740.):
                # Focused checks on either side of the rhombohedral pure-order
                # transition, including coexistence of local ordering minima.
                result['properties'] = [r for r in result['properties'] if r['phase'] == 'rhm-oxide']
            for key in ('properties', 'endpoints', 'interior_limits'):
                data[key].extend(result[key])
            print(T, P, flush=True)
    args.output.write_text(json.dumps(data, indent=2)+'\n')


if __name__ == '__main__':
    main()
