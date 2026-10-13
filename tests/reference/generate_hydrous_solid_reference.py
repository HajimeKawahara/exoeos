"""Generate independent hydrous-solid properties with hash-pinned alphaMELTS.

No ExoEOS or JAX thermodynamic code is imported. Each T/P uses a fresh
runtime process. Endpoint failures are preserved separately from interiors.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = ROOT/'examples/m2_solid_mixing/parameters.json'
SAMPLES = {
    'biotite': [[.3, .7], [.8, .2], [.05, .95]],
    'hornblende': [[.4, .4, .2], [.1, .2, .7], [-.2, .6, .6]],
    'clinoamphibole': [[.4, .3, .3], [.15, .75, .1], [-.2, .5, .7]],
    'orthoamphibole': [[.4, .3, .3], [.15, .75, .1], [-.2, .5, .7]],
}

DRY_HOST = [.25, .75, .02, .05, .02, .005, .005, .002, .01, .005, .002]


def cooling_reference(engine, native):
    """Restricted liquid+Fo equilibrium; independent native mu root, not phase selection."""
    from scipy.optimize import brentq
    indices = [0, 7, 5, 10, 2, 3, 1, 4, 11, 12, 13, 18]
    rows = []
    for t in (1600., 1800., 1900., 2000., 2100., 2200.):
        engine.temperature, engine.pressure = t-273.15, 500.
        engine.molarComposition = np.eye(19)[5]
        engine.calcMolarProperties('olivine')
        solid = float(engine.g['olivine'])
        for water in (0., .02, .1, .3):
            bulk = np.array(DRY_HOST+[water])

            def liquid(xi):
                n = np.zeros(19)
                n[indices] = bulk
                n[7] -= xi
                grams = (native.NU.T@n)*native.OXIDE_MASSES
                engine.calcPhaseProperties('liquid', grams.tolist())
                assert not engine.status.failed
                g = float(engine.g['liquid'])
                returned = float(engine.mass['liquid'])*np.array(engine.dispComposition['liquid'])/100
                native.require_match(returned, grams, 'cooling liquid inventory', atol=1e-9)
                engine.calcEndMemberProperties('liquid', grams.tolist())
                native.require_match(np.array(engine.X['liquid']), n/n.sum(), 'cooling fractions', atol=1e-10)
                return g, float(engine.mu['liquid'][7])

            gap = solid-liquid(0.)[1]
            xi = 0. if gap >= 0 else brentq(lambda xi: solid-liquid(xi)[1], 0., .75-1e-9, xtol=1e-12)
            g, mu = liquid(xi)
            rows.append({'T_K': t, 'P_Pa': 5e7, 'bulk_n_mol': bulk.tolist(),
                         'water_mol': water, 'forsterite_mol': xi, 'g_J': g+xi*solid,
                         'initial_insertion_J_mol': gap, 'final_insertion_J_mol': solid-mu})
    return rows


def worker(runtime, T, P):
    sys.path.insert(0, str(ROOT/'examples'))
    import melts_liquid_evaluator as native
    native.check_runtime(runtime)
    sys.path.insert(0, str(runtime))
    from meltsdynamic import MELTSdynamic
    engine = MELTSdynamic(1).engine
    library = ctypes.CDLL(str(runtime/'libalphamelts.so'))
    engine.setSystemProperties('Log fO2 Path', 'None')
    engine.temperature, engine.pressure = T-273.15, P/1e5
    data = json.loads(PARAMETERS.read_text())
    assert data['oxide_order'] == list(native.OXIDES)
    rows, endpoints = [], []
    for phase, samples in SAMPLES.items():
        model = data['models'][phase]
        indices = model['native_endmember_indices']
        matrix = np.array(model['endmember_oxide_moles']).T
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
            grams = matrix@n*native.OXIDE_MASSES
            returned = n.sum()*float(engine.mass[phase])*np.array(engine.dispComposition[phase])/100
            native.require_match(returned, grams, 'oxide ledger including H2O', atol=1e-9)
            engine.calcEndMemberProperties(phase, grams.tolist())
            assert not engine.status.failed
            row['standard_J_mol'] = np.array(engine.mu0[phase])[indices].tolist()
            returned_x = np.array(engine.X[phase])[indices]
            if phase == 'hornblende':
                # conHrn FIRST/SECOND projects Mg incorrectly (m[0]=Mg)
                # and imposes ferric/ferrous floors. Do not use its changed
                # composition as a reference for the supplied formula mol.
                # actHrn SECOND takes the requested r=[x1,x2] directly.
                row['oxide_projection_fractions'] = returned_x.tolist()
                r = (ctypes.c_double*2)(*(n/n.sum())[1:])
                mu = (ctypes.c_double*3)()
                library.actHrn(ctypes.c_int(2), ctypes.c_double(T), ctypes.c_double(P/1e5),
                               r, None, mu, None)
                row['mu_J_mol'] = (np.array(mu)+row['standard_J_mol']).tolist()
            else:
                native.require_match(returned_x, n/n.sum(), 'fractions', atol=1e-11)
                row['mu_J_mol'] = np.array(engine.mu[phase])[indices].tolist()
            native.require_match(n@np.array(row['mu_J_mol']), row['g_J'], 'Euler identity', atol=1e-6)
            assert all(np.all(np.isfinite(v)) for k, v in row.items() if k != 'phase')
            rows.append(row)
        if P == 5e7 and T in (800., 1200., 2200.):
            for n in np.eye(len(indices)):
                raw = np.zeros(19)
                raw[indices] = n
                engine.molarComposition = raw
                engine.calcMolarProperties(phase)
                assert not engine.status.failed
                row = {'phase': phase, 'T_K': T, 'P_Pa': P, 'n_mol': n.tolist()}
                for key, field, scale in [('g_J', 'g', 1.), ('s_J_K', 's', 1.),
                                           ('cp_J_K', 'cp', 1.), ('v_m3', 'v', 1e-6)]:
                    value = float(getattr(engine, field)[phase])*scale
                    row[key] = value if np.isfinite(value) else None
                endpoints.append(row)
    cooling = cooling_reference(engine, native) if (T, P) == (1600., 5e7) else []
    return {'properties': rows, 'endpoints': endpoints, 'cooling': cooling}


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
    data = {'backend': previous['backend'],
            'method': 'Supplied native properties, fixed Fe(II)/Fe(III), no phase selection or JAX imports.',
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'parameter_sha256': hashlib.sha256(PARAMETERS.read_bytes()).hexdigest(),
            'endpoint_policy': 'Preserve native endpoint failures; compare finite energies and separately verify fixed-face derivatives.',
            'hornblende_mu_method': 'Direct actHrn SECOND at requested fractions plus native mu0; oxide projection fractions retained as diagnostics.',
            'cooling_scope': 'Closed synthetic Mg-rich liquid plus pure forsterite only, 500 bar; all other solids and fluid suppressed. Native mu roots, not full MELTS crystallization.',
            'properties': [], 'endpoints': [], 'cooling': []}
    env = dict(os.environ, LD_LIBRARY_PATH=str(args.runtime.resolve())+os.pathsep+os.environ.get('LD_LIBRARY_PATH', ''))
    with tempfile.TemporaryDirectory() as folder:
        for T in (800., 1000., 1200., 1600., 2200.):
            for P in (1e5, 5e7, 5e8):
                output = Path(folder)/'worker.json'
                run = subprocess.run([sys.executable, __file__, '--runtime', str(args.runtime.resolve()),
                                      '--output', str(output), '--worker', str(T), str(P)],
                                     capture_output=True, text=True, env=env)
                if run.returncode:
                    raise RuntimeError(run.stdout+run.stderr)
                result = json.loads(output.read_text())
                for key in ('properties', 'endpoints', 'cooling'):
                    data[key].extend(result[key])
                print(T, P, flush=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
