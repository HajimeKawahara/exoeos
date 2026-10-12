"""Plot native dry-solid validation and mixing/ordering characteristics.

Uses committed independent MELTS fixtures. No external runtime or phase
equilibrium solver is needed. All curves are supplied phase properties.
"""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exoeos.magma_solids import feldspar, oxides, pyroxene, spinel

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[1]
MODELS = {'plagioclase': feldspar, 'alkali-feldspar': feldspar, 'spinel': spinel,
          'rhm-oxide': oxides, 'clinopyroxene': pyroxene, 'orthopyroxene': pyroxene}
FIELDS = ('g_J', 's_J_K', 'cp_J_K', 'v_m3', 'mu_J_mol', 'standard_J_mol')


def compare(reference):
    summary = {}
    for phase, module in MODELS.items():
        fun = partial(module.gibbs, phase=phase) if module in (feldspar, pyroxene) else module.gibbs
        std = partial(module.standard_gibbs, phase=phase) if module is feldspar else module.standard_gibbs

        @jax.jit
        def properties(T, P, n):
            return (fun(T, P, n), -jax.grad(fun, 0)(T, P, n),
                    -T*jax.grad(jax.grad(fun, 0), 0)(T, P, n), jax.grad(fun, 1)(T, P, n),
                    jax.grad(fun, 2)(T, P, n), std(T, P))

        errors = []
        for row in reference['properties']:
            if row['phase'] != phase:
                continue
            values = properties(row['T_K'], row['P_Pa'], jnp.array(row['n_mol']))
            errors.append([float(np.max(np.abs(np.asarray(v)-row[key]))) for key, v in zip(FIELDS, values)])
        assert np.all(np.isfinite(errors)), phase
        summary[phase] = {'states': len(errors), 'maximum_absolute_errors': dict(zip(FIELDS, np.max(errors, axis=0).tolist()))}
        print(phase, summary[phase], flush=True)
        jax.clear_caches()
    for phase in ('rutile', 'corundum'):
        fun = getattr(oxides, phase+'_gibbs')
        properties = jax.jit(lambda T, P: (fun(T, P), -jax.grad(fun)(T, P),
                                           -T*jax.grad(jax.grad(fun))(T, P), jax.grad(fun, 1)(T, P)))
        errors = [np.abs(np.array(properties(r['T_K'], r['P_Pa']))-np.array([r[k] for k in FIELDS[:4]]))
                  for r in reference['properties'] if r['phase'] == phase]
        summary[phase] = {'states': len(errors), 'maximum_absolute_errors': dict(zip(FIELDS[:4], np.max(errors, axis=0).tolist()))}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/magma_dry_solids'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = ROOT/'tests/reference/dry_solids_melts_v1.json'
    reference = json.loads(path.read_text())
    summary = {'scope': 'Supplied dry phase properties, not crystallization or redox equilibrium.',
               'reference_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'backend': reference['backend'], 'phases': compare(reference)}
    (args.output/'dry_solids_comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    blue, orange, green = '#345a96', '#c87530', '#27826b'
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    labels = ['Plg', 'Kfs', 'Sp', 'Rhm', 'Cpx', 'Opx', 'Rt', 'Cor']
    for ax, key, unit in zip(axes.flat, FIELDS, ['J', 'J/K', 'J/K', 'm3', 'J/mol', 'J/mol']):
        values = [v['maximum_absolute_errors'].get(key, np.nan) for v in summary['phases'].values()]
        ax.bar(np.arange(8), values, color=[blue]*6+[green]*2, width=.6)
        for i, value in enumerate(values):
            if np.isnan(value):
                ax.text(i, .02, 'n/a', transform=ax.get_xaxis_transform(), ha='center', fontsize=8)
        ax.set(xticks=np.arange(8), xticklabels=labels, ylabel=f'Maximum absolute difference ({unit})',
               title={'g_J': 'Total G', 's_J_K': 'Entropy', 'cp_J_K': 'Relaxed heat capacity',
                      'v_m3': 'Volume', 'mu_J_mol': 'Chemical potentials', 'standard_J_mol': 'Standards'}[key])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=False)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Native JAX versus independent MELTS: 226 supplied states\n'
                 '1200–2200 K, 1–5000 bar; six solutions and two pure oxides', fontsize=12)
    fig.savefig(args.output/'dry_solids_validation.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    pressure = 5e7
    k = np.linspace(.001, .899, 101)
    for phase, color in zip(('plagioclase', 'alkali-feldspar'), (blue, orange)):
        fun = jax.jit(partial(feldspar.mixing_gibbs, phase=phase))
        values = [float(fun(1600., pressure, jnp.array([.9-x, .1, x])))/1000 for x in k]
        axes[0, 0].plot(k, values, label=phase, color=color)
    axes[0, 0].set(title='Feldspar: distinct ternary mixing', xlabel='Or fraction (An = 0.10)', ylabel='Mixing G (kJ/mol)')

    temperatures = np.linspace(1200., 2200., 101)
    nsp = jnp.array([.1, .2, .2, .3, .2])
    order = jax.jit(spinel.order)
    values = np.array([order(t, pressure, nsp) for t in temperatures])
    assert np.all(np.isfinite(values))
    for index, label, color in ((0, 'Mg on tetrahedral site', blue), (1, 'Al on tetrahedral site', orange),
                                (2, 'Fe3+ on tetrahedral site', green)):
        axes[0, 1].plot(temperatures, values[:, index], label=label, color=color)
    axes[0, 1].set(title='Spinel: cation redistribution', xlabel='Temperature (K)', ylabel='Tetrahedral site fraction')

    nrhm = jnp.array([.05, .05, .88, .02])
    order = jax.jit(oxides.order)
    values = np.array([order(t, pressure, nrhm) for t in temperatures])
    assert np.all(np.isfinite(values))
    axes[1, 0].plot(temperatures, np.abs(values[:, 0]-values[:, 1])/.88, color=blue, label='Fe2+ A/B order')
    axes[1, 0].plot(temperatures, np.abs(values[:, 2]-values[:, 3])/.05, color=orange, label='Mg A/B order')
    axes[1, 0].set(title='Rhombohedral oxide: order disappears on heating', xlabel='Temperature (K)', ylabel='Normalized absolute A/B order')

    npx = jnp.array([.45, .1, .1, .05, .05, .1, .15])
    for phase, color in zip(('clinopyroxene', 'orthopyroxene'), (blue, orange)):
        order = jax.jit(partial(pyroxene.order, phase=phase))
        values = np.array([order(t, pressure, npx) for t in temperatures])
        assert np.all(np.isfinite(values))
        axes[1, 1].plot(temperatures, values[:, 2], color=color, label=phase+' Fe3+ on M1')
        axes[1, 1].plot(temperatures, values[:, 4], '--', color=color, label=phase+' Fe2+ on M2')
    axes[1, 1].set(title='Pyroxenes: ferric and ferrous site exchange', xlabel='Temperature (K)', ylabel='Site fraction')
    for ax in axes.flat:
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=.2)
    fig.suptitle('Dry solid characteristics at 500 bar\nSpecified compositions; no phase-equilibrium calculation', fontsize=12)
    fig.savefig(args.output/'dry_solids_features.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
