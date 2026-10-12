"""Plot hydrous minerals and a restricted wet-liquid crystallization control.

All properties come from the installed ExoEOS package. The saved independent
MELTS fixture is only needed for reference markers and error summaries.
The liquid+Fo example suppresses every other solid and the fluid phase.
"""

import argparse
from functools import partial
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exoeos.magma import forsterite_gibbs, liquid_gibbs
from exoeos.magma_solids import amphibole, biotite, hornblende

jax.config.update('jax_enable_x64', True)
DRY_HOST = jnp.array([.25, .75, .02, .05, .02, .005, .005, .002, .01, .005, .002])
MODELS = {'biotite': biotite.gibbs, 'hornblende': hornblende.gibbs,
          'clinoamphibole': partial(amphibole.gibbs, phase='clinoamphibole'),
          'orthoamphibole': partial(amphibole.gibbs, phase='orthoamphibole')}


def total_gibbs(T, P, xi, bulk):
    """Closed liquid + xi mol pure Fo, in J; 0<=xi<=bulk[1]."""
    return liquid_gibbs(T, P, bulk.at[1].add(-xi))+xi*forsterite_gibbs(T, P)


@jax.jit
def crystal_amount(T, P, bulk):
    """Restricted minimum for the documented convex control paths.

    Pass eleven strictly positive dry amounts or twelve positive wet amounts.
    The derivative is never evaluated at zero Fo in the liquid. This small
    example is not a general phase-selection or nonconvex minimization API.
    """
    slope = lambda xi: jax.grad(total_gibbs, 2)(T, P, xi, bulk)

    def step(_, bounds):
        lo, hi = bounds
        mid = (lo+hi)/2
        return jnp.where(slope(mid) < 0, mid, lo), jnp.where(slope(mid) < 0, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 60, step, (jnp.array(0.), bulk[1]))
    return jnp.where(slope(0.) >= 0, 0., (lo+hi)/2)


def properties(fun, t, p, n):
    return jnp.array([fun(t, p, n), -jax.grad(fun)(t, p, n),
                      -t*jax.grad(jax.grad(fun))(t, p, n), jax.grad(fun, 1)(t, p, n)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/magma_hydrous_solids'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root/'tests/reference/hydrous_solids_melts_v1.json').read_text())
    import matplotlib.pyplot as plt

    fields = ('g_J', 's_J_K', 'cp_J_K', 'v_m3', 'mu_J_mol', 'standard_J_mol')
    summary = {'native_states': len(reference['properties']), 'endpoint_records': len(reference['endpoints']),
               'native_nonfinite_endpoint_records': sum(any(r[k] is None for k in fields[:4]) for r in reference['endpoints']),
               'maximum_absolute_errors': {}, 'cooling_scope': reference['cooling_scope']}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), layout='constrained')
    for name, fun in MODELS.items():
        module = {'biotite': biotite, 'hornblende': hornblende}.get(name, amphibole)
        prop, mu = jax.jit(partial(properties, fun)), jax.jit(jax.grad(fun, 2))
        standard = jax.jit(module.standard_gibbs)
        errors = []
        for row in reference['properties']:
            if row['phase'] != name:
                continue
            t, p, n = row['T_K'], row['P_Pa'], jnp.array(row['n_mol'])
            err = list(abs(np.array(prop(t, p, n))-np.array([row[k] for k in fields[:4]])))
            err += [float(max(abs(mu(t, p, n)-np.array(row['mu_J_mol'])))),
                    float(max(abs(standard(t, p)-np.array(row['standard_J_mol']))))]
            errors.append(err)
        maxima = np.max(errors, axis=0)
        summary['maximum_absolute_errors'][name] = dict(zip(fields, maxima.tolist()))
        for ax, value in zip(axes.flat, maxima):
            ax.bar(name.replace('amphibole', '\namphibole'), value)
        jax.clear_caches()
    for ax, label in zip(axes.flat, ('G (J)', 'S (J/K)', 'Cp (J/K)', 'V (m3)', 'mu (J/mol)', 'G0 (J/mol)')):
        ax.set(title=label, ylabel='Max absolute difference')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Native JAX hydrous solids vs pinned MELTS\n180 supplied states, 800--2200 K, 1--5000 bar; numerical agreement, not calibration accuracy')
    fig.savefig(args.output/'hydrous_solids_validation.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    xs = jnp.linspace(0., 1., 201)
    mix = jax.jit(jax.vmap(lambda x, t: biotite.mixing_gibbs(t, 5e7, jnp.array([x, 1-x])), in_axes=(0, None)))
    for t in (800., 1200., 1600.):
        axes[0, 0].plot(xs, mix(xs, t)/1000, label=f'{t:g} K')
    axes[0, 0].set(xlabel='Annite fraction', ylabel='Mixing G (kJ/mol)', title='Biotite: one binary entropy')
    temperatures = jnp.linspace(800., 2200., 81)
    for name, style in (('clinoamphibole', '-'), ('orthoamphibole', '--')):
        q = jax.jit(jax.vmap(lambda t: amphibole.order(t, 5e7, jnp.array([.4, .3, .3]), phase=name)))(temperatures)
        for i, color, site in zip(range(3), ('C0', 'C1', 'C2'), ('M4', 'M13', 'M2')):
            axes[0, 1].plot(temperatures, q[:, i], style, color=color, label=f'{name[:5]} {site}')
    axes[0, 1].set(xlabel='Temperature (K)', ylabel='Site Fe fraction', title='Amphibole Fe/Mg ordering')
    cooling_errors, curves = [], []
    for water in (0., .02, .1, .3):
        bulk = DRY_HOST if water == 0 else jnp.r_[DRY_HOST, water]
        tx = jnp.linspace(1600., 2200., 101)
        xi = jax.jit(jax.vmap(crystal_amount, in_axes=(0, None, None)))(tx, 5e7, bulk)
        line, = axes[1, 0].plot(tx, xi, label=f'{water:g} mol H2O')
        rows = [r for r in reference['cooling'] if r['water_mol'] == water]
        axes[1, 0].plot([r['T_K'] for r in rows], [r['forsterite_mol'] for r in rows], 'o',
                        color=line.get_color(), mfc='white', ms=4)
        points = []
        for row in rows:
            t = row['T_K']
            amount = float(crystal_amount(t, 5e7, bulk))
            energy = float(total_gibbs(t, 5e7, amount, bulk))
            cooling_errors.append([abs(amount-row['forsterite_mol']), abs(energy-row['g_J'])])
            points.append({'T_K': t, 'forsterite_mol': amount, 'g_J': energy})
        curves.append({'water_mol': water, 'states': points})
        # Water remains dissolved in this deliberately restricted control.
        if water:
            axes[1, 1].plot(tx, water/(bulk.sum()-xi), color=line.get_color(), label=f'{water:g} mol H2O')
    axes[1, 0].set(xlabel='Temperature (K)', ylabel='Crystalline Fo (mol)', title='Restricted liquid + Fo, 500 bar')
    axes[1, 1].set(xlabel='Temperature (K)', ylabel='Liquid H2O component fraction', title='Residual-liquid enrichment')
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('Hydrous mineral characteristics and an early water-effect control\nCooling panels omit hydrous minerals, other solids and fluid; circles are independent MELTS roots')
    fig.savefig(args.output/'hydrous_solids_features.png', dpi=180)
    plt.close(fig)
    summary['cooling_maximum_absolute_errors'] = dict(zip(('forsterite_mol', 'g_J'), np.max(cooling_errors, axis=0).tolist()))
    summary['cooling'] = curves
    (args.output/'hydrous_solids_comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'cooling'}, indent=2))


if __name__ == '__main__':
    main()
