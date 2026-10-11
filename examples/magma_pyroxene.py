"""Dry CMFAS cooling with Ca--Mg--Fe clinopyroxene and orthopyroxene.

Requires SciPy, matplotlib and JAX x64. Retains crystals and one phase of
each structural type; only resolved, melt-bearing states are returned.
This numerical example is not a general or differentiable phase selector.
"""

import argparse
from functools import partial
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, minimize, root

from exoeos.magma import (
    R, anorthite_gibbs, liquid_gibbs, liquid_standard_gibbs, olivine_gibbs,
    pyroxene_gibbs, silica_gibbs,
)
from magma_dry import DEFAULT_OXIDES, LIQUID_MASSES, PHASES, component_amounts


# Fo, Fa, cpx [CaSiO3, MgSiO3, FeSiO3], An, opx [Ca, Mg, Fe], SiO2.
PYX_MAP = np.array([[0., .5, .5], [0., .5, 0.], [0., 0., .5],
                    [1., 0., 0.], [0., 0., 0.]])
SOLID_COMPONENTS = np.column_stack((np.eye(5)[:, 1:3], PYX_MAP,
                                    [1., 0., 0., 1., 1.], PYX_MAP, np.eye(5)[:, 0]))
LOW_CA_OXIDES = {"SiO2": 1., "MgO": .9, "FeO": .1, "CaO": .04, "Al2O3": .02}


def pyroxene_composition(c, y):
    """One mol Si: c is M2 Ca occupancy; y is bulk Fe/(Mg+Fe)."""
    return jnp.array([c/2, (1-c/2)*(1-y), (1-c/2)*y])


def solid_amounts(z):
    return jnp.concatenate((z[0]*jnp.array([1-z[5], z[5]]),
                            z[1]*pyroxene_composition(*z[6:8]), z[2:3],
                            z[3]*pyroxene_composition(*z[8:10]), z[4:5]))


@jax.jit
def liquid_amounts(z, bulk):
    return bulk-jnp.asarray(SOLID_COMPONENTS)@solid_amounts(z)


@jax.jit
def energy(z, T, P, bulk):
    return (liquid_gibbs(T, P, liquid_amounts(z, bulk))
            + z[0]*olivine_gibbs(T, P, jnp.array([1-z[5], z[5]]))
            + z[1]*pyroxene_gibbs(T, P, pyroxene_composition(*z[6:8]))
            + z[2]*anorthite_gibbs(T, P)
            + z[3]*pyroxene_gibbs(T, P, pyroxene_composition(*z[8:10]), phase="orthopyroxene")
            + z[4]*jnp.min(silica_gibbs(T, P)))


@jax.jit
def objective(z, T, P, bulk):
    baseline = bulk@liquid_standard_gibbs(T, P, include_fe=True, include_ca_al=True)
    return (energy(z, T, P, bulk)-baseline)/(R*T)


gradient = jax.jit(jax.grad(objective))
liquid_jacobian = jax.jit(jax.jacfwd(liquid_amounts))
liquid_mu = jax.jit(jax.grad(liquid_gibbs, 2))
olivine_mu = jax.jit(jax.grad(olivine_gibbs, 2))


@partial(jax.jit, static_argnames="phase")
def pyroxene_gap(x, T, P, potentials, phase):
    n = pyroxene_composition(*x)
    return (pyroxene_gibbs(T, P, n, phase=phase)-n@potentials)/(R*T)


gap_gradient = jax.jit(jax.grad(pyroxene_gap), static_argnames="phase")


def insertion_gaps(T, P, liquid):
    """Search all admitted solids, including Ca-rich and Ca-poor starts."""
    mu = np.asarray(liquid_mu(T, P, liquid))
    def slope(y):
        delta = np.asarray(olivine_mu(T, P, jnp.array([1-y, y])))-mu[1:3]
        return delta[1]-delta[0]
    y = brentq(slope, 1e-12, 1-1e-12, xtol=1e-14)
    n = np.array([1-y, y])
    gaps = [float(olivine_gibbs(T, P, n)-n@mu[1:3])]
    compositions = [y]
    for phase in ("clinopyroxene", "orthopyroxene"):
        candidates = [minimize(pyroxene_gap, [c, f], args=(T, P, PYX_MAP.T@mu, phase),
                               jac=gap_gradient, method="L-BFGS-B",
                               bounds=[(1e-9, 1-1e-9)]*2,
                               options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 200})
                      for c in (.02, .5, .95) for f in (.2, .8)]
        best = min(candidates, key=lambda row: row.fun)
        if not np.isfinite(best.fun) or np.max(np.abs(best.jac)) > 2e-6:
            raise RuntimeError(f"Unresolved {phase} tangent minimum.")
        gaps.append(float(best.fun*R*T))
        compositions.extend(best.x.tolist())
    pure = [float(anorthite_gibbs(T, P))-mu@SOLID_COMPONENTS[:, 5],
            float(jnp.min(silica_gibbs(T, P)))-mu[0]]
    return np.array([gaps[0], gaps[1], pure[0], gaps[2], pure[1]]), compositions


def equilibrium(T, P=1e5, oxides=None):
    """Return a conserved local minimum checked against solid tangent searches.

    Amounts use mol Si for both pyroxenes. Multiple liquids, an exhausted
    liquid, or unresolved solid splitting are not supported. The tangent
    searches use six starts per pyroxene; they are numerical checks, not
    certified global bounds on every possible assemblage.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 for phase equilibrium.")
    if not np.all(np.isfinite([T, P])) or not 1200 <= T <= 2200 or not 1e5 <= P <= 5e8:
        raise ValueError("Require 1200 <= T/K <= 2200 and 1 <= P/bar <= 5000.")
    oxides = DEFAULT_OXIDES if oxides is None else oxides
    original = component_amounts(oxides)
    scale = original.sum()
    bulk = original/scale
    constraints = {"type": "ineq", "fun": lambda z: np.asarray(liquid_amounts(z, bulk))-1e-12,
                   "jac": lambda z: np.asarray(liquid_jacobian(z, bulk))}
    composition_indices = {0: [5], 1: [6, 7], 3: [8, 9]}
    candidates = []
    for compositions in (insertion_gaps(T, P, bulk)[1], [.3, .95, .3, .02, .3]):
        start = np.r_[np.zeros(5), compositions]
        for _ in range(6):
            result = minimize(objective, start, args=(T, P, bulk), jac=gradient,
                              method="SLSQP", bounds=[(0., 2.)]*5+[(1e-8, 1-1e-8)]*5,
                              constraints=constraints, options={"ftol": 2e-13, "maxiter": 400})
            liquid = np.asarray(liquid_amounts(result.x, bulk))
            if not result.success or min(liquid) <= 1e-10:
                break
            gaps, incipient = insertion_gaps(T, P, liquid)
            missing = [i for i in composition_indices if result.x[i] < 1e-8 and gaps[i] < -5e-4]
            if not missing:
                break
            start = result.x.copy()
            for i in missing:
                start[i] = 0.
                indices = composition_indices[i]
                start[indices] = np.asarray(incipient)[np.array(indices)-5]
        if result.success and np.all(constraints["fun"](result.x) >= -1e-10):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("Pyroxene cooling did not resolve a melt-bearing equilibrium.")
    z = min(candidates, key=lambda row: row.fun).x.copy()
    if min(liquid_amounts(z, bulk)) <= 1e-10:
        raise RuntimeError("Unresolved liquid disappearance.")
    active = z[:5] > 1e-8
    z[:5][~active] = 0.
    indices = np.array(list(np.flatnonzero(active)) + [j for i in composition_indices
                       if active[i] for j in composition_indices[i]], dtype=int)
    if len(indices):
        def residual(values):
            trial = z.copy()
            trial[indices] = values
            return np.asarray(gradient(trial, T, P, bulk))[indices]
        polished = root(residual, z[indices], tol=1e-10)
        z[indices] = polished.x
        if np.max(np.abs(residual(polished.x))) > 1e-8:
            raise RuntimeError("Unresolved coexistence chemical potentials.")
    liquid = np.asarray(liquid_amounts(z, bulk))
    if min(liquid) <= 1e-10 or min(z[:5]) < 0 or np.any((z[5:] <= 0) | (z[5:] >= 1)):
        raise RuntimeError("Unresolved liquid or solid composition boundary.")
    gaps, _ = insertion_gaps(T, P, liquid)
    if min(gaps) < -5e-4 or (np.any(active) and max(abs(gaps[active])) > 5e-4):
        raise RuntimeError(f"Unresolved solid stability or splitting: {gaps}")
    return {"T_K": float(T), "P_Pa": float(P), "initial_oxide_moles": dict(oxides),
            "liquid_mol": (liquid*scale).tolist(), "solid_mol": (np.asarray(solid_amounts(z))*scale).tolist(),
            "phase_mol_Si": (z[:5]*scale).tolist(), "g_J": float(energy(z, T, P, bulk)*scale),
            "solid_insertion_J_mol_Si": gaps.tolist(),
            "cpx_Ca_M2": float(z[6]) if active[1] else None,
            "cpx_Fe_fraction": float(z[7]) if active[1] else None,
            "opx_Ca_M2": float(z[8]) if active[3] else None,
            "opx_Fe_fraction": float(z[9]) if active[3] else None}


def mass_fractions(state):
    solids = np.array(state["solid_mol"])*(LIQUID_MASSES@SOLID_COMPONENTS)
    masses = np.r_[LIQUID_MASSES@state["liquid_mol"], solids[:2].sum(), solids[2:5].sum(),
                    solids[5], solids[6:9].sum(), solids[9]]
    return masses/masses.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/magma_pyroxene"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = json.loads((Path(__file__).resolve().parents[1]
                            / "tests/reference/pyroxene_melts_v1.json").read_text())
    temperatures = np.r_[np.linspace(1800., 1500., 16), np.arange(1495., 1469., -5),
                         np.arange(1469., 1459., -1)]
    states = [equilibrium(float(T)) for T in temperatures]
    low_ca = [equilibrium(float(T), oxides=LOW_CA_OXIDES) for T in np.linspace(1900., 1550., 15)]
    compared = [equilibrium(row["T_K"], row["P_Pa"], row["initial_oxide_moles"])
                for row in reference["restricted_equilibria"]]
    errors = {
        "equilibrium_amount_mol": max(float(np.max(np.abs(np.r_[a["liquid_mol"], a["solid_mol"]]
            -np.r_[b["liquid_mol"], b["solid_mol"]]))) for a, b in zip(compared, reference["restricted_equilibria"])),
        "equilibrium_G_J": max(abs(a["g_J"]-b["g_J"]) for a, b in zip(compared, reference["restricted_equilibria"])),
    }
    functions = {phase: jax.jit(partial(pyroxene_gibbs, phase=phase))
                 for phase in ("clinopyroxene", "orthopyroxene")}
    for phase, fun in functions.items():
        derivatives = [jax.jit(jax.grad(fun, 0)), jax.jit(jax.grad(fun, 1)),
                       jax.jit(jax.grad(jax.grad(fun, 0), 0)), jax.jit(jax.grad(fun, 2))]
        for row in reference["properties"]:
            if row["phase"] != phase or "status" in row:
                continue
            t, p, n = row["T_K"], row["P_Pa"], jnp.array(row["n_mol"])
            values = {"g_J": fun(t, p, n)}
            if "mu_J_mol" in row:
                dt, dp, dtt, dn = derivatives
                values.update(s_J_K=-dt(t, p, n), v_m3=dp(t, p, n),
                              cp_J_K=-t*dtt(t, p, n), mu_J_mol=dn(t, p, n))
            for key, value in values.items():
                errors[key] = max(errors.get(key, 0.), float(np.max(np.abs(value-np.array(row[key])))))
    summary = {"scope": "One liquid, Fo--Fa olivine, Ca--Mg--Fe cpx/opx, pure An and silica; retained crystals, Fe(II). Melt-bearing numerical equilibria with multi-start solid tangent checks, not a full phase diagram.",
               "phase_order": ["liquid"]+list(PHASES), "maximum_absolute_errors": errors,
               "default_bulk": states, "low_Ca_bulk": low_ca,
               "native_reference": reference["backend"],
               "native_unavailable_endpoints": sum("status" in r for r in reference["properties"])}
    (args.output / "pyroxene_comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")

    import matplotlib.pyplot as plt
    colors = ["#2166ac", "#d89021", "#218e68", "#b85084", "#7558a3", "#777777"]
    labels = ["Liquid", "Olivine", "Clinopyroxene", "Anorthite", "Orthopyroxene", "Silica"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    for column, (path, title) in enumerate([(states, "Bulk from #65"), (low_ca, "Low-Ca bulk")]):
        ts = [r["T_K"] for r in path]
        masses = np.array([mass_fractions(r) for r in path])
        selected = [r for r in reference["restricted_equilibria"]
                    if r["initial_oxide_moles"] == path[0]["initial_oxide_moles"]]
        for i, label in enumerate(labels):
            if masses[:, i].max() > 0:
                axes[0, column].plot(ts, masses[:, i], color=colors[i], label=label)
                axes[0, column].plot([r["T_K"] for r in selected], [mass_fractions(r)[i] for r in selected],
                                     "o", mfc="white", ms=5, color=colors[i])
        axes[0, column].set(ylabel="Mass fraction", title=title, ylim=(-.02, 1.02))
        axes[0, column].legend(fontsize=8)
        for key, label, color in [("cpx_Ca_M2", "Cpx Ca on M2", colors[2]),
                                  ("opx_Ca_M2", "Opx Ca on M2", colors[4])]:
            if any(r[key] is not None for r in path):
                axes[1, column].plot(ts, [r[key] for r in path], color=color, label=label)
        for key, label, color in [("cpx_Fe_fraction", "Cpx Fe/(Mg+Fe)", colors[2]),
                                  ("opx_Fe_fraction", "Opx Fe/(Mg+Fe)", colors[4])]:
            if any(r[key] is not None for r in path):
                axes[1, column].plot(ts, [r[key] for r in path], "--", color=color, label=label)
        for row in selected:
            for indices, color in [(slice(2, 5), colors[2]), (slice(6, 9), colors[4])]:
                n = np.array(row["solid_mol"])[indices]
                if sum(n) > 0:
                    axes[1, column].plot(row["T_K"], 2*n[0]/sum(n), "o", mfc="white", ms=5, color=color)
                    axes[1, column].plot(row["T_K"], n[2]/sum(n[1:]), "o", mfc="white", ms=5, color=color)
        axes[1, column].set(ylabel="Cation fraction", title="Equilibrated pyroxene compositions",
                            ylim=(0, .8 if column == 0 else .12))
        axes[1, column].legend(fontsize=8)
        for ax in axes[:, column]:
            ax.set(xlabel="Temperature (K); cooling to the right", xlim=(max(ts), min(ts)))
            ax.grid(alpha=.2)
        axes[1, column].set_xlim(1485 if column == 0 else 1850, min(ts))
    fig.suptitle("Ca-Mg-Fe pyroxenes, 1 bar\nLines: native JAX; circles: independent MELTS property roots", fontsize=12)
    fig.savefig(args.output / "pyroxene_cooling.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    grid = jnp.linspace(1200., 2200., 101)
    compositions = [([.8, 1., .2], "Ca-rich: Ca0.80 Mg1.00 Fe0.20"),
                    ([.05, 1.5, .45], "Ca-poor: Ca0.05 Mg1.50 Fe0.45"),
                    ([.2, .3, 1.5], "Fe-rich: Ca0.20 Mg0.30 Fe1.50")]
    for ax, P in zip(axes, (1e5, 5e8)):
        for i, (amounts, label) in enumerate(compositions):
            n = jnp.array(amounts)
            delta = jax.vmap(lambda t: (functions["orthopyroxene"](t, P, n)
                                       -functions["clinopyroxene"](t, P, n))/n.sum())(grid)
            ax.plot(grid, delta, color=colors[i+1], label=label)
            matched = [r for r in reference["properties"] if r["n_mol"] == amounts and r["P_Pa"] == P]
            ts = sorted(set(r["T_K"] for r in matched))
            native_delta = []
            for t in ts:
                pair = {r["phase"]: r["g_J"] for r in matched if r["T_K"] == t}
                native_delta.append((pair["orthopyroxene"]-pair["clinopyroxene"])/sum(amounts))
            ax.plot(ts, native_delta, "o", color=colors[i+1], mfc="white", ms=6)
        ax.axhline(0, color="black", lw=.8)
        ax.set(title=f"{P/1e5:g} bar", xlabel="Temperature (K)", ylabel="G(opx) - G(cpx) (J/mol Si)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Same composition, different pyroxene structures\nLines: native JAX; circles: alphaMELTS 2.3.2 supplied properties", fontsize=12)
    fig.savefig(args.output / "pyroxene_melts_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(errors, indent=2))


if __name__ == "__main__":
    main()
