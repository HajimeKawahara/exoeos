"""Closed native JAX magma equilibrium with Fo, En and three SiO2 solids.

The scalar convex binary example requires x64 and T > W/(2R). Gibbs
functions are JAX-transformable; phase selection is an eager example,
not a general solver. Crystals remain in the system and can disappear.
"""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exoeos.magma import (
    R, W, SILICA_POLYMORPHS, enstatite_gibbs, forsterite_gibbs,
    liquid_gibbs, liquid_standard_gibbs, silica_gibbs,
)
from magma_cooling import _bisect_increasing, component_amounts


PHASES = ("liquid", "forsterite", "enstatite") + SILICA_POLYMORPHS
AMOUNT_ORDER = ("liquid SiO2", "liquid Mg2SiO4") + PHASES[1:]
# Rows are liquid Q/F components, columns pure solids; all contain one Si.
SOLID_COMPONENTS = np.array([[0., .5, 1., 1., 1.], [1., .5, 0., 0., 0.]])
ENERGY_TOLERANCE = 1e-7  # J per initial mol Si, including invariant ties.


@jax.jit
def solid_gibbs(T, P):
    return jnp.concatenate((jnp.array([forsterite_gibbs(T, P), enstatite_gibbs(T, P)]),
                            silica_gibbs(T, P)))


@jax.jit
def total_gibbs(T, P, n):
    """Return J for amounts in AMOUNT_ORDER, in mol of one-Si formula units."""
    return liquid_gibbs(T, P, n[:2]) + jnp.dot(n[2:], solid_gibbs(T, P))


_standards = jax.jit(liquid_standard_gibbs)


def equilibrium(T, P=1e5, mgo_mol=1.5, sio2_mol=1.):
    """Minimize G over one liquid and five pure solids in 0 <= MgO/SiO2 <= 2.

    Enumerate all liquid/single-solid edges and feasible all-solid pairs.
    Strict liquid convexity and two conservation constraints ensure every
    minimum has an endpoint in this list. Equal-energy endpoints span the
    complete equilibrium set; amounts at an invariant are not unique.
    Return the least-liquid endpoint as representative, and None for the
    composition of an absent liquid. Amounts and G scale with inventory.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 for phase energy differences.")
    if not np.all(np.isfinite([T, P])) or T <= W/(2*R) or P <= 0:
        raise ValueError("Require finite T > W/(2R) and P > 0.")
    bulk = component_amounts(mgo_mol, sio2_mol) / sio2_mol
    gl, gs = np.asarray(_standards(T, P)), np.asarray(solid_gibbs(T, P))
    candidates = [np.r_[bulk, np.zeros(5)]]
    for i, column in enumerate(SOLID_COMPONENTS.T):
        present = column > 0
        limit = min(bulk[present] / column[present])
        if limit == 0:
            continue
        if np.array_equal(bulk, column):
            amount = 1.  # Pure melting: compare the two endpoints directly.
        else:
            def slope(amount):
                n = bulk - amount*column
                x = n/n.sum()
                # Analytic gradient of the same regular-solution liquid G.
                mu = gl[present] + R*T*np.log(x[present]) + W*x[::-1][present]**2
                return gs[i] - column[present] @ mu

            if slope(0.) >= 0:
                continue
            amount = _bisect_increasing(slope, 0., limit)
        n = np.r_[np.maximum(0., bulk - amount*column), np.zeros(5)]
        n[i+2] = amount
        candidates.append(n)
    for i, j in combinations(range(5), 2):
        columns = SOLID_COMPONENTS[:, [i, j]]
        if columns[1, 0] == columns[1, 1]:
            continue  # Identical-composition polymorph ties are edge endpoints.
        amounts = np.linalg.solve(columns, bulk)
        if np.all(amounts >= 0):
            n = np.zeros(7)
            n[[i+2, j+2]] = amounts
            candidates.append(n)
    energies = np.array([float(total_gibbs(T, P, n)) for n in candidates])
    tied = []
    for n, g in zip(candidates, energies):
        if g <= energies.min() + ENERGY_TOLERANCE:
            if not any(np.allclose(n, old, atol=1e-12, rtol=0) for old in tied):
                tied.append(n)
    tied.sort(key=lambda n: n[:2].sum())
    n = tied[0]
    liquid = n[:2].sum()
    x = float(n[1]/liquid) if liquid > 0 else None
    return {"amounts_mol": (n*sio2_mol).tolist(),
            "g_J": float(total_gibbs(T, P, n))*sio2_mol,
            "phases": [p for p, a in zip(PHASES, np.r_[liquid, n[2:]]) if a > 1e-10],
            "liquid_x_F": x,
            "liquid_MgO_SiO2": 2*x if x is not None else None,
            "degenerate": len(tied) > 1,
            "equilibrium_endpoints_mol": [(end*sio2_mol).tolist() for end in tied]}


def cooling_path(temperatures, P=1e5, mgo_mol=1.5, sio2_mol=1.):
    """Return cooling states and sampled brackets for appearance/disappearance.

    Events are grid brackets, not interpolated transition temperatures.
    Decrease the temperature step to narrow them; invariant endpoint sets
    remain available through equilibrium(). No liquid composition is
    assigned after complete solidification.
    """
    temperatures = np.asarray(temperatures, dtype=float)
    if (temperatures.ndim != 1 or len(temperatures) < 2
            or not np.all(np.diff(temperatures) < 0)):
        raise ValueError("Supply at least two strictly decreasing temperatures.")
    states = [{"T_K": float(T), **equilibrium(T, P, mgo_mol, sio2_mol)} for T in temperatures]
    events = []
    for previous, current in zip(states, states[1:]):
        before, after = set(previous["phases"]), set(current["phases"])
        if before != after:
            events.append({"T_bracket_K": [current["T_K"], previous["T_K"]],
                           "appeared": [p for p in PHASES if p in after-before],
                           "disappeared": [p for p in PHASES if p in before-after]})
    return {"P_Pa": P, "initial_oxide_moles": {"MgO": mgo_mol, "SiO2": sio2_mol},
            "states": states, "events": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/magma_multiphase"))
    parser.add_argument("--pressure", type=float, default=1e5, help="Pa")
    parser.add_argument("--mgo", type=float, nargs="+", default=[.1, .5, .9, 1.5],
                        help="MgO moles for each 1 mol SiO2 inventory")
    parser.add_argument("--temperature-min", type=float, default=1600.)
    parser.add_argument("--temperature-max", type=float, default=2600.)
    parser.add_argument("--step", type=float, default=5., help="Maximum cooling step in K")
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Run with JAX_ENABLE_X64=1.")
    if (not np.all(np.isfinite([args.temperature_min, args.temperature_max, args.step]))
            or args.step <= 0 or args.temperature_min >= args.temperature_max):
        parser.error("Require finite Tmin < Tmax and step > 0.")
    import matplotlib.pyplot as plt

    reference_path = Path(__file__).resolve().parents[1]/"tests/reference/silica_melts_v1.json"
    reference = json.loads(reference_path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(args.temperature_max, args.temperature_min,
                       int(np.ceil((args.temperature_max-args.temperature_min)/args.step))+1)
    paths = [cooling_path(grid, args.pressure, M) for M in args.mgo]
    comparisons = []
    for s in reference["states"]:
        result = equilibrium(s["T_K"], s["P_Pa"], s["MgO_mol"])
        comparisons.append({"T_K": s["T_K"], "MgO_mol": s["MgO_mol"],
            "amount_error_mol": float(np.max(np.abs(np.array(result["amounts_mol"])-s["amounts_mol"]))),
            "g_error_J": abs(result["g_J"]-s["g_J"]),
            "liquid_x_F_error": (abs(result["liquid_x_F"]-s["liquid_x_F"])
                                 if s["liquid_x_F"] is not None else None)})

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red", "tab:brown"]
    labels = ["Liquid", "Forsterite", "Enstatite", "Quartz", "Tridymite", "Cristobalite"]
    fig, axes = plt.subplots(2, len(paths), figsize=(3.4*len(paths), 6.1), sharex="col",
                             sharey="row", squeeze=False, layout="constrained")
    for column, path in enumerate(paths):
        M = path["initial_oxide_moles"]["MgO"]
        amounts = np.array([s["amounts_mol"] for s in path["states"]])
        phases = np.column_stack([amounts[:, :2].sum(axis=1), amounts[:, 2:]])
        for i, label in enumerate(labels):
            axes[0, column].plot(grid, phases[:, i], color=colors[i], label=label)
        axes[1, column].plot(grid, [np.nan if s["liquid_x_F"] is None else s["liquid_x_F"]
                                    for s in path["states"]], color=colors[0])
        native_states = [s for s in reference["states"] if s["MgO_mol"] == M
                         and s["P_Pa"] == args.pressure and grid[-1] <= s["T_K"] <= grid[0]]
        for s in native_states:
            n = np.array(s["amounts_mol"])
            for i, a in enumerate(np.r_[n[:2].sum(), n[2:]]):
                if a > 0:
                    axes[0, column].plot(s["T_K"], a, "o", mfc="none", color=colors[i], ms=5)
            if s["liquid_x_F"] is not None:
                axes[1, column].plot(s["T_K"], s["liquid_x_F"], "o", mfc="none", color=colors[0], ms=5)
        for event in path["events"]:
            for ax in axes[:, column]:
                ax.axvspan(*event["T_bracket_K"], color="0.3", alpha=.12)
        axes[0, column].set(title=f"MgO / SiO2 = {M:g}", ylim=(-.03, 1.03))
        axes[1, column].set(xlabel="Temperature (K)", xlim=(grid[0], grid[-1]), ylim=(-.03, 1.03))
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("Amount (mol Si in phase)")
    axes[1, 0].set_ylabel(r"Residual liquid $x_F$ (MgO/SiO$_2$ = $2x_F$)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Native JAX silicate magma, {args.pressure/1e5:g} bar: closed equilibrium\n"
                 "Lines: JAX; circles: independent MELTS property roots; bands: sampled phase changes", fontsize=11)
    fig.savefig(args.output/"multiphase_cooling.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.7), sharex=True, sharey="row",
                             layout="constrained", gridspec_kw={"height_ratios": [3, 1]})
    silica_values = jax.jit(jax.vmap(silica_gibbs, in_axes=(0, None)))
    t = np.linspace(300., 2600., 700)
    max_standard_error = 0.
    for column, P in enumerate([1e5, 5e7, 5e8]):
        g = np.asarray(silica_values(t, P))
        rows = [s for s in reference["standards"] if s["P_Pa"] == P]
        native_t = np.array([s["T_K"] for s in rows])
        native_g = np.array([s["g_J_mol"] for s in rows])
        errors = np.asarray(silica_values(native_t, P))-native_g
        max_standard_error = max(max_standard_error, float(abs(errors).max()))
        for i, label in enumerate(labels[3:]):
            axes[0, column].plot(t, (g[:, i]-g[:, 0])/1000, color=colors[i+3], label=label)
            axes[0, column].plot(native_t, (native_g[:, i]-native_g[:, 0])/1000,
                                 "o", mfc="none", ms=3, color=colors[i+3])
            axes[1, column].plot(native_t, errors[:, i], ".", color=colors[i+3])
        axes[0, column].set(title=f"{P/1e5:g} bar")
        axes[1, column].set(xlabel="Temperature (K)")
        axes[1, column].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel(r"$g - g_{Quartz}$ (kJ/mol SiO$_2$)")
    axes[1, 0].set_ylabel("JAX - MELTS (J/mol)")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("SiO2 polymorph standards: JAX lines and pinned rhyolite-MELTS 1.0.2 points\n"
                 "Runtime offsets included; pressure checks test formulas, not experimental calibration", fontsize=11)
    fig.savefig(args.output/"silica_melts_comparison.png", dpi=180)
    plt.close(fig)

    result = {"scope": "Native JAX silicate magma, one binary liquid + Fo + Mg orthopyroxene + quartz/tridymite/cristobalite. Closed equilibrium, not a complete experimental phase diagram.",
              "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
              "amount_order": AMOUNT_ORDER, "paths": paths, "comparisons": comparisons,
              "max_standard_error_J_mol": max_standard_error,
              "enstatite_tridymite_invariant": reference["enstatite_tridymite_invariant"],
              "quartz_tridymite_transition_T_K": reference["quartz_tridymite_transition_T_K"]}
    (args.output/"multiphase_cooling.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"Maximum silica standard difference: {max_standard_error:.3g} J/mol")
    print(f"Maximum amount difference: {max(c['amount_error_mol'] for c in comparisons):.3g} mol")
    print(f"Maximum total-G difference: {max(c['g_error_J'] for c in comparisons):.3g} J")
    for path in paths:
        print(f"MgO/SiO2={path['initial_oxide_moles']['MgO']:g}: {path['events']}")


if __name__ == "__main__":
    main()
