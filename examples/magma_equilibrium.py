"""Closed binary cooling with liquid, pure forsterite and Mg orthopyroxene.

This small convex example compares liquid + Fo, liquid + En, and Fo + En.
Crystals remain in the system and may react back into other phases. It is
not a general phase-selection solver or a complete MgO--SiO2 phase diagram.
Run with JAX_ENABLE_X64=1; only the Gibbs functions are JAX-transformable.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exoeos.magma import enstatite_gibbs, forsterite_gibbs, liquid_gibbs
from magma_cooling import (
    _bisect_increasing, component_amounts, crystal_amount, enstatite_insertion_energy,
)


ENERGY_TOLERANCE = 1e-7  # J per mol of initial Si; identify invariant ties.


@jax.jit
def total_gibbs(T, P, amounts):
    """Return J for nonnegative [liquid Q, liquid F, solid Fo, solid En] mol.

    En is measured as MgSiO3, not the MELTS Mg2Si2O6 endmember. All four
    entries contain one Si per formula unit. Absent phases contribute zero.
    """
    return (liquid_gibbs(T, P, amounts[:2]) + amounts[2] * forsterite_gibbs(T, P)
            + amounts[3] * enstatite_gibbs(T, P))


def equilibrium(T, P=1e5, mgo_mol=1.5, sio2_mol=1.):
    """Minimize G over liquid + Fo + En and return amounts and tied endpoints.

    Requires finite T > W/(2R), P > 0, and 0 <= MgO/SiO2 <= 2, SiO2 > 0.
    Strict liquid convexity makes the two liquid/solid edge searches unique
    except at pure melting. Any three-phase minimum has a flat direction
    reaching these edges or the all-solid state, so comparing them suffices.

    ``amounts_mol`` and ``equilibrium_endpoints_mol`` use [Q, F, Fo, En].
    Energies within ENERGY_TOLERANCE * sio2_mol are treated as tied. Distinct
    tied endpoints describe a segment of equilibrium states, not separate
    unique solutions. Return the least-liquid endpoint as a representative;
    T/P and bulk composition alone cannot fix the amount along that segment.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 for phase energy differences.")
    # The existing Fo solve also validates T, P and the oxide inventory.
    xi = crystal_amount(T, P, mgo_mol, sio2_mol)
    b, a = component_amounts(mgo_mol, sio2_mol)
    candidates = [np.array([b, a, 0., 0.]), np.array([b, a - xi, xi, 0.])]

    if a > 0 and b > 0:
        if a == b:
            # At fixed x_F=1/2 the energy is linear in the amount melted.
            candidates.append(np.array([0., 0., 0., 2*a]))
        else:
            def slope(eta):
                return float(enstatite_insertion_energy(T, P, jnp.array([b, a]) - eta / 2))

            # At the far boundary a liquid component vanishes and slope -> +inf.
            eta = _bisect_increasing(slope, 0., 2 * min(a, b)) if slope(0.) < 0 else 0.
            candidates.append(np.array([b - eta/2, a - eta/2, 0., eta]))

    if a >= b:
        candidates.append(np.array([0., 0., a - b, 2*b]))

    energies = np.array([float(total_gibbs(T, P, n)) for n in candidates])
    tied = []
    for n, g in zip(candidates, energies):
        if g <= energies.min() + ENERGY_TOLERANCE * sio2_mol:
            if not any(np.allclose(n / sio2_mol, old / sio2_mol, atol=1e-12, rtol=0)
                       for old in tied):
                tied.append(n)
    tied.sort(key=lambda n: n[:2].sum())
    return {"amounts_mol": tied[0].tolist(), "g_J": float(total_gibbs(T, P, tied[0])),
            "degenerate": len(tied) > 1,
            "equilibrium_endpoints_mol": [n.tolist() for n in tied]}


def invariant_temperature(P=1e5):
    """Locate the Fo/En/liquid invariant on the example's 1800--2000 K bracket.

    The MgO/SiO2=1.5 Fo-only path has liquid and Fo throughout this bracket
    at 1 bar. Reject pressures where that path or the sign bracket fails.
    """
    def insertion(T):
        xi = crystal_amount(T, P)
        if not 0 < xi < .75:
            raise ValueError("The invariant bracket requires liquid + forsterite.")
        return float(enstatite_insertion_energy(T, P, jnp.array([.25, .75 - xi])))

    if not insertion(1800.) < 0 < insertion(2000.):
        raise ValueError("No enstatite invariant bracketed between 1800 and 2000 K.")
    return _bisect_increasing(insertion, 1800., 2000.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/magma_equilibrium"))
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1]
                        / "tests/reference/magma_equilibrium_melts_v1.json")
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Run with JAX_ENABLE_X64=1 for phase energy differences.")
    import matplotlib.pyplot as plt

    reference = json.loads(args.reference.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    Ti = invariant_temperature()
    result = {"scope": "Closed liquid + pure Fo + pure Mg orthopyroxene equilibrium; no silica solids or polymorph selection.",
              "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              "P_Pa": 1e5, "invariant_T_K": Ti,
              "melts_invariant_T_K": reference["invariant"]["T_K"],
              "max_amount_difference_mol": 0., "max_g_difference_J": 0.,
              "invariant_states": [], "states": []}
    for state in reference["states"]:
        M = state["initial_oxide_moles"]["MgO"]
        actual = equilibrium(state["T_K"], mgo_mol=M)
        error = np.max(np.abs(np.array(actual["amounts_mol"]) - state["amounts_mol"]))
        result["max_amount_difference_mol"] = max(result["max_amount_difference_mol"], float(error))
        result["max_g_difference_J"] = max(result["max_g_difference_J"], abs(actual["g_J"] - state["g_J"]))
        result["states"].append({"T_K": state["T_K"], "MgO_mol": M, "SiO2_mol": 1., **actual})

    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["Liquid", "Forsterite", "Enstatite (MgSiO3)"]
    styles = ["-", "-", "--"]

    def phase_amounts(n):
        n = np.asarray(n)
        return np.column_stack([n[:, :2].sum(axis=1), n[:, 2], n[:, 3]])

    fig, axes = plt.subplots(1, 2, figsize=(10., 4.5), sharey=True, layout="constrained")
    for ax, M, high in zip(axes, [1.5, .9], [2200., 2000.]):
        invariant = equilibrium(Ti, mgo_mol=M)
        result["invariant_states"].append({"MgO_mol": M, "SiO2_mol": 1., **invariant})
        ends = phase_amounts(invariant["equilibrium_endpoints_mol"])
        for side, grid in enumerate([np.linspace(Ti, 1800., 81), np.linspace(Ti, high, 101)]):
            amounts = phase_amounts([equilibrium(t, mgo_mol=M)["amounts_mol"] for t in grid])
            amounts[0] = ends[side]
            for i in range(3):
                ax.plot(grid, amounts[:, i], styles[i], color=colors[i],
                        label=labels[i] if side == 0 else None)
        # At the invariant, every point on the conserved segment is allowed.
        for i in range(3):
            ax.plot([Ti, Ti], ends[:, i], ":", color=colors[i], linewidth=2)
        states = [s for s in reference["states"] if s["initial_oxide_moles"]["MgO"] == M]
        native = phase_amounts([s["amounts_mol"] for s in states])
        for i in range(3):
            ax.plot([s["T_K"] for s in states], native[:, i], "o", mfc="none", color=colors[i])
        ax.axvline(Ti, color="0.5", linewidth=.6)
        ax.set(xlabel="Temperature (K)", title=f"MgO {M:g} mol + SiO2 1 mol", xlim=(high, 1800.), ylim=(-.04, 1.05))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Amount (mol Si in each phase)")
    axes[0].annotate("Fo = En = 0.5 mol", (1855, .51), ha="center", xytext=(1900, .68),
                     arrowprops={"arrowstyle": "-", "color": "0.4"}, fontsize=9)
    axes[0].legend(loc="center left", fontsize=9)
    fig.suptitle("Closed equilibrium, 1 bar: JAX lines, MELTS property roots (circles)\n"
                 f"Dotted segments: nonunique amounts at {Ti:.3f} K", fontsize=11)
    fig.savefig(args.output / "enstatite_equilibrium.png", dpi=180)
    plt.close(fig)

    # The Fo--En line extends to x_F<1/2 as a chemical-potential plane.
    # It is a realizable solid mixture only for 1/2 <= x_F <= 1.
    grid = jnp.linspace(.2, 1., 501)
    liquid_values = jax.jit(jax.vmap(liquid_gibbs, in_axes=(None, None, 0)))
    fig, ax = plt.subplots(figsize=(7., 4.6), layout="constrained")
    for T in [1925., Ti, 1900.]:
        gf, ge = float(forsterite_gibbs(T, 1e5)), float(enstatite_gibbs(T, 1e5))
        plane = (1-grid)*(2*ge-gf) + grid*gf
        gap = liquid_values(T, 1e5, jnp.column_stack([1-grid, grid])) - plane
        line, = ax.plot(grid, gap/1000, label=f"{T:.3f} K")
        states = [s for s in reference["states"] if s["T_K"] == T]
        if states:
            state = states[0]
            ax.plot(state["tangent_liquid_x_F"], state["minimum_liquid_gap_J_mol_Si"]/1000,
                    "o", color=line.get_color(), mfc="none", ms=7)
    ax.plot([.5, 1.], [0., 0.], "ks", label="En, Fo")
    ax.axhline(0, color="0.5", linewidth=.8)
    ax.set(xlabel=r"Liquid component fraction $x_F$", ylabel=r"$g_l - g_{\mathrm{Fo-En\ line}}$ (kJ/mol Si)",
           title="Liquid stability against the Fo-En supporting line\nJAX curves; MELTS minima (circles)",
           xlim=(.2, 1.02), ylim=(-.6, 1.5))
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    fig.savefig(args.output / "enstatite_liquid_stability.png", dpi=180)
    plt.close(fig)

    (args.output / "enstatite_equilibrium.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Invariant: JAX {Ti:.12f} K, MELTS {result['melts_invariant_T_K']:.12f} K")
    print(f"Maximum differences: {result['max_amount_difference_mol']:.3g} mol, {result['max_g_difference_J']:.3g} J")


if __name__ == "__main__":
    main()
