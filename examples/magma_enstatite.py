"""Compare native JAX Mg orthopyroxene and its insertion test with MELTS.

Uses saved independent MELTS evaluations; no runtime is needed to plot.
The liquid + forsterite path continues below enstatite saturation solely
as a restricted, unstable reference. No enstatite amount is calculated.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exoeos.magma import enstatite_gibbs
from magma_cooling import _bisect_increasing, crystal_amount, enstatite_insertion_energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/magma_cooling"))
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1]
                        / "tests/reference/enstatite_melts_v1.json")
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Run with JAX_ENABLE_X64=1 for phase energy differences.")
    import matplotlib.pyplot as plt

    reference = json.loads(args.reference.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    values = jax.jit(jax.vmap(enstatite_gibbs, in_axes=(0, None)))
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.0), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [2, 1]})
    max_g_error = 0.
    for pressure, label in [(1e5, "1 bar"), (5e7, "500 bar"), (5e8, "5000 bar")]:
        states = [s for s in reference["states"] if s["P_Pa"] == pressure]
        t = np.array([s["T_K"] for s in states])
        native = np.array([s["orthopyroxene_g_J_per_mol_Mg2Si2O6"] / 2 for s in states])
        grid = jnp.linspace(t.min(), t.max(), 121)
        line, = axes[0].plot(grid, values(grid, pressure) / 1000, label=label)
        axes[0].plot(t, native / 1000, "o", mfc="none", color=line.get_color())
        residual = np.asarray(values(t, pressure)) - native
        axes[1].plot(t, residual, "o-", color=line.get_color())
        max_g_error = max(max_g_error, float(np.max(np.abs(residual))))
    axes[0].set(ylabel=r"$g_{En}$ (kJ / mol MgSiO$_3$)",
                title="Pure Mg orthopyroxene: JAX lines, MELTS circles")
    axes[0].legend()
    axes[1].set(xlabel="Temperature (K)", ylabel="JAX - MELTS (J/mol)")
    axes[1].axhline(0, color="0.5", linewidth=.8)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(args.output / "enstatite_melts_comparison.png", dpi=180)
    plt.close(fig)

    def insertion(T):
        xi = crystal_amount(T)
        return float(enstatite_insertion_energy(T, 1e5, jnp.array([.25, .75 - xi])))

    onset = _bisect_increasing(insertion, 1800., 2000.)
    temperatures = np.linspace(2200., 1800., 201)
    amounts = np.array([crystal_amount(t) for t in temperatures])
    liquid = jnp.stack((jnp.full(len(amounts), .25), .75 - amounts), axis=1)
    delta = np.asarray(jax.vmap(enstatite_insertion_energy, in_axes=(0, None, 0))(
        temperatures, 1e5, liquid))
    native_t = np.array([s["T_K"] for s in reference["cooling"]])
    native_delta = np.array([s["insertion_J_mol_MgSiO3"] for s in reference["cooling"]])
    native_xi = np.array([s["forsterite_mol"] for s in reference["cooling"]])
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.0), sharex=True, layout="constrained")
    axes[0].plot(temperatures, delta / 1000, label="JAX")
    axes[0].plot(native_t, native_delta / 1000, "o", mfc="none", color="tab:red", label="MELTS")
    axes[0].axhline(0, color="0.4", linewidth=.8)
    axes[0].set(ylabel=r"$\Delta g_{En}$ (kJ / mol MgSiO$_3$)",
                title="Enstatite insertion on the liquid + forsterite path, 1 bar")
    axes[0].legend()
    axes[1].plot(temperatures, amounts)
    axes[1].plot(native_t, native_xi, "o", mfc="none", color="tab:red")
    axes[1].set(xlabel="Temperature (K)", ylabel="Restricted-path forsterite (mol)")
    for ax in axes:
        ax.axvline(onset, color="0.4", linestyle="--", linewidth=1)
        ax.axvspan(1800., onset, color="tab:red", alpha=.08)
        ax.grid(alpha=.2)
    axes[0].text(1810., 8.5, "En insertion favorable\n(restricted path unstable)",
                 ha="right", fontsize=9)
    axes[1].invert_xaxis()
    fig.savefig(args.output / "enstatite_saturation.png", dpi=180)
    plt.close(fig)

    delta_errors = np.array([insertion(t) for t in native_t]) - native_delta
    amount_errors = np.array([crystal_amount(t) for t in native_t]) - native_xi
    result = {"scope": "Pure Mg orthopyroxene insertion test on closed liquid + forsterite cooling; no enstatite phase amounts or full phase selection.",
              "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              "P_Pa": 1e5, "initial_oxide_moles": {"MgO": 1.5, "SiO2": 1.},
              "enstatite_saturation_T_K": onset,
              "max_standard_difference_J_mol_MgSiO3": max_g_error,
              "max_insertion_difference_J_mol_MgSiO3": float(np.max(np.abs(delta_errors))),
              "max_forsterite_amount_difference_mol": float(np.max(np.abs(amount_errors))),
              "states": [{"T_K": float(t), "insertion_J_mol_MgSiO3": insertion(t),
                          "forsterite_mol": crystal_amount(t)} for t in (2000., onset, 1900., 1800.)]}
    (args.output / "enstatite_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
