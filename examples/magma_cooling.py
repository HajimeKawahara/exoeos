"""Closed MgO--SiO2 cooling restricted to liquid plus pure forsterite.

Run with JAX_ENABLE_X64=1. Crystals remain in the system. The default
inventory is 1.5 mol MgO + 1 mol SiO2 at 1 bar. This one-dimensional
example does not replace ExoGibbs phase selection or include enstatite.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exoeos.magma import R, W, forsterite_gibbs, liquid_gibbs, liquid_standard_gibbs


def component_amounts(mgo_mol, sio2_mol):
    """Convert oxide inventory to nonnegative [SiO2, Mg2SiO4] moles."""
    if (not np.all(np.isfinite([mgo_mol, sio2_mol]))
            or mgo_mol < 0 or sio2_mol <= 0 or mgo_mol > 2 * sio2_mol):
        raise ValueError("Require finite MgO >= 0, SiO2 > 0 and MgO/SiO2 <= 2.")
    a = mgo_mol / 2
    return np.array([sio2_mol - a, a])


def total_gibbs(T, P, xi, a=.75, b=.25):
    """Return G in J after xi mol crystallize; require 0 <= xi <= a."""
    return liquid_gibbs(T, P, jnp.array([b, a - xi])) + xi * forsterite_gibbs(T, P)


crystallization_slope = jax.jit(jax.grad(total_gibbs, argnums=2))


def _bisect_increasing(function, lower, upper):
    for _ in range(60):
        middle = (lower + upper) / 2
        if not lower < middle < upper:
            break
        if function(middle) < 0:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2


def crystal_amount(T, P=1e5, mgo_mol=1.5, sio2_mol=1.):
    """Minimize the restricted G and return forsterite moles.

    Eager input validation and scalar bisection stay outside the JAX model.
    T > W/(2R) makes the binary liquid strictly convex, so an increasing
    slope has at most one interior root. Pure forsterite selects all solid
    or all liquid; at equal standards any phase fraction minimizes G and
    this example returns zero crystal amount.
    """
    if not np.all(np.isfinite([T, P])) or T <= W / (2 * R) or P <= 0:
        raise ValueError("Require finite T > W/(2R) and P > 0.")
    b, a = component_amounts(mgo_mol, sio2_mol)
    if a == 0:
        return 0.
    if b == 0:
        return float(a) if forsterite_gibbs(T, P) < liquid_standard_gibbs(T, P)[1] else 0.
    def slope(xi):
        return float(crystallization_slope(T, P, xi, a, b))

    if slope(0.) >= 0:
        return 0.
    # The slope tends to +infinity as xi -> a; never differentiate there.
    return _bisect_increasing(slope, 0., float(a))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/magma_cooling"))
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Run with JAX_ENABLE_X64=1 for phase energy differences.")
    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True, exist_ok=True)
    temperatures = np.linspace(2200., 1800., 161)
    amounts = np.array([crystal_amount(t) for t in temperatures])
    fractions = (.75 - amounts) / (1. - amounts)
    onset = _bisect_increasing(lambda t: float(crystallization_slope(t, 1e5, 0.)), 2000., 2100.)

    xi = jnp.linspace(0., .75, 401)
    curve = jax.jit(jax.vmap(total_gibbs, in_axes=(None, None, 0)))
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    for t in (2100., 2050., 2000.):
        energy = np.asarray(curve(t, 1e5, xi) - total_gibbs(t, 1e5, 0.))
        line, = ax.plot(xi, energy / 1000, label=f"{t:.0f} K")
        equilibrium = crystal_amount(t)
        minimum = float(total_gibbs(t, 1e5, equilibrium) - total_gibbs(t, 1e5, 0.))
        ax.plot(equilibrium, minimum / 1000, "o", color=line.get_color())
    ax.axhline(0, color="0.7", linewidth=.8)
    ax.set(xlabel="Crystalline forsterite (mol)", ylabel=r"$G(\xi)-G(0)$ (kJ)",
           title="Liquid + forsterite only, 1 bar")
    ax.legend()
    fig.savefig(args.output / "gibbs_minima.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True, layout="constrained")
    axes[0].plot(temperatures, amounts)
    axes[1].plot(temperatures, fractions, color="tab:orange")
    axes[0].set(ylabel="Crystalline forsterite (mol)", title="Closed equilibrium cooling, 1 bar")
    axes[1].set(xlabel="Temperature (K)", ylabel=r"Residual liquid $x_F$")
    for ax in axes:
        ax.axvline(onset, color="0.5", linestyle="--", linewidth=1)
        ax.grid(alpha=.2)
    axes[1].invert_xaxis()
    fig.savefig(args.output / "cooling_curve.png", dpi=180)
    plt.close(fig)

    result = {"scope": "Closed liquid + pure forsterite restricted equilibrium; no enstatite or silica solids.",
              "P_Pa": 1e5, "initial_oxide_moles": {"MgO": 1.5, "SiO2": 1.},
              "onset_T_K": onset, "states": []}
    print(f"Restricted crystallization onset: {onset:.6f} K")
    for t in (2100., 2050., 2000.):
        n = crystal_amount(t)
        print(f"{t:.0f} K: {n:.6f} mol forsterite, liquid x_F = {(.75 - n) / (1 - n):.6f}")
        result["states"].append({"T_K": t, "forsterite_mol": n, "liquid_x_F": (.75 - n) / (1 - n)})
    (args.output / "cooling.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
