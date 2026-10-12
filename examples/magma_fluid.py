"""Plot supplied liquid/fluid water potentials and independent references.

No equilibrium root, saturation curve or phase amount is calculated.
The thermodynamic kernels and coefficients come from installed ExoEOS.
The saved reference fixtures are used only to draw validation points.
"""

import argparse
from functools import partial
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exoeos import IdealEOS, ZhangDuanEOS
from exoeos.magma import liquid_gibbs
from exoeos.magma_fluid import water_fluid_gibbs, water_gas_standard_gibbs


jax.config.update("jax_enable_x64", True)
ZD = ZhangDuanEOS.from_species(("H2O",))
FLUID = partial(water_fluid_gibbs, eos=ZD)
MU_FLUID = jax.jit(jax.grad(FLUID, 2))
MU_IDEAL = jax.jit(jax.grad(partial(water_fluid_gibbs, eos=IdealEOS()), 2))


def water_mu(t, p, dry, water):
    return jax.grad(lambda w: liquid_gibbs(t, p, jnp.r_[dry, w]))(water)


MU_LIQUID = jax.jit(water_mu)
FIELDS = ("g_J_mol", "s_J_mol_K", "cp_J_mol_K", "v_m3_mol",
          "dvdt_m3_mol_K", "dvdp_m3_mol_Pa")


def properties(t, p):
    return jnp.array([FLUID(t, p, 1.), -jax.grad(FLUID, 0)(t, p, 1.),
                      -t*jax.grad(jax.grad(FLUID, 0), 0)(t, p, 1.),
                      jax.grad(FLUID, 1)(t, p, 1.),
                      jax.grad(jax.grad(FLUID, 1), 0)(t, p, 1.),
                      jax.grad(jax.grad(FLUID, 1), 1)(t, p, 1.)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/magma_fluid"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]/"tests/reference"
    reference = json.loads((root/"magma_fluid_v1.json").read_text())
    melts = json.loads((root/"hydrous_melts_v1.json").read_text())
    evaluate = jax.jit(properties)
    actual = np.array([evaluate(r["T_K"], r["P_Pa"]) for r in reference["states"]])
    errors = actual-np.array([[r[k] for k in FIELDS] for r in reference["states"]])
    std = jax.jit(lambda t: jnp.array([water_gas_standard_gibbs(t),
        -jax.grad(water_gas_standard_gibbs)(t),
        -t*jax.grad(jax.grad(water_gas_standard_gibbs))(t)]))
    std_errors = np.array([std(r["T_K"])-np.array([r[k] for k in FIELDS[:3]]) for r in reference["standards"]])
    interior = [r for r in melts["states"] if min(r["n_mol"]) > 0]
    liquid_errors = [float(MU_LIQUID(r["T_K"], r["P_Pa"], jnp.array(r["n_mol"][:-1]), r["n_mol"][-1]))-r["mu_J_mol"][-1] for r in interior]
    table = []
    for recipe in melts["recipes"][3:6]:
        n = np.array(recipe["n_mol"])
        for p in (5e6, 5e7, 5e8):
            liquid = float(MU_LIQUID(1700., p, n[:-1], n[-1]))
            fluid = float(MU_FLUID(1700., p, 1.))
            table.append({"T_K": 1700., "P_Pa": p, "water_wt_percent": recipe["water_wt_percent"],
                          "liquid_mu_J_mol": liquid, "fluid_mu_J_mol": fluid,
                          "fluid_minus_liquid_J_mol": fluid-liquid})
    summary = {"scope": "Supplied compositions only; Haar ideal standard plus ZD09 residual, not MELTS fluid EOS or experimental saturation validation.",
               "standard_states": len(reference["standards"]), "fluid_states": len(errors),
               "liquid_mu_states": len(interior),
               "standard_maximum_absolute_errors": dict(zip(FIELDS[:3], np.max(abs(std_errors), axis=0).tolist())),
               "fluid_maximum_absolute_errors": dict(zip(FIELDS, np.max(abs(errors), axis=0).tolist())),
               "liquid_mu_maximum_absolute_error_J_mol": max(abs(np.array(liquid_errors))),
               "chemical_potential_comparison": table}
    (args.output/"fluid_comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")

    import matplotlib.pyplot as plt
    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x = jnp.geomspace(.001, .65, 160)
    for recipe, color in zip(melts["recipes"][::3], colors):
        dry = np.array(recipe["n_mol"][:-1])
        dry /= dry.sum()
        # Each derivative holds the supplied dry amounts fixed.
        values = jax.jit(jax.vmap(MU_LIQUID, in_axes=(None, None, 0, 0)))(1700., 5e7, (1-x[:, None])*dry, x)
        name = recipe["name"].rsplit("_", 1)[0].replace("_", " ")
        axes[0].plot(x, values/1000, color=color, label=f"Liquid: {name}")
    axes[0].axhline(float(MU_FLUID(1700., 5e7, 1.))/1000, color="black", label="Fluid: ZD09")
    axes[0].axhline(float(MU_IDEAL(1700., 5e7, 1.))/1000, color="black", ls="--", label="Fluid: ideal control")
    axes[0].set(xscale="log", xlabel=r"Liquid H$_2$O component mole fraction $x_w$",
                ylabel=r"$\mu_w$ (kJ mol$^{-1}$)", title="Chemical potentials at 1700 K, 500 bar")
    pressures = jnp.geomspace(1e6, 5e8, 100)
    fluid = jax.jit(jax.vmap(MU_FLUID, in_axes=(None, 0, None)))(1700., pressures, 1.)
    for recipe, color in zip(melts["recipes"][3:6], colors):
        n = np.array(recipe["n_mol"])
        values = jax.jit(jax.vmap(MU_LIQUID, in_axes=(None, 0, None, None)))(1700., pressures, n[:-1], n[-1])
        axes[1].plot(pressures/1e5, (fluid-values)/1000, color=color, label=f"{recipe['water_wt_percent']:g} wt% H2O")
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set(xscale="log", xlabel="Pressure (bar)", ylabel=r"$\mu_w^{fluid}-\mu_w^{liquid}$ (kJ mol$^{-1}$)",
                title="Fixed basalt-like recipes at 1700 K")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Supplied compositions: chemical-potential comparison\n"
                 "Positive difference raises G for liquid-to-fluid transfer. No saturation or phase-amount solve.", fontsize=11)
    fig.savefig(args.output/"fluid_chemical_potentials.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    temperatures = jnp.linspace(700., 2500., 150)
    axes[0, 0].plot(temperatures, jax.jit(jax.vmap(water_gas_standard_gibbs))(temperatures)/1000, color=colors[0], label="JAX ideal standard")
    axes[0, 0].plot([r["T_K"] for r in reference["standards"]], [r["g_J_mol"]/1000 for r in reference["standards"]], "o", color=colors[1], mfc="white", label="Compiled Haar dilute limit")
    axes[0, 0].set(title="Explicit ideal gas reference at 1 bar", xlabel="Temperature (K)", ylabel=r"$g_w^\circ$ (kJ mol$^{-1}$)")
    for t, color in zip((700., 1700., 2500.), colors):
        values = jax.jit(jax.vmap(MU_FLUID, in_axes=(None, 0, None)))(t, pressures, 1.)
        axes[0, 1].plot(pressures/1e5, (values-water_gas_standard_gibbs(t))/1000, color=color, label=f"{t:g} K")
        rows = [r for r in reference["states"] if r["T_K"] == t and r["P_Pa"] > 1e5]
        standard = next(r["g_J_mol"] for r in reference["standards"] if r["T_K"] == t)
        axes[0, 1].plot([r["P_Pa"]/1e5 for r in rows], [(r["g_J_mol"]-standard)/1000 for r in rows], "o", color=color, mfc="white")
    axes[0, 1].set(title="Fluid pressure contribution / independent points", xscale="log", xlabel="Pressure (bar)", ylabel=r"$\mu_w-g_w^\circ$ (kJ mol$^{-1}$)")
    ts = sorted({r["T_K"] for r in reference["states"]})
    for index, color, label in ((0, colors[0], "G / mu (J/mol)"), (1, colors[1], "S (J/mol/K)"), (2, colors[2], "Cp (J/mol/K)")):
        maximum = [max(abs(errors[i, index]) for i, r in enumerate(reference["states"]) if r["T_K"] == t) for t in ts]
        axes[1, 0].plot(ts, np.maximum(maximum, 1e-13), "o-", color=color, ms=3, label=label)
    axes[1, 0].set(title="Maximum absolute fluid errors", yscale="log", xlabel="Temperature (K)", ylabel="Absolute error (units in legend)")
    v_errors = abs(errors[:, 3]/actual[:, 3])
    axes[1, 1].plot(ts, [max(v_errors[i] for i, r in enumerate(reference["states"]) if r["T_K"] == t) for t in ts], "o-", color=colors[0], ms=3, label="V = dG/dP vs independent 1/rho")
    axes[1, 1].set(title="Volume from the same scalar G", xlabel="Temperature (K)", ylabel="Maximum relative error")
    axes[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Independent numerical verification: compiled MAGMA standard + ZD09 quadrature\n"
                 "Error panels include 1, 500, 5000 bar; 1 bar is a low-pressure numerical control.", fontsize=11)
    fig.savefig(args.output/"fluid_validation.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
