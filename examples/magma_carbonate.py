"""Plot mode-4 internal reaction, binary fluid potentials and reference errors.

All kernels/coefficients come from installed ExoEOS. Saved fixtures supply
independent comparison points only; no external MELTS runtime is needed.
These are supplied-state property controls, not saturation/phase equilibria.
"""

import argparse
from functools import partial
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exoeos import ZhangDuanEOS, magma, magma_carbonate as liquid
from exoeos.magma_fluid import FLUID_COMPONENTS, mixed_fluid_gibbs


jax.config.update("jax_enable_x64", True)
FIELDS = ("g_J", "s_J_K", "cp_J_K", "v_m3", "dvdt_m3_K", "dvdp_m3_Pa")
FLUID = partial(mixed_fluid_gibbs, eos=ZhangDuanEOS.from_species(FLUID_COMPONENTS))


def properties(fun, t, p, n):
    return jnp.array([fun(t, p, n), -jax.grad(fun, 0)(t, p, n),
                      -t*jax.grad(jax.grad(fun, 0), 0)(t, p, n),
                      jax.grad(fun, 1)(t, p, n),
                      jax.grad(jax.grad(fun, 1), 0)(t, p, n),
                      jax.grad(jax.grad(fun, 1), 1)(t, p, n)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/magma_carbonate"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]/"tests/reference"
    reference = json.loads((root/"carbonate_melts_v1.json").read_text())
    source = json.loads((root/"carbonate_source_v1.json").read_text())
    prop = jax.jit(partial(properties, liquid.liquid_gibbs))
    fprop = jax.jit(partial(properties, FLUID))
    mu = jax.jit(jax.grad(liquid.liquid_gibbs, 2))
    fmu = jax.jit(jax.grad(FLUID, 2))
    species = jax.jit(liquid.liquid_species_moles)
    states = reference["states"]
    actual = np.array([prop(r["T_K"], r["P_Pa"], jnp.array(r["n_mol"])) for r in states])
    expected = np.array([[r[k] for k in FIELDS] for r in states])
    errors = actual-expected
    interiors = [r for r in states if min(r["n_mol"]) > 0 and r["name"] != "trace_carbon"]
    mu_errors = np.array([mu(r["T_K"], r["P_Pa"], jnp.array(r["n_mol"]))-np.array(r["mu_J_mol"]) for r in interiors])
    species_errors = np.array([species(r["T_K"], r["P_Pa"], jnp.array(r["n_mol"]))-np.array(r["species_mol"]) for r in states])
    fluid_errors = np.array([fprop(r["T_K"], r["P_Pa"], jnp.array(r["n_mol"]))-np.array([r[k] for k in FIELDS]) for r in source["fluid_states"]])
    n = jnp.array(reference["recipes"][0]["n_mol"])
    t, p = 1700., 5e7
    w = liquid._interaction_matrix()
    def unrelaxed(t, p, n, xi):
        return n@liquid.liquid_standard_gibbs(t, p)+n.sum()*liquid._mixing_gibbs(t, p, n/n.sum(), xi/n.sum(), w)
    frozen_cp = jax.jit(lambda t, p, n, xi: -t*jax.grad(jax.grad(unrelaxed, 0), 0)(t, p, n, xi))
    temps = np.linspace(1200, 2200, 81)
    extent = np.array([species(ti, p, n)[-1] for ti in temps])
    cp = np.array([prop(ti, p, n)[2] for ti in temps])
    frozen = np.array([frozen_cp(ti, p, n, xi) for ti, xi in zip(temps, extent)])
    summary = {
        "scope": "Mode-4 supplied liquid; Haar/CO2 ideal standards plus selected ZD09 binary fluid. No saturation or phase-amount solution, no S/N extension.",
        "liquid_states": len(states), "interior_mu_states": len(interiors), "source_standard_states": len(source["standards"]),
        "fluid_states": len(fluid_errors), "liquid_maximum_absolute_errors": dict(zip(FIELDS, np.max(abs(errors), axis=0).tolist())),
        "interior_mu_maximum_absolute_error_J_mol": float(np.max(abs(mu_errors))),
        "species_maximum_absolute_error_mol": float(np.max(abs(species_errors))),
        "fluid_maximum_absolute_errors": dict(zip(FIELDS, np.max(abs(fluid_errors), axis=0).tolist())),
        "maximum_Cp_relaxation_J_K": float(max(cp-frozen)),
        "zero_carbon_mode4_minus_mode1_J": float(liquid.liquid_gibbs(t, p, n.at[-1].set(0))-magma.liquid_gibbs(t, p, n[:12])),
        "unavailable_runtime_cases": reference["unavailable_cases"],
    }
    trace = next(r for r in states if r["name"] == "trace_carbon")
    tn = jnp.array(trace["n_mol"])
    summary["trace_carbon_diagnostic"] = {
        "n_CO2_mol": float(tn[-1]), "MELTS_carbonate_mol": trace["species_mol"][-1],
        "JAX_carbonate_mol": float(species(trace["T_K"], trace["P_Pa"], tn)[-1]),
        "JAX_minus_MELTS_mu_CO2_J_mol": float(mu(trace["T_K"], trace["P_Pa"], tn)[-1]-trace["mu_J_mol"][-1]),
        "interpretation": "Pinned runtime leaves carbonate at zero at trace carbon; exclude this trace potential from the regular-interior agreement statistic. Preserve raw reference; native trace stationarity and dilution are checked separately.",
    }
    (args.output/"carbonate_comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2))

    import matplotlib.pyplot as plt
    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    xi = np.linspace(0, float(n[-1]), 201)
    unrel = jax.jit(unrelaxed)
    minimum = float(liquid.liquid_gibbs(t, p, n))
    values = np.array([unrel(t, p, n, s) for s in xi])-minimum
    axes[0, 0].plot(xi/n[-1], values, color=colors[0])
    axes[0, 0].plot(float(species(t, p, n)[-1]/n[-1]), 0, "o", color=colors[1], label="Relaxed minimum")
    axes[0, 0].set(xlabel="Fraction of total C as carbonate", ylabel="G - minimum G [J]", title="Internal reaction at 1700 K, 500 bar")
    axes[0, 0].legend()
    for pressure, color in zip((1e5, 5e7, 5e8), colors):
        fractions = np.array([species(ti, pressure, n)[-1]/n[-1] for ti in temps])
        axes[0, 1].plot(temps, fractions, color=color, label=f"{pressure/1e5:g} bar")
        rows = [r for r in states if r["name"] == "basalt" and r["P_Pa"] == pressure]
        axes[0, 1].plot([r["T_K"] for r in rows], [r["species_mol"][-1]/r["n_mol"][-1] for r in rows], "o", mfc="none", color=color)
    axes[0, 1].set(xlabel="Temperature [K]", ylabel="Carbonate / total carbon", title="Lines: JAX; circles: pinned MELTS mode 4")
    axes[0, 1].legend()
    axes[1, 0].plot(temps, cp-frozen, color=colors[1])
    axes[1, 0].set(xlabel="Temperature [K]", ylabel="Relaxed Cp - frozen-reaction Cp [J/K]", title="Reaction response in the second derivative")
    xs = np.linspace(.01, .99, 99)
    potentials = np.array([fmu(t, p, jnp.array([x, 1-x])) for x in xs])/1000
    liquid_mu = np.asarray(mu(t, p, n)[-2:])/1000
    for i, (label, color) in enumerate(zip(FLUID_COMPONENTS, colors)):
        axes[1, 1].plot(xs, potentials[:, i], color=color, label=f"Fluid {label}")
        axes[1, 1].axhline(liquid_mu[i], ls="--", color=color, label=f"Supplied liquid {label}")
    axes[1, 1].set(xlabel="Fluid H2O mole fraction", ylabel="Chemical potential [kJ/mol]", title="Haar/CO2 standards + ZD09, 1700 K, 500 bar")
    axes[1, 1].legend(fontsize=8)
    fig.savefig(args.output/"carbonate_model.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for row, (name, color) in enumerate(zip(("basalt", "alkali_rich", "carbon_rich"), colors)):
        indices = [i for i, r in enumerate(states) if r["name"] == name and r["P_Pa"] == p]
        ti = [states[i]["T_K"] for i in indices]
        for ax, k, factor, label in zip(axes[0], (0, 2, 3), (1e-6, 1., 1e6), ("G [MJ]", "Cp [J/K]", "V [cm3]")):
            ax.plot(ti, actual[indices, k]*factor, color=color, label=name.replace("_", " "))
            ax.plot(ti, expected[indices, k]*factor, "o", mfc="none", color=color)
            ax.set(xlabel="Temperature [K]", ylabel=label)
        chosen = [i for i, r in enumerate(states) if r["name"] == name]
        axes[1, 0].plot([states[i]["T_K"] for i in chosen], errors[chosen, 0], "o", color=color)
        chosen_mu = [i for i, r in enumerate(interiors) if r["name"] == name]
        axes[1, 1].plot([interiors[i]["T_K"] for i in chosen_mu], np.max(abs(mu_errors[chosen_mu]), axis=1), "o", color=color)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set(xlabel="Temperature [K]", ylabel="JAX - MELTS G [J]")
    axes[1, 1].set(xlabel="Temperature [K]", ylabel="Maximum |mu error| [J/mol]")
    for pressure, color in zip((1e5, 5e7, 5e8), colors):
        ids = [i for i, r in enumerate(source["fluid_states"]) if r["P_Pa"] == pressure]
        axes[1, 2].plot([source["fluid_states"][i]["T_K"] for i in ids], fluid_errors[ids, 2], "o", color=color, label=f"{pressure/1e5:g} bar")
    axes[1, 2].set(xlabel="Temperature [K]", ylabel="Fluid Cp error [J/K]")
    axes[1, 2].legend(fontsize=8)
    fig.suptitle("Supplied-state validation: lines = native JAX; circles = independent references")
    fig.savefig(args.output/"carbonate_validation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
