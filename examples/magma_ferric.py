"""Plot supplied ferric liquid properties and the zero-Fe2O3 limit.

Uses saved independent MELTS references; no external runtime or equilibrium
solver. The dilution series holds the original five amounts fixed, so it is
a property limit, not a closed-system redox or crystallization path.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exoeos.magma import FERRIC_COMPONENTS, liquid_gibbs, liquid_standard_gibbs


jax.config.update("jax_enable_x64", True)
FIELDS = ("g_J", "s_J_K", "cp_J_K", "v_m3", "dvdt_m3_K", "dvdp_m3_Pa")
STANDARDS = jax.jit(lambda t, p: liquid_standard_gibbs(
    t, p, include_fe=True, include_ca_al=True, include_ferric=True))
MU = jax.jit(jax.grad(liquid_gibbs, 2))


@jax.jit
def properties(t, p, n):
    return jnp.array([
        liquid_gibbs(t, p, n), -jax.grad(liquid_gibbs, 0)(t, p, n),
        -t*jax.grad(jax.grad(liquid_gibbs, 0), 0)(t, p, n),
        jax.grad(liquid_gibbs, 1)(t, p, n),
        jax.grad(jax.grad(liquid_gibbs, 1), 0)(t, p, n),
        jax.grad(jax.grad(liquid_gibbs, 1), 1)(t, p, n),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/magma_ferric"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = json.loads((Path(__file__).resolve().parents[1]
                            / "tests/reference/ferric_melts_v1.json").read_text())
    errors, mu_errors, standard_errors = [], [], []
    for row in reference["states"]:
        t, p, n = row["T_K"], row["P_Pa"], np.array(row["n_mol"])
        errors.append(np.asarray(properties(t, p, n))-np.array([row[k] for k in FIELDS]))
        present = n > 0
        standard_errors.extend((np.asarray(STANDARDS(t, p))
                                -np.array(row["standard_J_mol"], float))[present])
        if np.all(present):
            mu_errors.append(max(abs(np.asarray(MU(t, p, n))-row["mu_J_mol"])))
    maximum = dict(zip(FIELDS, np.max(np.abs(errors), axis=0).tolist()))
    maximum.update(mu_J_mol=max(mu_errors), standard_J_mol=max(abs(np.array(standard_errors))))
    n0 = jnp.array([.4, .275, .075, .25, .12])
    t, p = 1700., 5e7
    baseline, mu0 = float(liquid_gibbs(t, p, n0)), MU(t, p, n0)
    epsilon = np.logspace(-12, -1, 80)
    gdiff, mudiff, ferric_mu = [], [], []
    for amount in epsilon:
        n = jnp.r_[n0, amount]
        gdiff.append(abs(float(liquid_gibbs(t, p, n))-baseline))
        mu = MU(t, p, n)
        mudiff.append(float(max(abs(mu[:5]-mu0))))
        ferric_mu.append(float(mu[5]))
    zero_error = np.asarray(properties(t, p, jnp.r_[n0, 0.])-properties(t, p, n0))
    summary = {
        "scope": "Supplied six-component liquid properties; no redox or assemblage equilibrium, magnetite or other ferric solids.",
        "component_order": FERRIC_COMPONENTS, "backend": reference["backend"],
        "native_property_states": len(errors), "native_interior_mu_states": len(mu_errors),
        "maximum_absolute_errors": maximum,
        "native_unavailable_cases": reference["unavailable_cases"],
        "zero_ferric": {"T_K": t, "P_Pa": p, "five_component_mol": n0.tolist(),
                        "absolute_errors": dict(zip(FIELDS, abs(zero_error).tolist())),
                        "minimum_positive_ferric_mol": float(epsilon[0]),
                        "G_limit_error_J": gdiff[0], "present_mu_limit_error_J_mol": mudiff[0]},
    }
    (args.output/"ferric_comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")

    import matplotlib.pyplot as plt
    colors = ("#2166ac", "#d89021", "#218e68")
    grid = jnp.linspace(1200., 2200., 121)
    pure = jnp.array([0., 0., 0., 0., 0., 1.])
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.3), constrained_layout=True)
    for pressure, color in zip((1e5, 5e7, 5e8), colors):
        label = f"{pressure/1e5:g} bar"
        curves = np.asarray(jax.vmap(properties, in_axes=(0, None, None))(grid, pressure, pure))
        rows = [r for r in reference["states"] if r["P_Pa"] == pressure and r["n_mol"] == pure.tolist()]
        for ax, index, scale in zip(axes[0], (0, 1, 3), (1e-3, 1., 1e6)):
            ax.plot(grid, curves[:, index]*scale, color=color, label=label)
            ax.plot([r["T_K"] for r in rows], [r[FIELDS[index]]*scale for r in rows],
                    "o", color=color, mfc="white", ms=4)
        for column, field in enumerate(("g_J", "mu_J_mol", "cp_J_K")):
            ts, maxima = [], []
            for temperature in sorted({r["T_K"] for r in rows}):
                indices = [i for i, r in enumerate(reference["states"])
                           if r["T_K"] == temperature and r["P_Pa"] == pressure]
                if field == "mu_J_mol":
                    candidates = [reference["states"][i] for i in indices
                                  if min(reference["states"][i]["n_mol"]) > 0]
                    value = max(max(abs(np.asarray(MU(temperature, pressure, np.array(r["n_mol"])))
                                        -r["mu_J_mol"])) for r in candidates)
                else:
                    value = max(abs(errors[i][FIELDS.index(field)]) for i in indices)
                ts.append(temperature)
                maxima.append(value)
            axes[1, column].plot(ts, maxima, "o-", color=color, label=label, ms=4)
    for ax, title, ylabel in zip(axes.flat,
            ("Fe2O3 liquid standard", "Fe2O3 liquid entropy", "Fe2O3 liquid volume",
             "Maximum total-G error", "Maximum interior-mu error", "Maximum heat-capacity error"),
            ("G (kJ/mol)", "S (J/mol/K)", "V (cm3/mol)", "|JAX - MELTS| (J)",
             "|JAX - MELTS| (J/mol)", "|JAX - MELTS| (J/K)")):
        ax.set(title=title, xlabel="Temperature (K)", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_ylim(bottom=0.)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
    fig.suptitle("Fe(III) liquid properties: native JAX and independent MELTS\n"
                 "Top: curves = JAX, circles = MELTS. Bottom: largest sampled discrepancy at each T/P.", fontsize=11)
    fig.savefig(args.output/"ferric_validation.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, values, ylabel in zip(axes, (gdiff, mudiff, np.array(ferric_mu)/1e3),
                                  ("|G6 - G5| (J)", "max |mu6[:5] - mu5| (J/mol)", "Ferric mu (kJ/mol)")):
        ax.plot(epsilon, values, color=colors[0], lw=2)
        ax.set(xscale="log", xlabel="Fe2O3 amount (mol)", ylabel=ylabel)
        ax.grid(alpha=.2)
    for ax in axes[:2]:
        ax.set_yscale("log")
    fig.suptitle("Zero-ferric limit at 1700 K and 500 bar; the original five amounts stay fixed\n"
                 "G and the five present chemical potentials converge; ferric mu has a logarithmic divergence.", fontsize=11)
    fig.savefig(args.output/"ferric_zero_limit.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
