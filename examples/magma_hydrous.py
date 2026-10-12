"""Plot native hydrous-liquid properties and independent MELTS comparisons.

No external MELTS runtime is required. Compositions are supplied, with no
fluid saturation, crystallization or redox solve. Basalt/rhyolite are
synthetic controls; water mass percent uses the complete liquid mass.
"""

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exoeos.magma import DRY11_COMPONENTS, R, liquid_gibbs, liquid_standard_gibbs


jax.config.update("jax_enable_x64", True)
FIELDS = ("g_J", "s_J_K", "cp_J_K", "v_m3", "dvdt_m3_K", "dvdp_m3_Pa")
G = jax.jit(liquid_gibbs)
MU = jax.jit(jax.grad(liquid_gibbs, 2))
STANDARD = jax.jit(lambda t, p: liquid_standard_gibbs(t, p, include_h2o=True))


def _properties(fun, t, p, *args):
    return jnp.array([fun(t, p, *args), -jax.jacfwd(fun, 0)(t, p, *args),
                      -t*jax.jacfwd(jax.jacfwd(fun, 0), 0)(t, p, *args),
                      jax.jacfwd(fun, 1)(t, p, *args),
                      jax.jacfwd(jax.jacfwd(fun, 1), 0)(t, p, *args),
                      jax.jacfwd(jax.jacfwd(fun, 1), 1)(t, p, *args)])


PROPERTIES = jax.jit(lambda t, p, n: _properties(liquid_gibbs, t, p, n))
STANDARD_PROPERTIES = jax.jit(lambda t, p: _properties(STANDARD, t, p))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/magma_hydrous"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root/"tests/reference/hydrous_melts_v1.json").read_text())
    source = json.loads((root/"tests/reference/hydrous_source_v1.json").read_text())
    rows = reference["states"]
    errors, mu_errors, standard_errors = [], [], []
    for row in rows:
        t, p, n = row["T_K"], row["P_Pa"], np.array(row["n_mol"])
        errors.append(np.asarray(PROPERTIES(t, p, n))-np.array([row[k] for k in FIELDS]))
        present = n > 0
        standard_errors.extend((np.asarray(STANDARD(t, p))-np.array(row["standard_J_mol"], float))[present])
        mu_errors.append(float(max(abs(MU(t, p, n)-np.array(row["mu_J_mol"])))) if np.all(present) else None)
    errors = np.array(errors)
    upstream = [np.asarray(STANDARD_PROPERTIES(r["T_K"], r["P_Pa"]))[:, -1]-np.array([r[k] for k in FIELDS])
                for r in source["states"]]
    maximum = dict(zip(FIELDS, np.max(abs(errors), axis=0).tolist()))
    maximum.update(mu_J_mol=max(v for v in mu_errors if v is not None),
                   standard_J_mol=float(max(abs(np.array(standard_errors)))))
    summary = {"scope": "Supplied hydrous liquid; fixed Fe(II)/Fe(III), no equilibrium or empirical water correction.",
               "component_order": reference["component_order"], "backend": reference["backend"],
               "native_states": len(rows), "native_interior_mu_states": sum(v is not None for v in mu_errors),
               "compiled_standard_states": len(upstream), "water_pair_count": len(source["pairs"]),
               "maximum_absolute_errors": maximum,
               "compiled_standard_maximum_absolute_errors": dict(zip(FIELDS, np.max(abs(np.array(upstream)), axis=0).tolist())),
               "recipes": reference["recipes"], "unavailable_cases": reference["unavailable_cases"]}
    (args.output/"hydrous_comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    with (args.output/"hydrous_interactions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source["pairs"][0]))
        writer.writeheader()
        writer.writerows(source["pairs"])

    import matplotlib.pyplot as plt
    colors = ("#0072B2", "#D55E00", "#009E73")
    temperatures = jnp.linspace(1200., 2200., 101)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for recipe, color in zip(reference["recipes"][3:6], colors):
        n = np.array(recipe["n_mol"])
        values = np.asarray(jax.vmap(PROPERTIES, in_axes=(0, None, None))(temperatures, 5e7, n))
        samples = [r for r in rows if r["name"] == recipe["name"] and r["P_Pa"] == 5e7]
        for ax, field, scale in zip(axes[0], (0, 1, 3), (1e-3, 1., 1e6)):
            ax.plot(temperatures, values[:, field]*scale, color=color, label=f"{recipe['water_wt_percent']:g} wt% H2O")
            ax.plot([r["T_K"] for r in samples], [r[FIELDS[field]]*scale for r in samples], "o", mfc="white", color=color, ms=4)
    for host, color in zip(("BSE", "basalt_like", "rhyolite_like"), colors):
        for ax, field in zip(axes[1], ("g_J", "mu_J_mol", "cp_J_K")):
            tx, residuals = [], []
            for t in (1200., 1479., 1481., 1800., 2200.):
                selection = [i for i, r in enumerate(rows) if r["name"].startswith(host+"_") and r["T_K"] == t]
                values = [mu_errors[i] for i in selection if mu_errors[i] is not None] if field == "mu_J_mol" else abs(errors[selection, FIELDS.index(field)]).tolist()
                if values:
                    tx.append(t)
                    residuals.append(max(values))
            if tx:
                ax.plot(tx, residuals, "o-", color=color, ms=4, label=host.replace("_", " "))
    titles = ("Basalt-like liquid G", "Basalt-like entropy", "Basalt-like volume",
              "Max G error", "Max interior chemical-potential error", "Max heat-capacity error")
    labels = ("G (kJ / 100 g)", "S (J / K / 100 g)", "V (cm3 / 100 g)",
              "|JAX - MELTS| (J / 100 g)", "|JAX - MELTS| (J/mol)", "|JAX - MELTS| (J / K / 100 g)")
    for ax, title, ylabel in zip(axes.flat, titles, labels):
        ax.set(title=title, xlabel="Temperature (K)", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_ylim(bottom=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
    fig.suptitle("Hydrous twelve-component liquid: supplied compositions\n"
                 "Top: 500 bar, JAX curves and MELTS circles. Bottom: max over 1/500/5000 bar and 0.1/2/5 wt% water.", fontsize=11)
    fig.savefig(args.output/"hydrous_validation.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for p, color in zip((1e5, 5e7, 5e8), colors):
        standard = np.asarray(jax.vmap(STANDARD, in_axes=(0, None))(temperatures, p))[:, -1]
        axes[0, 0].plot(temperatures, standard/1000, color=color, label=f"{p/1e5:g} bar")
        samples = [r for r in source["states"] if r["P_Pa"] == p and 1200 <= r["T_K"] <= 2200]
        axes[0, 0].plot([r["T_K"] for r in samples], [r["g_J"]/1000 for r in samples], "o", mfc="white", color=color, ms=4)
    axes[0, 0].set(title="H2O standard: JAX / compiled MAGMA", xlabel="Temperature (K)", ylabel="g0 H2O (kJ/mol)")
    x = np.linspace(.001, .999, 200)
    extra = R*1700*(x*np.log(x)+(1-x)*np.log1p(-x))/1000
    axes[0, 1].plot(x, extra, color=colors[0], label="Additional water mixing term")
    axes[0, 1].plot(x, 2*extra, "--", color=colors[1], label="Total ideal term on a dry/water binary")
    axes[0, 1].set(title="Water entropy contribution at 1700 K", xlabel="Water component mole fraction", ylabel="G mixing (kJ / mol components)")
    x = jnp.logspace(-8, -.15, 100)
    for recipe, color in zip(reference["recipes"][::3], colors):
        dry = np.array(recipe["n_mol"][:11])
        # Positive-host controls for composition derivatives; BSE Fe2O3=0
        # uses only the scalar water derivative, with dry amounts held fixed.
        dry /= dry.sum()
        def fun(nw, nd):
            return liquid_gibbs(1700., 5e7, jnp.r_[nd, nw])
        water_mu = jax.jit(jax.vmap(jax.grad(fun), in_axes=(0, 0)))(x, (1-x[:, None])*dry)
        axes[1, 0].plot(x, (water_mu-STANDARD(1700., 5e7)[-1])/1000, color=color, label=recipe["name"].rsplit("_", 1)[0].replace("_", " "))
    axes[1, 0].plot(x, 2*R*1700*jnp.log(x)/1000, "k--", lw=1, label="2 RT ln(xw)")
    axes[1, 0].set(title="Dissolved-water chemical potential", xlabel="Water component mole fraction", ylabel="mu H2O - g0 H2O (kJ/mol)", xscale="log")
    pair = {r["component"]: r["H_J_mol"]/1000 for r in source["pairs"]}
    values = [pair[name] for name in DRY11_COMPONENTS]
    axes[1, 1].barh(DRY11_COMPONENTS, values, color=[colors[0] if value >= 0 else colors[1] for value in values])
    axes[1, 1].axvline(0, color="k", lw=.7)
    axes[1, 1].set(title="All eleven H2O interactions", xlabel="W(i, H2O) (kJ/mol)")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.legend(fontsize=8)
    fig.suptitle("MELTS water standard, extra mixing entropy and host interactions\n"
                 "Formula comparisons and specified liquid properties; no solubility or phase-equilibrium fit.", fontsize=11)
    fig.savefig(args.output/"hydrous_model.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"native_states": len(rows), "maximum_absolute_errors": maximum}, indent=2))


if __name__ == "__main__":
    main()
