"""Plot dry eleven-component properties, independent errors and reduced faces.

Reads saved MELTS/source fixtures. No external runtime or equilibrium solve.
Non-BSE recipes are synthetic controls, not experimental rock analyses.
"""

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exoeos.magma import (
    COMPONENTS, TERNARY_COMPONENTS, CMAS_COMPONENTS, CMFAS_COMPONENTS,
    FERRIC_COMPONENTS, DRY11_COMPONENTS, DRY11_OXIDES, ELEMENTS,
    element_matrix, liquid_gibbs, liquid_standard_gibbs,
    liquid_moles_from_oxides, oxide_moles_from_liquid,
)


jax.config.update("jax_enable_x64", True)
FIELDS = ("g_J", "s_J_K", "cp_J_K", "v_m3", "dvdt_m3_K", "dvdp_m3_Pa")
MU = jax.jit(jax.grad(liquid_gibbs, 2))
STANDARD = jax.jit(lambda t, p: liquid_standard_gibbs(t, p, include_dry11=True))


@jax.jit
def properties(t, p, n):
    return jnp.array([liquid_gibbs(t, p, n), -jax.grad(liquid_gibbs, 0)(t, p, n),
                      -t*jax.grad(jax.grad(liquid_gibbs, 0), 0)(t, p, n),
                      jax.grad(liquid_gibbs, 1)(t, p, n),
                      jax.grad(jax.grad(liquid_gibbs, 1), 0)(t, p, n),
                      jax.grad(jax.grad(liquid_gibbs, 1), 1)(t, p, n)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/magma_dry11"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root/"tests/reference/dry11_melts_v1.json").read_text())
    source = json.loads((root/"tests/reference/dry11_source_v1.json").read_text())
    rows = reference["states"]
    errors, mu_errors, standard_errors = [], [], []
    for row in rows:
        t, p, n = row["T_K"], row["P_Pa"], np.array(row["n_mol"])
        errors.append(np.asarray(properties(t, p, n))-np.array([row[k] for k in FIELDS]))
        present = n > 0
        standard_errors.extend((np.asarray(STANDARD(t, p))-np.array(row["standard_J_mol"], float))[present])
        mu_errors.append(float(max(abs(MU(t, p, n)-np.array(row["mu_J_mol"])))) if np.all(present) else None)
    errors = np.asarray(errors)
    maximum = dict(zip(FIELDS, np.max(abs(errors), axis=0).tolist()))
    maximum.update(mu_J_mol=max(v for v in mu_errors if v is not None),
                   standard_J_mol=float(max(abs(np.array(standard_errors)))))
    recipes = []
    for recipe in reference["recipes"]:
        n, m = np.array(recipe["n_mol"]), np.array(recipe["oxide_mol"])
        selection = [i for i, r in enumerate(rows) if r["name"] == recipe["name"]]
        inventory_error = element_matrix(DRY11_COMPONENTS)@liquid_moles_from_oxides(m)-element_matrix(DRY11_OXIDES)@m
        recipes.append({**recipe, "maximum_absolute_errors": dict(zip(FIELDS, np.max(abs(errors[selection]), axis=0).tolist())),
                        "max_atom_error_mol": float(max(abs(inventory_error))),
                        "max_oxide_roundtrip_error_mol": float(max(abs(oxide_moles_from_liquid(n)-m)))})
    bases = [COMPONENTS, TERNARY_COMPONENTS, CMAS_COMPONENTS, CMFAS_COMPONENTS, FERRIC_COMPONENTS]
    epsilon = np.logspace(-12, -2, 65)
    faces = []
    for basis in bases:
        n = jnp.linspace(.07, .43, len(basis))
        indices = [DRY11_COMPONENTS.index(name) for name in basis]
        full = np.zeros(11)
        full[indices] = n
        absent = full == 0
        base = np.asarray(properties(1700., 5e7, n))
        error = np.asarray(properties(1700., 5e7, full))-base
        mu0 = MU(1700., 5e7, n)
        limits = [float(max(abs(np.asarray(MU(1700., 5e7, full+amount*absent))[indices]-mu0))) for amount in epsilon]
        faces.append({"basis": basis, "absolute_errors": dict(zip(FIELDS, abs(error).tolist())),
                      "maximum_relative_error": float(max(abs(error)/np.maximum(abs(base), 1e-30))),
                      "dilution_mol": epsilon.tolist(), "present_mu_error_J_mol": limits})
    summary = {"scope": "Supplied dry-liquid properties; fixed Fe(II)/Fe(III), no phase or redox equilibrium.",
               "component_order": DRY11_COMPONENTS, "oxide_order": DRY11_OXIDES, "element_order": ELEMENTS,
               "backend": reference["backend"], "native_states": len(rows),
               "native_interior_mu_states": sum(v is not None for v in mu_errors),
               "all_pair_count": len(source["pairs"]), "maximum_absolute_errors": maximum,
               "recipes": recipes, "faces": faces, "unavailable_cases": reference["unavailable_cases"]}
    (args.output/"dry11_comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    with (args.output/"dry11_interactions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["component_i", "component_j", "H_J_mol", "S_J_mol_K", "V_J_mol_bar"])
        for pair in source["pairs"]:
            writer.writerow([*pair["components"], pair["H_J_mol"], pair["S_J_mol_K"], pair["V_J_mol_bar"]])

    import matplotlib.pyplot as plt
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    grid = jnp.linspace(1200., 2200., 121)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.3), constrained_layout=True)
    for recipe, color in zip(recipes, colors):
        name, n = recipe["name"], np.array(recipe["n_mol"])
        label = name.replace("_", " ")
        curves = np.asarray(jax.vmap(properties, in_axes=(0, None, None))(grid, 5e7, n))
        samples = [r for r in rows if r["name"] == name and r["P_Pa"] == 5e7]
        for ax, index, scale in zip(axes[0], (0, 1, 3), (1e-3, 1., 1e6)):
            ax.plot(grid, curves[:, index]*scale, color=color, label=label)
            ax.plot([r["T_K"] for r in samples], [r[FIELDS[index]]*scale for r in samples],
                    "o", color=color, mfc="white", ms=4)
        for column, field in enumerate(("g_J", "mu_J_mol", "cp_J_K")):
            temperatures, values = [], []
            for t in sorted({r["T_K"] for r in samples}):
                selection = [i for i, r in enumerate(rows) if r["name"] == name and r["T_K"] == t]
                if field == "mu_J_mol":
                    candidates = [mu_errors[i] for i in selection if mu_errors[i] is not None]
                    if not candidates:  # BSE has exactly zero Fe2O3.
                        continue
                    value = max(candidates)
                else:
                    value = max(abs(errors[selection, FIELDS.index(field)]))
                temperatures.append(t)
                values.append(value)
            if values:
                axes[1, column].plot(temperatures, values, "o-", color=color, label=label, ms=4)
    titles = ("Liquid G at 500 bar", "Entropy at 500 bar", "Volume at 500 bar",
              "G error: max over pressures", "Interior mu error: max over pressures", "Cp error: max over pressures")
    labels = ("G (kJ / 100 g)", "S (J / K / 100 g)", "V (cm3 / 100 g)",
              "|JAX - MELTS| (J / 100 g)", "|JAX - MELTS| (J/mol)", "|JAX - MELTS| (J / K / 100 g)")
    for ax, title, ylabel in zip(axes.flat, titles, labels):
        ax.set(title=title, xlabel="Temperature (K)", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_ylim(bottom=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
    fig.suptitle("Dry eleven-component liquid: BSE and three synthetic controls\n"
                 "Curves: native JAX. Circles: pinned MELTS. Error pressures: 1, 500, 5000 bar. Supplied compositions.", fontsize=11)
    fig.savefig(args.output/"dry11_validation.png", dpi=180)
    plt.close(fig)

    matrix = np.full((11, 11), np.nan)
    for pair in source["pairs"]:
        i, j = sorted(DRY11_COMPONENTS.index(name) for name in pair["components"])
        matrix[i, j] = pair["H_J_mol"]/1000
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), width_ratios=[1.7, 1, 1], constrained_layout=True)
    im = axes[0].imshow(matrix, cmap="RdBu_r", vmin=-180, vmax=180)
    for i in range(11):
        for j in range(i+1, 11):
            axes[0].text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", fontsize=6)
    axes[0].set(xticks=range(11), yticks=range(11), xticklabels=DRY11_COMPONENTS,
                yticklabels=DRY11_COMPONENTS, title="All 55 pair interactions (kJ/mol)")
    axes[0].tick_params(axis="x", labelrotation=90, labelsize=8)
    axes[0].tick_params(axis="y", labelsize=8)
    fig.colorbar(im, ax=axes[0], shrink=.65)
    labels = ("2", "3", "4 (CMAS)", "5", "6")
    for i, (face, color, label) in enumerate(zip(faces, colors, labels)):
        axes[1].plot(i, max(face["maximum_relative_error"], 1e-17), "o", color=color)
        axes[2].loglog(epsilon, face["present_mu_error_J_mol"], color=color, label=f"{label} components")
    axes[1].set(xticks=range(5), xticklabels=labels, yscale="log", ylim=(5e-18, 1e-13),
                title="Exact reduced faces", xlabel="Present component count",
                ylabel="Max relative difference in G, S, Cp, V, VT, VP")
    axes[1].tick_params(axis="x", labelrotation=30)
    axes[1].text(.04, .94, "Zero errors shown at 1e-17", transform=axes[1].transAxes, fontsize=8, va="top")
    axes[2].set(title="Dilution toward each face", xlabel="Each added amount (mol)",
                ylabel="Max present |mu11 - mu_face| (J/mol)")
    axes[2].legend(fontsize=8)
    for ax in axes[1:]:
        ax.grid(alpha=.2)
    fig.suptitle("One regular-solution model; existing faces recovered at 1700 K and 500 bar\n"
                 "Exact zeros use continuous energies. Composition derivatives are evaluated in the positive interior.", fontsize=11)
    fig.savefig(args.output/"dry11_faces.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"native_states": len(rows), "maximum_absolute_errors": maximum,
                      "native_unavailable": len(reference["unavailable_cases"])}, indent=2))


if __name__ == "__main__":
    main()
