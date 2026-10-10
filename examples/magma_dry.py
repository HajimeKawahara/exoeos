"""Restricted dry CMFAS equilibrium with competing silicate minerals.

Requires SciPy and JAX x64; plotting additionally requires matplotlib.
The example retains crystals and admits one liquid, Fo--Fa olivine,
Di--Hd clinopyroxene, pure Mg orthopyroxene, anorthite and silica.
It requires a present liquid with all five components positive. It is an
eager numerical example, not a general or differentiable phase selector.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, minimize, root

from exoeos.magma import (
    CMFAS_COMPONENTS, R, SILICA_POLYMORPHS, anorthite_gibbs,
    clinopyroxene_gibbs, clinopyroxene_standard_gibbs, enstatite_gibbs, liquid_gibbs,
    liquid_standard_gibbs, olivine_gibbs, silica_gibbs,
)


PHASES = ("olivine", "clinopyroxene", "anorthite", "orthopyroxene", "silica")
# Columns: Fo, Fa, Di, Hd, An, En (MgSiO3), SiO2; rows: liquid coordinates.
SOLID_COMPONENTS = np.array([
    [0., 0., .5, .5, 1., .5, 1.],
    [1., 0., .5, 0., 0., .5, 0.],
    [0., 1., 0., .5, 0., 0., 0.],
    [0., 0., 1., 1., 1., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0.],
])
DEFAULT_OXIDES = {"SiO2": 1., "MgO": .55, "FeO": .15, "CaO": .25, "Al2O3": .12}
# MELTS oxide masses, g/mol, propagated through the explicit component map.
LIQUID_MASSES = np.array([60.0843, 60.0843+2*40.3044, 60.0843+2*71.8464,
                          60.0843+56.0794, 101.96128])


def mass_fractions(state):
    """Return [liquid, olivine, cpx, anorthite, opx, silica] mass fractions."""
    solid_mass = np.asarray(state["solid_mol"]) * (LIQUID_MASSES @ SOLID_COMPONENTS)
    masses = np.r_[LIQUID_MASSES @ state["liquid_mol"], solid_mass[:2].sum(),
                    solid_mass[2:4].sum(), solid_mass[4:]]
    return masses / masses.sum()


def component_amounts(oxides):
    """Convert positive oxide mol to CMFAS mol on its nonnegative basis."""
    if set(oxides) != set(DEFAULT_OXIDES):
        raise ValueError("Supply SiO2, MgO, FeO, CaO and Al2O3 in mol.")
    s, m, f, c, a = [float(oxides[key]) for key in DEFAULT_OXIDES]
    if not np.all(np.isfinite([s, m, f, c, a])) or min(s, m, f, c, a) <= 0:
        raise ValueError("All oxide amounts must be finite and positive.")
    if s <= (m + f) / 2 + c:
        raise ValueError("Require SiO2 > (MgO + FeO)/2 + CaO for a positive liquid basis.")
    return np.array([s - (m + f) / 2 - c, m / 2, f / 2, c, a])


def solid_amounts(z):
    """Expand [N_ol, N_cpx, N_an, N_en, N_si, y_Fa, y_Hd] to mol."""
    return jnp.array([z[0]*(1-z[5]), z[0]*z[5], z[1]*(1-z[6]), z[1]*z[6],
                      z[2], z[3], z[4]])


@jax.jit
def liquid_amounts(z, bulk):
    return bulk - jnp.asarray(SOLID_COMPONENTS) @ solid_amounts(z)


@jax.jit
def _energy(z, T, P, bulk):
    # Molar solution energies keep the amount derivative defined at phase absence.
    gl = liquid_gibbs(T, P, liquid_amounts(z, bulk))
    return (gl + z[0] * olivine_gibbs(T, P, jnp.array([1-z[5], z[5]]))
            + z[1] * clinopyroxene_gibbs(T, P, jnp.array([1-z[6], z[6]]))
            + z[2] * anorthite_gibbs(T, P) + z[3] * enstatite_gibbs(T, P)
            + z[4] * jnp.min(silica_gibbs(T, P)))


@jax.jit
def _objective(z, T, P, bulk):
    baseline = bulk @ liquid_standard_gibbs(T, P, include_fe=True, include_ca_al=True)
    return (_energy(z, T, P, bulk) - baseline) / (R*T)


_gradient = jax.jit(jax.grad(_objective))
_liquid_jacobian = jax.jit(jax.jacfwd(liquid_amounts))
_liquid_mu = jax.jit(jax.grad(liquid_gibbs, 2))
_olivine_mu = jax.jit(jax.grad(olivine_gibbs, 2))
_cpx_mu = jax.jit(jax.grad(clinopyroxene_gibbs, 2))
_olivine_g = jax.jit(olivine_gibbs)
_cpx_g = jax.jit(clinopyroxene_gibbs)


def insertion_gaps(T, P, liquid):
    """Minimum solid tangent gaps and incipient Fe fractions (J/mol)."""
    mu = np.asarray(_liquid_mu(T, P, liquid))
    gaps, compositions = [], []
    for fun, derivative, indices in [(_olivine_g, _olivine_mu, [0, 1]),
                                     (_cpx_g, _cpx_mu, [2, 3])]:
        potentials = SOLID_COMPONENTS[:, indices].T @ mu

        def slope(y):
            delta = np.asarray(derivative(T, P, np.array([1-y, y]))) - potentials
            return delta[1] - delta[0]

        y = brentq(slope, 1e-14, 1-1e-14, xtol=1e-14)
        x = np.array([1-y, y])
        gaps.append(float(fun(T, P, x) - x @ potentials))
        compositions.append(y)
    pure = np.array([anorthite_gibbs(T, P), enstatite_gibbs(T, P),
                     jnp.min(silica_gibbs(T, P))])
    return np.r_[gaps, pure - SOLID_COMPONENTS[:, 4:].T @ mu], compositions


def equilibrium(T, P=1e5, oxides=None):
    """Numerical restricted melt-bearing equilibrium, with tangent checks.

    Require 1200 <= T/K <= 2200 and 1 <= P/bar <= 5000. All five
    liquid components must remain resolved. Returned solid amounts are
    [Fo, Fa, Di, Hd, An, En, SiO2] mol; En uses MgSiO3, not Mg2Si2O6.
    A failed solve or unresolved boundary raises rather than reporting
    an equilibrium. Tangent checks cover all admitted solid compositions;
    multiple liquids and omitted mineral components are not tested.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 for phase equilibrium.")
    if not np.all(np.isfinite([T, P])) or not 1200 <= T <= 2200 or not 1e5 <= P <= 5e8:
        raise ValueError("Require 1200 <= T/K <= 2200 and 1 <= P/bar <= 5000.")
    oxides = DEFAULT_OXIDES if oxides is None else oxides
    original = component_amounts(oxides)
    scale = original.sum()
    bulk = original / scale
    constraints = {"type": "ineq", "fun": lambda z: np.asarray(liquid_amounts(z, bulk)) - 1e-12,
                   "jac": lambda z: np.asarray(_liquid_jacobian(z, bulk))}
    candidates = []
    for composition in (insertion_gaps(T, P, bulk)[1], [.7, .7]):
        start = np.r_[np.zeros(5), composition]
        for _ in range(6):
            result = minimize(_objective, start, args=(T, P, bulk), jac=_gradient,
                              method="SLSQP", bounds=[(0., 2.)]*5 + [(1e-10, 1-1e-10)]*2,
                              constraints=constraints, options={"ftol": 2e-13, "maxiter": 300})
            liquid = np.asarray(liquid_amounts(result.x, bulk))
            if not result.success or min(liquid) <= 1e-10:
                break
            gaps, incipient = insertion_gaps(T, P, liquid)
            missing = (result.x[:2] < 1e-8) & (gaps[:2] < -2e-4)
            if not np.any(missing):
                break
            # An absent solution's composition has no amount-weighted gradient.
            # Search its tangent plane explicitly before accepting phase absence.
            start = result.x.copy()
            start[:2][missing] = 0.
            start[5:][missing] = np.asarray(incipient)[missing]
        if result.success and np.all(constraints["fun"](result.x) >= -1e-10):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("Restricted dry equilibrium did not converge.")
    z = min(candidates, key=lambda row: row.fun).x.copy()
    if np.min(liquid_amounts(z, bulk)) <= 1e-10:
        raise RuntimeError("Unresolved liquid disappearance; this example requires a positive liquid interior.")
    active = z[:5] > 1e-8
    z[:5][~active] = 0.
    indices = np.r_[np.flatnonzero(active), [5] if active[0] else [],
                     [6] if active[1] else []].astype(int)
    # Remove inactive phases exactly and polish coexistence chemical potentials.
    if len(indices):
        def residual(values):
            trial = z.copy()
            trial[indices] = values
            return np.asarray(_gradient(trial, T, P, bulk))[indices]

        polished = root(residual, z[indices], tol=1e-10)
        z[indices] = polished.x
        if np.max(np.abs(residual(polished.x))) > 1e-8:
            raise RuntimeError("Unresolved coexistence chemical potentials.")
    liquid = np.asarray(liquid_amounts(z, bulk))
    if min(liquid) <= 1e-10 or min(z[:5]) < 0 or np.any((z[5:] <= 0) | (z[5:] >= 1)):
        raise RuntimeError("Unresolved boundary; this example requires a positive liquid interior.")
    gaps, incipient = insertion_gaps(T, P, liquid)
    if min(gaps) < -2e-4 or (np.any(active) and max(abs(gaps[active])) > 2e-4):
        raise RuntimeError(f"Unresolved solid stability: {gaps}")
    solids = np.asarray(solid_amounts(z)) * scale
    np.testing.assert_allclose(liquid*scale + SOLID_COMPONENTS @ solids, original, atol=1e-12)
    return {"T_K": float(T), "P_Pa": float(P), "initial_oxide_moles": dict(oxides),
            "liquid_mol": (liquid*scale).tolist(), "solid_mol": solids.tolist(),
            "phase_mol": (z[:5]*scale).tolist(), "g_J": float(_energy(z, T, P, bulk)*scale),
            "solid_insertion_J_mol": gaps.tolist(), "incipient_Fe_fractions": incipient,
            "silica_polymorph": SILICA_POLYMORPHS[int(jnp.argmin(silica_gibbs(T, P)))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/magma_dry"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    states = [equilibrium(float(T)) for T in np.linspace(1800., 1400., 41)]
    (args.output / "dry_cooling.json").write_text(json.dumps(
        {"liquid_order": CMFAS_COMPONENTS, "phase_order": PHASES, "states": states},
        indent=2, allow_nan=False) + "\n")
    reference = json.loads((Path(__file__).resolve().parents[1]
                            / "tests/reference/dry_magma_melts_v1.json").read_text())
    reference_states = reference["restricted_equilibria"]
    compared = [equilibrium(row["T_K"], row["P_Pa"], row["initial_oxide_moles"])
                for row in reference_states]
    errors = {"equilibrium_amount_mol": max(float(np.max(np.abs(
        np.r_[a["liquid_mol"], a["solid_mol"]]-np.r_[b["liquid_mol"], b["solid_mol"]])))
        for a, b in zip(compared, reference_states)),
        "equilibrium_G_J": max(abs(a["g_J"]-b["g_J"]) for a, b in zip(compared, reference_states))}
    standards = jax.jit(lambda t, p: liquid_standard_gibbs(t, p, include_fe=True, include_ca_al=True))
    liquid_energy = jax.jit(liquid_gibbs)
    for row in reference["properties"]:
        t, p, n = row["T_K"], row["P_Pa"], np.array(row["liquid_mol"])
        comparisons = [("liquid_standard_J_mol", standards(t, p), row["liquid_standard_J_mol"]),
                       ("liquid_G_J", liquid_energy(t, p, n), row["liquid_g_J"]),
                       ("liquid_mu_J_mol", _liquid_mu(t, p, n), row["liquid_mu_J_mol"]),
                       ("anorthite_G_J_mol", anorthite_gibbs(t, p), row["anorthite_g_J_mol"])]
        for cpx in row["clinopyroxene"]:
            if "status" in cpx:
                continue
            n = np.array(cpx["n_mol"])
            comparisons.extend([
                ("cpx_standard_J_mol", clinopyroxene_standard_gibbs(t, p), cpx["standard_J_mol"]),
                ("cpx_G_J", _cpx_g(t, p, n), cpx["g_J"]),
                ("cpx_mu_J_mol", _cpx_mu(t, p, n), cpx["mu_J_mol"])])
        for key, actual, expected in comparisons:
            errors[key] = max(errors.get(key, 0.), float(np.max(np.abs(np.array(actual)-expected))))
    (args.output / "dry_comparison.json").write_text(json.dumps(
        {"maximum_absolute_errors": errors, "property_TP_states": len(reference["properties"]),
         "restricted_equilibria": len(compared), "full_equilibria": len(reference["full_equilibria"]),
         "unavailable_full_equilibria": reference["unavailable_full_equilibria"]}, indent=2) + "\n")
    plot_comparison(states, reference, args.output)
    print(f"Saved {len(states)} dry equilibrium states to {args.output}")
    print(json.dumps(errors, indent=2))


def plot_comparison(states, reference, output):
    """Plot both the matched candidate set and the separate full-MELTS result."""
    import matplotlib.pyplot as plt

    colors = ["#2474a6", "#d28c24", "#288354", "#ad4972", "#6d5ba6", "#777777"]
    labels = ["Liquid", "Olivine", "Clinopyroxene", "Anorthite", "Mg orthopyroxene", "Silica"]
    temperatures = [row["T_K"] for row in states]
    saved = reference["restricted_equilibria"]
    native_t = [row["T_K"] for row in saved]
    fractions, native_fractions = np.array([mass_fractions(s) for s in states]), np.array([mass_fractions(s) for s in saved])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for i in range(4):
        axes[0].plot(temperatures, fractions[:, i], color=colors[i], label=labels[i])
        axes[0].plot(native_t, native_fractions[:, i], "o", ms=4, mfc="white", color=colors[i])
    axes[0].set(title="Competing phases: same restricted model", ylabel="Mass fraction", ylim=(-.02, 1.02))
    axes[0].legend(fontsize=9)

    def fe_fractions(row):
        liquid, solids = np.array(row["liquid_mol"]), np.array(row["solid_mol"])
        return [liquid[2]/(liquid[1]+liquid[2])] + [
            solids[i+1]/solids[i:i+2].sum() if solids[i:i+2].sum() > 0 else np.nan for i in (0, 2)]

    y, native_y = np.array([fe_fractions(s) for s in states]), np.array([fe_fractions(s) for s in saved])
    for i in range(3):
        axes[1].plot(temperatures, y[:, i], color=colors[i], label=labels[i])
        axes[1].plot(native_t, native_y[:, i], "o", ms=4, mfc="white", color=colors[i])
    axes[1].set(title="Fe/Mg partitioning from Gibbs minimization", ylabel="Fe / (Mg + Fe)", ylim=(0, .4))
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set(xlabel="Temperature (K); cooling to the right", xlim=(1800, 1400))
        ax.grid(alpha=.2)
    fig.suptitle("Dry CMFAS at 1 bar — lines: native JAX; circles: independent restricted MELTS", fontsize=12)
    fig.savefig(output / "dry_cooling_comparison.png", dpi=180)
    plt.close(fig)

    full = sorted(reference["full_equilibria"], key=lambda s: -s["T_K"])
    grouped = []
    for row in full:
        mass = np.zeros(6)
        for phase in row["phases"]:
            name = phase["name"]
            index = next((i for i, prefix in enumerate(
                ["liquid", "olivine", "clinopyroxene", "plagioclase", "orthopyroxene"])
                if name.startswith(prefix)), 5)
            mass[index] += phase["mass_g"]
        grouped.append(mass / mass.sum())
    grouped = np.array(grouped)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    axes[0].plot(temperatures, fractions[:, 0], color=colors[0], lw=2, label="Restricted native JAX")
    axes[0].plot([s["T_K"] for s in full], grouped[:, 0], "s--", color="#292929", label="Full MELTS phase selection")
    axes[0].set(xlabel="Temperature (K); cooling to the right", ylabel="Liquid mass fraction",
                title="Omitted solution components change crystallization", xlim=(1800, 1400), ylim=(-.03, 1.03))
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=.2)
    axes[0].text(.04, .05, "Full MELTS: 1800/2000 K runs rejected\nfor native convergence warnings.",
                 transform=axes[0].transAxes, fontsize=8, bbox={"facecolor": "white", "edgecolor": "none"})
    bottom = np.zeros(len(full))
    for i in range(6):
        if not np.any(grouped[:, i]):
            continue
        label = "Plagioclase (An endpoint)" if i == 3 else "Orthopyroxene" if i == 4 else labels[i]
        axes[1].bar(range(len(full)), grouped[:, i], bottom=bottom, color=colors[i], label=label)
        bottom += grouped[:, i]
    axes[1].set(xticks=range(len(full)), xticklabels=[f"{s['T_K']:g}" for s in full],
                xlabel="Temperature (K)", ylabel="Mass fraction", ylim=(0, 1.02),
                title="Full MELTS allows Ca-poor / Al-bearing pyroxenes")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Same dry bulk, different admitted mineral compositions — not a full-MELTS replacement", fontsize=12)
    fig.savefig(output / "dry_full_melts_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
