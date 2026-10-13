"""Closed Fe(II) MgO--FeO--SiO2 cooling with one liquid and one olivine.

The small equilibrium example requires positive Q, Fo and Fa inventories,
T >= 1600 K and 1 <= P/bar <= 5000, where both solution energies are convex.
It retains all crystals and excludes pyroxenes and silica solids. General
phase selection remains an ExoGibbs responsibility. No SciPy/MELTS runtime
is needed; run with JAX_ENABLE_X64=1 and optional matplotlib for figures.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exoeos.magma import liquid_gibbs, olivine_gibbs


liquid_mu = jax.jit(jax.grad(liquid_gibbs, 2))
olivine_mu = jax.jit(jax.grad(olivine_gibbs, 2))


def _bisect(function, lower, upper):
    """Increasing root with known endpoint signs; evaluate interiors only."""
    def step(_, bounds):
        lo, hi = bounds
        mid = (lo + hi) / 2
        negative = function(mid) < 0
        return jnp.where(negative, mid, lo), jnp.where(negative, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 52, step, (lower, upper))
    return (lo + hi) / 2


@jax.jit
def insertion(T, P, bulk):
    """Minimum olivine insertion G (J/mol) and incipient Fa fraction.

    Requires a present liquid with all three amounts positive. Negative
    gap means an olivine of the returned composition can lower total G.
    """
    mu = liquid_mu(T, P, bulk)[1:]

    def slope(y):
        delta = olivine_mu(T, P, jnp.array([1-y, y])) - mu
        return delta[1] - delta[0]

    y = _bisect(slope, 0., 1.)
    n = jnp.array([1-y, y])
    return olivine_gibbs(T, P, n) - n @ mu, y


@jax.jit
def total_gibbs(T, P, solid, bulk):
    """Extensive G; solid=[Fo, Fa] and bulk=[Q, F, A], all in mol."""
    liquid = jnp.concatenate((bulk[:1], bulk[1:] - solid))
    return liquid_gibbs(T, P, liquid) + olivine_gibbs(T, P, solid)


@jax.jit
def _crystals(T, P, bulk):
    # Normalize to one initial mol Si to keep tolerances scale independent.
    a, c = bulk[1], bulk[2]
    gradient = jax.grad(total_gibbs, 2)

    def at_amount(s):
        # At fixed total crystal amount s, exchange Fe for Mg until the
        # two crystallization derivatives agree. Endpoint slopes diverge.
        lo, hi = jnp.maximum(0., s-c), jnp.minimum(a, s)

        def exchange(u):
            d = gradient(T, P, jnp.array([u, s-u]), bulk)
            return d[0] - d[1]

        u = _bisect(exchange, lo, hi)
        solid = jnp.array([u, s-u])
        return solid, gradient(T, P, solid, bulk)[1]

    s = _bisect(lambda value: at_amount(value)[1], 0., a+c)
    return at_amount(s)[0]


def component_amounts(mgo_mol, feo_mol, sio2_mol):
    """Convert oxides to [SiO2, Mg2SiO4, Fe2SiO4] mol for this example."""
    values = np.array([mgo_mol, feo_mol, sio2_mol], dtype=float)
    if (not np.all(np.isfinite(values)) or np.any(values <= 0)
            or mgo_mol + feo_mol >= 2 * sio2_mol):
        raise ValueError("Require MgO > 0, FeO > 0, SiO2 > (MgO+FeO)/2, all finite.")
    return np.array([sio2_mol - (mgo_mol + feo_mol)/2, mgo_mol/2, feo_mol/2])


def equilibrium(T, P=1e5, mgo_mol=1.2, feo_mol=.3, sio2_mol=1.):
    """Return the unique restricted equilibrium and Fe/Mg partitioning.

    ``amounts_mol`` order is [liquid Q, liquid F, liquid A, solid Fo, solid Fa].
    ``liquid_fe_fraction`` and ``olivine_fe_fraction`` are Fe/(Mg+Fe),
    not the ternary liquid x_A. Kd=(Fe/Mg)_ol/(Fe/Mg)_liq. Solid composition
    and Kd are None when olivine is absent; incipient composition is separate.
    This eager phase selector is not a differentiable equilibrium API.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 for phase equilibrium.")
    if not np.all(np.isfinite([T, P])) or T < 1600 or not 1e5 <= P <= 5e8:
        raise ValueError("Require finite T >= 1600 K and 1 <= P/bar <= 5000.")
    bulk = component_amounts(mgo_mol, feo_mol, sio2_mol) / sio2_mol
    gap, incipient_y = insertion(T, P, bulk)
    solid = np.asarray(_crystals(T, P, bulk)) if gap < 0 else np.zeros(2)
    liquid = np.r_[bulk[0], bulk[1:] - solid]
    residual = None
    if solid.sum() > 0:
        residual = np.asarray(olivine_mu(T, P, solid) - liquid_mu(T, P, liquid)[1:])
        if not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > 1e-5:
            raise RuntimeError(f"Unresolved solid/liquid chemical potentials: {residual}")
    return {
        "T_K": float(T), "P_Pa": float(P),
        "initial_oxide_moles": {"MgO": mgo_mol, "FeO": feo_mol, "SiO2": sio2_mol},
        "amounts_mol": (np.r_[liquid, solid] * sio2_mol).tolist(),
        "g_J": float(total_gibbs(T, P, solid, bulk)) * sio2_mol,
        "liquid_fe_fraction": float(liquid[2] / liquid[1:].sum()),
        "olivine_fe_fraction": float(solid[1] / solid.sum()) if solid.sum() > 0 else None,
        "Kd_Fe_Mg": float((solid[1]/solid[0]) / (liquid[2]/liquid[1])) if solid.sum() > 0 else None,
        "initial_insertion_J_mol": float(gap),
        "incipient_olivine_fe_fraction": float(incipient_y),
        "coexistence_residual_J_mol": None if residual is None else residual.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/magma_olivine"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    temperatures = np.linspace(2200., 1600., 121)
    states = [equilibrium(t) for t in temperatures]
    lo, hi = 1900., 2100.
    bulk = component_amounts(1.2, .3, 1.)
    for _ in range(50):
        mid = (lo+hi)/2
        if insertion(mid, 1e5, bulk)[0] < 0:
            lo = mid
        else:
            hi = mid
    onset = (lo+hi)/2
    reference = json.loads((Path(__file__).resolve().parents[1] / "tests/reference/olivine_melts_v1.json").read_text())
    points = [r for r in reference["equilibria"] if r["P_Pa"] == 1e5
              and r["initial_oxide_moles"] == {"MgO": 1.2, "FeO": .3, "SiO2": 1.}]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True, layout="constrained")
    amounts = np.array([s["amounts_mol"] for s in states])
    for label, values, color in [("Liquid", amounts[:, :3].sum(axis=1), "tab:blue"),
                                 ("Olivine", amounts[:, 3:].sum(axis=1), "tab:orange")]:
        axes[0].plot(temperatures, values, label=label, color=color)
        axes[0].plot([p["T_K"] for p in points],
                     [sum(p["amounts_mol"][:3] if label == "Liquid" else p["amounts_mol"][3:]) for p in points],
                     "o", mfc="none", color=color)
    for field, label, color in [("liquid_fe_fraction", "Liquid", "tab:blue"),
                                ("olivine_fe_fraction", "Olivine", "tab:orange")]:
        axes[1].plot(temperatures, [s[field] for s in states], label=label, color=color)
        axes[1].plot([p["T_K"] for p in points], [p[field] for p in points], "o", mfc="none", color=color)
    axes[1].axhline(.2, color=".5", ls=":", label="Bulk Fe/(Mg+Fe)")
    axes[2].plot(temperatures, [s["Kd_Fe_Mg"] for s in states], color="tab:green")
    axes[2].plot([p["T_K"] for p in points], [p["Kd_Fe_Mg"] for p in points], "o", mfc="none", color="tab:green")
    axes[0].set(ylabel="Amount (mol Si)", title="Closed liquid + olivine equilibrium, Fe(II), 1 bar\nLines: native JAX; circles: independent MELTS properties")
    axes[1].set(ylabel="Fe / (Mg + Fe)")
    axes[2].set(ylabel=r"$K_D=(Fe/Mg)_{ol}/(Fe/Mg)_{liq}$", xlabel="Temperature (K); cooling to the right")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.axvline(onset, color=".5", ls="--", lw=.8)
    axes[0].legend()
    axes[1].legend()
    axes[2].invert_xaxis()
    fig.savefig(args.output / "olivine_partitioning.png", dpi=180)
    plt.close(fig)

    # Compare mixed-phase energies at independently prescribed compositions.
    rows = reference["properties"]
    liquid_energy, solid_energy = jax.jit(liquid_gibbs), jax.jit(olivine_gibbs)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True, layout="constrained")
    for P, color in [(1e5, "tab:blue"), (5e7, "tab:orange"), (5e8, "tab:green")]:
        group = [r for r in rows if r["P_Pa"] == P and r["olivine_n_mol"] == [.8, .2]]
        ts = np.array([r["T_K"] for r in group])
        for i, (function, nkey, gkey) in enumerate([
            (liquid_energy, "liquid_n_mol", "liquid_g_J"),
            (solid_energy, "olivine_n_mol", "olivine_g_J"),
        ]):
            error = [float(function(r["T_K"], P, r[nkey])) - r[gkey] for r in group]
            axes[i].plot(ts, error, "o-", color=color, label=f"{P/1e5:g} bar")
            axes[i].axhline(0, color=".6", lw=.6)
            axes[i].grid(alpha=.2)
    axes[0].set(title="Native JAX minus pinned MELTS (same compositions)", ylabel="Liquid G difference (J)")
    axes[1].set(ylabel="Olivine G difference (J)", xlabel="Temperature (K)")
    axes[0].legend()
    fig.savefig(args.output / "olivine_melts_comparison.png", dpi=180)
    plt.close(fig)
    comparison = [equilibrium(r["T_K"], r["P_Pa"], r["initial_oxide_moles"]["MgO"],
                              r["initial_oxide_moles"]["FeO"], r["initial_oxide_moles"]["SiO2"])
                  for r in reference["equilibria"]]
    summary = {
        "scope": "Closed Fe(II) liquid + binary olivine only; no other solids or redox.",
        "onset_T_K": onset,
        "max_liquid_g_difference_J": max(abs(float(liquid_energy(r["T_K"], r["P_Pa"], r["liquid_n_mol"])) - r["liquid_g_J"]) for r in rows),
        "max_olivine_g_difference_J": max(abs(float(solid_energy(r["T_K"], r["P_Pa"], r["olivine_n_mol"])) - r["olivine_g_J"]) for r in rows),
        "max_total_g_difference_J": max(abs(s["g_J"]-r["g_J"]) for s, r in zip(comparison, reference["equilibria"])),
        "max_amount_difference_mol": max(float(np.max(np.abs(np.array(s["amounts_mol"]) - r["amounts_mol"])))
                                         for s, r in zip(comparison, reference["equilibria"])),
        "max_Kd_difference": max(abs(s["Kd_Fe_Mg"] - r["Kd_Fe_Mg"])
                                  for s, r in zip(comparison, reference["equilibria"]) if s["Kd_Fe_Mg"] is not None),
        "states": [equilibrium(t) for t in (2200., 2000., 1900., 1800., 1600.)],
    }
    (args.output / "olivine_partitioning.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
