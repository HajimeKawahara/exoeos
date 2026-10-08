"""Generate the documentation's capability plots from the implemented APIs.

Run from a source checkout with the docs extra installed. Published tables
are fetched and checksum-verified by the package loaders; subsequent runs
reuse the cache. Building the documentation uses the committed PNGs only.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/exoeos-matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exoeos import (
    ChabrierDebrasTableLoader, FixedCompositionDensityProvider,
    IdealEOS, IdealGas, IdealSolution, MaFeSiOLiquid, MaFeSiOHLiquid,
    MarcumSilicateHydrogenTableLoader, SecondVirialEOS,
    TPHelmholtzDensityProvider, additive_volume_mass_density,
    mass_fraction_solute_state, solution_state, state_tp, total_solution_state,
)
from exoeos.constants import MOLAR_GAS_CONSTANT as R
from exoeos.ma_interval import ma_alloy_curvature_lower_bound
from examples import melts_liquid_mixing as liquid
from examples import melts_solid_mixing as solid
from examples.m2_material.hydrogen_reference import chaudhari_buffered_h2_mass_ppm
from examples.m2_material.major_gas_eos import make_major_gas_eos
from examples.m2_material.sossi_water_calibration import calibration_report

jax.config.update("jax_enable_x64", True)
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 11.5,
    "legend.fontsize": 10, "lines.linewidth": 2, "axes.grid": True,
    "grid.alpha": 0.22, "axes.spines.top": False, "axes.spines.right": False,
    "axes.prop_cycle": matplotlib.cycler(color=["#0072B2", "#D55E00", "#009E73", "#CC79A7"]),
})


def sweep(function, values):
    """Evaluate one-state APIs with a single compiled vectorized call."""
    return jax.jit(jax.vmap(function))(jnp.asarray(values))


def finite(*values):
    for value in values:
        if not np.all(np.isfinite(np.asarray(value))):
            raise ValueError("An unmasked plotted value is not finite.")


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)


def panels(title, rows=1):
    fig, axes = plt.subplots(rows, 2, figsize=(9.2, 4.2 * rows), layout="constrained")
    fig.suptitle(title, fontsize=14)
    return fig, np.asarray(axes).ravel()


def virial():
    fig, (a, b) = panels("Second virial: attraction and repulsion | synthetic coefficients, 400 K")
    pressures = np.linspace(1e5, 2e7, 121)
    for coefficient, style in [(-20e-6, "-"), (0., "--"), (20e-6, "-.")]:
        model = IdealEOS() if coefficient == 0 else SecondVirialEOS([[coefficient]])
        state = sweep(lambda p: state_tp(model, 400., p, jnp.ones(1)), pressures)
        finite(state.rho, state.Z, state.lnphi)
        close(state.Z, 1 + state.rho * coefficient)
        close(state.rho * R * 400 * state.Z, pressures)
        label = f"B = {coefficient * 1e6:g} cm³/mol"
        a.plot(pressures / 1e5, state.Z, style, label=label)
        b.plot(pressures / 1e5, np.asarray(state.lnphi)[:, 0], style, label=label)
    for ax in (a, b):
        ax.set_xlabel("Pressure [bar]")
        ax.legend()
    a.set_ylabel("Compressibility Z")
    b.set_ylabel(r"Log fugacity coefficient $\ln\phi$")
    return fig, {"T_K": 400, "P_Pa": [1e5, 2e7], "points": 121,
                 "B_m3_mol": [-20e-6, 0, 20e-6],
                 "checks": ["Z = 1 + rho B", "TP pressure round trip", "finite curves"]}


def ideal_caloric():
    fig, (a, b) = panels("Calorically perfect H₂/He control | declared constant heat capacities")
    model = IdealGas([.002016, .004003], [3.5 * R, 2.5 * R], reference_temperature=300.)
    temperatures = np.linspace(300., 2000., 101)
    for helium, style in [(0., "-"), (.5, "--"), (1., "-.")]:
        x = jnp.array([1 - helium, helium])
        state = model.state(temperatures, 1e5, x)
        finite(state.h, state.s, state.cp, state.cv)
        close(state.h, state.cp * (temperatures - 300.))
        close(state.cp - state.cv, np.full(101, R))
        a.plot(temperatures, state.h / 1000, style, label=f"He mole fraction = {helium:g}")
    helium = np.linspace(0., 1., 101)
    state = model.state(300., 1e5, np.column_stack((1 - helium, helium)))
    finite(state.s)
    close(np.asarray(state.s)[[0, -1]], [0., 0.])
    close(state.s[50], R * np.log(2))
    b.plot(helium, state.s)
    b.scatter([.5], [R * np.log(2)], color="#D55E00", zorder=3, label=r"$R\ln 2$")
    a.set(xlabel="Temperature [K]", ylabel="Molar enthalpy from 300 K [kJ/mol]")
    b.set(xlabel="He mole fraction", ylabel="Mixing entropy [J/(mol K)]", title="300 K, 1 bar; zero component references")
    a.legend(); b.legend()
    return fig, {"T_K": [300, 2000], "P_Pa": 1e5, "cp_over_R": [3.5, 2.5],
                 "reference_T_K": 300, "reference_h_s": 0,
                 "checks": ["h = cp (T - Tref)", "cp - cv = R", "endpoint and equimolar mixing entropy"]}


def alloy_activities():
    fig, axes = panels("Ma alloy solutions | 1873 K, 1 bar; symmetric endmember standards", rows=2)
    silicon = np.linspace(.001, .12, 101)
    x = np.column_stack((.98 - silicon, silicon, np.full(101, .02)))
    model = MaFeSiOLiquid()
    state = sweep(lambda v: solution_state(model, 1873., 1e5, v), x)
    finite(state.gex_RT, state.lngamma)
    close(np.sum(x * state.lngamma, axis=1), state.gex_RT)
    axes[0].plot(silicon, state.gex_RT)
    axes[0].set(ylabel=r"Excess $g^{E}/(RT)$", title="Dry Fe–Si–O; O atomic fraction = 0.02")
    for i, (name, style) in enumerate(zip(["Fe", "Si", "O"], ["-", "--", "-."])):
        axes[1].plot(silicon, state.lngamma[:, i], style, label=name)
    axes[1].set(ylabel=r"$\ln\gamma_i$", title="Activities share the same excess scalar")
    for ax in axes[:2]:
        ax.set_xlabel("Si atomic fraction")
    hydrogen = np.linspace(0., .15, 101)
    dry = np.array([.90, .08, .02])
    wet_x = np.column_stack(((1 - hydrogen[:, None]) * dry, hydrogen))
    wet = sweep(lambda v: solution_state(MaFeSiOHLiquid(), 1873., 1e5, v), wet_x)
    dry_state = solution_state(model, 1873., 1e5, dry)
    finite(wet.gex_RT, wet.lngamma)
    close(wet.gex_RT, (1 - hydrogen) * dry_state.gex_RT)
    close(wet.lngamma[:, 3], np.zeros(101))
    axes[2].plot(hydrogen, wet.gex_RT)
    axes[2].set(ylabel=r"Excess $g^{E}/(RT)$", title="Fixed dry Fe:Si:O = 0.90:0.08:0.02")
    axes[3].plot(hydrogen, wet.lngamma[:, 3], "--", label=r"$\ln\gamma_H = 0$")
    axes[3].plot(hydrogen[1:], np.log(hydrogen[1:]), label=r"$\ln a_H = \ln x_H$")
    axes[3].set(ylabel="Log activity / activity coefficient", title="H is an ideal-dilution control")
    for ax in axes[2:]:
        ax.set_xlabel("H atomic fraction")
    axes[1].legend(); axes[3].legend()
    return fig, {"T_K": 1873, "P_Pa": 1e5, "x_O_dry": .02,
                 "x_Si": [.001, .12], "wet_dry_ratio": dry.tolist(), "x_H": [0, .15],
                 "checks": ["excess Euler identity", "dry energy dilution", "zero H excess potential"]}


def solution_gibbs():
    fig, (a, b) = panels("Full solution Gibbs | Ma Fe–Si–O, 1873 K; synthetic standards")
    silicon = np.linspace(.001, .12, 101)
    amounts = jnp.asarray(np.column_stack((.98 - silicon, silicon, np.full(101, .02))))
    model, standards = MaFeSiOLiquid(), jnp.array([0., 2., -1.])
    states = sweep(lambda n: total_solution_state(model, 1873., 1e5, n, standards), amounts)
    ideal = sweep(lambda n: total_solution_state(IdealSolution(), 1873., 1e5, n, jnp.zeros(3)), amounts)
    excess = sweep(lambda x: solution_state(model, 1873., 1e5, x).gex_RT, amounts)
    standard = amounts @ standards
    finite(states.gibbs_RT, states.mu_RT, ideal.gibbs_RT, excess)
    close(states.gibbs_RT, standard + ideal.gibbs_RT + excess)
    close(states.gibbs_RT, jnp.sum(amounts * states.mu_RT, axis=1))
    for values, label, style in [(states.gibbs_RT, "Total", "-"), (standard, "Standard", ":"),
                                  (ideal.gibbs_RT, "Ideal mixing", "--"), (excess, "Excess", "-.")]:
        a.plot(silicon, values, style, label=label)
    for i, (name, style) in enumerate(zip(["Fe", "Si", "O"], ["-", "--", "-."])):
        b.plot(silicon, states.mu_RT[:, i], style, label=name)
    a.set(ylabel=r"Whole-phase $G/(RT)$ [mol]", title="Total amount = 1 mol; O fraction = 0.02")
    b.set(ylabel=r"Chemical potential $\mu_i/(RT)$", title=r"Supplied $\mu^0/(RT)=(0,2,-1)$")
    for ax in (a, b):
        ax.set_xlabel("Si atomic fraction"); ax.legend()
    return fig, {"T_K": 1873, "P_Pa": 1e5, "total_mol": 1, "mu0_RT": [0, 2, -1],
                 "x_O": .02, "x_Si": [.001, .12],
                 "checks": ["standard + ideal + excess decomposition", "full Gibbs Euler identity"]}


def mass_solute():
    fig, (a, b) = panels("Mass-fraction solute | synthetic host and solute standard")
    n, masses, solute_mass = jnp.array([.8, .2]), jnp.array([.06, .10]), .002
    host = total_solution_state(IdealSolution(), 1800., 1e5, n, jnp.zeros(2))
    w = np.geomspace(1e-6, .1, 121)
    h = (float(n @ masses) / solute_mass) * w / (1 - w)
    states = sweep(lambda value: mass_fraction_solute_state(host.gibbs_RT, host.mu_RT, n, masses, value, solute_mass, 3.), h)
    finite(states.gibbs_RT, states.host_mu_RT, states.solute_mu_RT)
    correction = states.host_mu_RT - host.mu_RT
    close(states.solute_mu_RT, 3 + np.log(w))
    close(correction, np.log1p(-w[:, None]) * np.asarray(masses) / solute_mass)
    close(states.gibbs_RT, states.host_mu_RT @ n + h * states.solute_mu_RT)
    a.semilogx(w * 1e6, states.solute_mu_RT, label=r"$\mu_s/(RT)=3+\ln w_s$")
    for i, style in enumerate(["-", "--"]):
        b.semilogx(w * 1e6, correction[:, i], style, label=f"Host {i + 1}: {masses[i] * 1000:g} g/mol")
    a.set_ylabel(r"Solute $\mu_s/(RT)$")
    b.set_ylabel(r"Host correction $\Delta\mu_i/(RT)$")
    for ax in (a, b):
        ax.set_xlabel("Solute mass / complete-liquid mass [ppm]"); ax.legend()
    return fig, {"host_amounts_mol": [.8, .2], "host_masses_kg_mol": [.06, .10],
                 "solute_mass_kg_mol": .002, "solute_standard_RT": 3, "mass_fraction": [1e-6, .1],
                 "checks": ["log mass-fraction solute law", "reciprocal host correction", "extensive Euler identity"]}


def density():
    fig, (a, b) = panels("Density helpers | mass conversion and a declared volume mixing rule")
    helium = np.linspace(0, 1, 101)
    x = np.column_stack((1 - helium, helium))
    masses = jnp.array([.002016, .004003])
    provider = TPHelmholtzDensityProvider(IdealEOS(), masses)
    rho = sweep(lambda v: provider.mass_density_tp(1000., 1e6, v), x)
    close(rho, 1e6 / (R * 1000) * (x @ masses))
    a.plot(helium, rho)
    a.set(xlabel="He mole fraction", ylabel="Mass density [kg/m³]", title="Ideal H₂/He: 1000 K, 10 bar")
    w = np.linspace(0, 1, 101)
    densities = jnp.array([1000., 4000.])
    mixed = sweep(lambda value: additive_volume_mass_density(jnp.array([1 - value, value]), densities), w)
    finite(rho, mixed)
    close(np.asarray(mixed)[[0, -1]], densities)
    close(1 / mixed, (1 - w) / 1000 + w / 4000)
    b.plot(w, mixed, label="Additive volume")
    b.plot(w, 1000 * (1 - w) + 4000 * w, "--", label="Arithmetic density average")
    b.set(xlabel="Mass fraction of component 2", ylabel="Mass density [kg/m³]", title="Synthetic densities: 1000 and 4000 kg/m³")
    b.legend()
    return fig, {"ideal_T_K": 1000, "ideal_P_Pa": 1e6, "masses_kg_mol": masses.tolist(),
                 "synthetic_densities_kg_m3": densities.tolist(),
                 "checks": ["ideal mass conversion", "pure endpoints", "additive specific volume"]}


def hhe_table(cache):
    loader = ChabrierDebrasTableLoader(cache_directory=None if cache is None else cache / "DirEOS2021")
    model = loader.load()
    fig, (a, b) = panels("Chabrier–Debras H/He table | fixed Y = 0.275")
    pressures = np.geomspace(1e5, 1e12, 141)
    for temperature, style in [(1000., "-"), (5000., "--"), (20000., "-.")]:
        state = sweep(lambda p: model.state_tp(temperature, p), pressures)
        finite(state.rho, state.s)
        if np.any(np.asarray(state.rho) <= 0):
            raise ValueError("Table density must be positive.")
        a.loglog(pressures / 1e9, state.rho / 1000, style, label=f"{temperature:g} K")
        b.semilogx(pressures / 1e9, state.s / 1000, style, label=f"{temperature:g} K")
    masses = jnp.array([.001008, .004003])
    fractions = jnp.array([.725, .275])
    x = fractions / masses; x = x / x.sum()
    provider = FixedCompositionDensityProvider(model, masses, fractions, composition_rtol=1e-12)
    close(provider.mass_density_tp(5000., 1e9, x), model.state_tp(5000., 1e9).rho)
    a.set_ylabel("Mass density [g/cm³]")
    b.set_ylabel("Specific entropy [kJ/(kg K)]")
    for ax in (a, b):
        ax.set_xlabel("Pressure [GPa]"); ax.legend()
    return fig, {"variant": "Y0275", "T_K": [1000, 5000, 20000], "P_Pa": [1e5, 1e12],
                 "archive_url": loader.archive_url, "archive_sha256": loader.checksum, "tables_sha256": loader.checksums,
                 "checks": ["verified published files", "finite curves and positive density", "fixed-composition density adapter"]}


def silicate_table(cache):
    loader = MarcumSilicateHydrogenTableLoader(cache_directory=None if cache is None else cache / "MgSiO3-H-EOS")
    model = loader.load()
    fig, (a, b) = panels("Marcum silicate–H table | composition and missing low-pressure cells")
    pressures = np.geomspace(5e9, 2e11, 101)
    for fraction, style in [(2.5e-5, "-"), (.5, "--"), (1., "-.")]:
        state = sweep(lambda p: model.state_tp(5000., p, jnp.array([1 - fraction, fraction])), pressures)
        finite(state.rho)
        a.semilogx(pressures / 1e9, state.rho / 1000, style, label=f"Hydrous endmember x = {fraction:g}")
    temperatures = np.linspace(3000., 10000., 141)
    for pressure, style in [(2e9, "-"), (20e9, "--")]:
        state = sweep(lambda t: model.state_tp(t, pressure, jnp.array([.5, .5])), temperatures)
        valid = temperatures <= 6000 if pressure == 2e9 else np.ones(141, dtype=bool)
        finite(np.asarray(state.cp)[valid])
        if not np.all(np.isnan(np.asarray(state)[:, ~valid])):
            raise ValueError("Missing table cells must remain all-NaN states.")
        b.plot(temperatures, state.cp / 1000, style, label=f"{pressure / 1e9:g} GPa")
    b.axvspan(6000., 10000., color="0.9", zorder=0)
    b.text(.98, .95, "2 GPa: no table cells\nabove 6000 K", transform=b.transAxes, ha="right", va="top", fontsize=9)
    a.set(xlabel="Pressure [GPa]", ylabel="Mass density [g/cm³]", title="5000 K; MgSiO₃ / MgSiO₃H₄ basis")
    b.set(xlabel="Temperature [K]", ylabel="Specific heat capacity [kJ/(kg K)]", title="Hydrous endmember mole fraction = 0.5")
    a.legend(); b.legend(loc="lower right")
    return fig, {"density_T_K": 5000, "density_P_Pa": [5e9, 2e11], "x_hydrous": [2.5e-5, .5, 1],
                 "heat_capacity_T_K": [3000, 10000], "heat_capacity_P_Pa": [2e9, 20e9],
                 "table_url": loader.table_url, "table_sha256": loader.checksum,
                 "checks": ["verified published file", "finite supported curves", "all-NaN states in missing cells"]}


def alloy_bounds():
    fig, ax = plt.subplots(figsize=(8, 4.3), layout="constrained")
    model = MaFeSiOHLiquid()
    def energy(solutes):
        x = jnp.concatenate((jnp.atleast_1d(1 - solutes.sum()), solutes))
        return jnp.sum(x * jnp.log(x)) + model.gex_RT(2173.15, 1e5, x)
    hessian = jax.jit(jax.hessian(energy))
    scales = np.linspace(.5, 3., 41)
    bounds, samples = [], []
    for scale in scales:
        upper = np.array([.08, .02, .04]) * scale
        bound = ma_alloy_curvature_lower_bound(model, 2173.15, [1 - upper.sum(), 0., 0., 0.], [1., *upper])
        sample = np.linalg.eigvalsh(np.asarray(hessian(jnp.asarray(upper / 2)))).min()
        if sample < bound:
            raise ValueError("A sampled Hessian violates the declared domain bound.")
        bounds.append(bound); samples.append(sample)
    finite(bounds, samples)
    ax.plot(scales, samples, "--", label="Local Hessian: box midpoint")
    ax.plot(scales, bounds, label="Interval lower bound: entire box")
    ax.axhline(0, color="0.4", lw=1)
    ax.set(xlabel="Scale s of the solute box", ylabel="Smallest curvature / lower bound",
           title="Ma Fe–Si–O–H curvature | 2173.15 K\nSi ≤ 0.08s, O ≤ 0.02s, H ≤ 0.04s; Fe = remainder")
    ax.legend()
    return fig, {"T_K": 2173.15, "scale": [.5, 3.], "solute_upper_at_unit_scale": [.08, .02, .04],
                 "checks": ["finite bounds", "midpoint eigenvalues above whole-box bounds"],
                 "scope": "A negative lower bound is inconclusive; midpoint sampling is not a proof."}


def melts_mixing():
    fig, (a, b) = panels("MELTS mixing expressions | specified compositions, without phase selection")
    water = np.linspace(0., .15, 101)
    liquid_parameters = liquid.liquid_mixing_parameters(1873., 1e5)
    values = []
    for fraction in water:
        amounts = np.zeros(len(liquid.COMPONENTS))
        for component, amount in [("sio2", .6 * (1 - fraction)), ("mg2sio4", .4 * (1 - fraction)), ("h2o", fraction)]:
            amounts[liquid.COMPONENTS.index(component)] = amount
        state = liquid.liquid_mixing_state(liquid_parameters, amounts)
        present = amounts > 0
        close(state["mixing_gibbs_rt"], amounts[present] @ np.asarray(state["mixing_mu_rt"], dtype=float)[present])
        values.append(state["mixing_gibbs_rt"])
    finite(values)
    a.plot(water, values)
    a.set(xlabel="H₂O component mole fraction", ylabel=r"Mixing $G/(RT)$ [mol]", title="Liquid: 1873 K, 1 bar, 1 mol\nDry SiO₂:Mg₂SiO₄ = 0.6:0.4")
    iron = np.linspace(0, 1, 101)
    for temperature, style in [(1200., "-"), (1800., "--"), (2400., "-.")]:
        parameters = solid.solid_mixing_parameters("olivine", temperature, 1e5)
        curve = [solid.solid_mixing_state(parameters, [x, x, 0.])["mixing_gibbs_rt"] for x in iron]
        finite(curve)
        close([curve[0], curve[-1]], [0, 0])
        b.plot(iron, curve, style, label=f"{temperature:g} K")
    b.set(xlabel="Fayalite endmember mole fraction", ylabel=r"Molar mixing $g/(RT)$", title="Olivine: equal Fe occupations, no Ca\nFixed site path; ordering is not minimized")
    b.legend()
    return fig, {"liquid_T_K": 1873, "P_Pa": 1e5, "liquid_dry_ratio": {"sio2": .6, "mg2sio4": .4},
                 "water_mole_fraction": [0, .15], "olivine_T_K": [1200, 1800, 2400], "olivine_coordinates": "[x, x, 0]",
                 "checks": ["liquid Euler identity", "pure olivine mixing endpoints", "finite curves"],
                 "native_runtime_used": False}


def major_gas():
    fig, (a, b) = panels("M2 major-gas potential | conditional coefficients, 2173.15 K")
    species = ("H2", "He1", "H2O1", "O2", "Mg1")
    x = jnp.array([.75, .17, .06, .012, .008])
    pressures = np.linspace(1e5, 3e7, 121)
    for cross in [0., 10., 20.]:
        for policy, style in [("extrapolate", "-"), ("hold_2000", "--")]:
            model = make_major_gas_eos(species, {"h2_he_cm3_mol": cross,
                "water_cross_temperature_policy": policy, "trace_pair_policy": "zero"})
            # Host validation precedes the traced pressure sweep.
            model.parameters(2173.15, float(pressures[-1]))
            state = sweep(lambda p: model.state(2173.15, p, x), pressures)
            finite(state.Z, state.lnphi)
            a.plot(pressures / 1e5, state.Z, style, color=plt.rcParams["axes.prop_cycle"].by_key()["color"][[0., 10., 20.].index(cross)],
                   label=f"B(H₂,He)={cross:g}; {policy}")
            if cross == 10. and policy == "extrapolate":
                close(state.lnphi[:, 3], -jnp.log(state.Z))
                close(state.lnphi[:, 4], state.lnphi[:, 3])
                for i, label, line in [(0, "H₂", "-"), (1, "He", "--"), (2, "H₂O", "-."), (3, "O₂ and Mg (zero pair rows)", ":")]:
                    b.plot(pressures / 1e5, state.lnphi[:, i], line, label=label)
    a.set(ylabel="Compressibility Z", title="Six declared alternatives; B in cm³/mol")
    b.set(ylabel=r"$\ln\phi_i$", title="B(H₂,He) = 10; water cross extrapolated")
    for ax in (a, b):
        ax.set_xlabel("Pressure [bar]"); ax.legend(fontsize=9)
    return fig, {"T_K": 2173.15, "P_Pa": [1e5, 3e7], "species": list(species), "x": x.tolist(),
                 "h2_he_cm3_mol": [0, 10, 20], "water_policies": ["extrapolate", "hold_2000"],
                 "checks": ["host parameter validation", "finite curves", "zero-pair trace lnphi = -ln Z"],
                 "scope": "Constitutive alternatives, not empirical uncertainty bands."}


def material_references():
    fig, (a, b) = panels("Material references | measured hosts and recorded observation bases")
    data = json.loads((ROOT / "examples/m2_material/hydrogen_reference.json").read_text())["chaudhari_2025"]
    for host, style, marker in [("basalt", "-", "o"), ("andesite", "--", "s")]:
        rows = [r for r in data["observations"] if r["host"] == host and not r["excluded_from_further_analysis_by_authors"]]
        pressures = np.linspace(min(r["pressure_Pa"] for r in rows), max(r["pressure_Pa"] for r in rows), 81)
        values = [chaudhari_buffered_h2_mass_ppm(host, 1673.15, p, buffer="Fe-FeO-H2O")["molecular_h2_mass_ppm"] for p in pressures]
        line, = a.plot(pressures / 1e9, values, style, label=f"{host}: published regression")
        a.errorbar([r["pressure_Pa"] / 1e9 for r in rows], [r["H2_mass_ppm"] for r in rows],
                   yerr=[r["H2_mass_ppm_one_sigma"] for r in rows], fmt=marker, color=line.get_color(), capsize=3)
        finite(values)
    report = calibration_report()
    maximum = 0.
    for name, marker in [("basalt_epsilon_6.3", "o"), ("peridotite_epsilon_5.1", "s")]:
        rows = [r for r in report["branches"][name]["all_observation_residuals"] if r["included_in_fit"]]
        measured = np.array([r["observed_corrected_water_mass_ppm"] for r in rows])
        predicted = np.array([r["predicted_water_mass_ppm"] for r in rows])
        finite(measured, predicted)
        close(np.sqrt(np.mean((predicted - measured)**2)), report["branches"][name]["fit_rmse_ppm"])
        maximum = max(maximum, measured.max(), predicted.max())
        b.scatter(measured, predicted, marker=marker, label=f"IR coefficient {name.split('_')[-1]}")
    b.plot([0, maximum * 1.05], [0, maximum * 1.05], color="0.4", linestyle="--", label="1:1")
    a.set(xlabel="Total buffered pressure\n[GPa]", ylabel="Molecular H₂ mass / glass mass [ppm]",
          title="Chaudhari: 1673.15 K, Fe–FeO–H₂O buffer")
    b.set(xlabel="Observed H₂O-equivalent mass\n[ppm]", ylabel="Reconstructed H₂O-equivalent mass [ppm]",
          title="Sossi: same samples, two IR alternatives")
    a.legend(fontsize=9); b.legend()
    return fig, {"hydrogen_hosts": ["basalt", "andesite"], "hydrogen_T_K": 1673.15,
                 "water_fit_samples": 11, "water_excluded": report["fit_excluded_samples"],
                 "checks": ["source-restricted H2 API", "Sossi input checksum", "parity residuals reproduce fit RMSE"],
                 "scope": "H2 error bars are reported observation SDs; water branches are not independent observations or external validation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "documents/_static/feature_plots")
    parser.add_argument("--table-cache", type=Path, help="Parent of DirEOS2021 and MgSiO3-H-EOS caches.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = {}
    functions = [virial, ideal_caloric, alloy_activities, solution_gibbs, mass_solute, density,
                 hhe_table, silicate_table, alloy_bounds, melts_mixing, major_gas, material_references]
    for function in functions:
        print(f"Rendering {function.__name__}", flush=True)
        fig, metadata = function(args.table_cache) if function in (hhe_table, silicate_table) else function()
        path = args.output_dir / f"{function.__name__}.png"
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        plots[path.name] = {**metadata, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        jax.clear_caches()
    sources = list((ROOT / "src/exoeos").glob("*.py")) + [Path(__file__).resolve()]
    sources += [ROOT / f"examples/{name}" for name in [
        "melts_liquid_mixing.py", "melts_solid_mixing.py", "m2_liquid_mixing/parameters.json",
        "m2_solid_mixing/parameters.json", "m2_material/major_gas_eos.py", "m2_material/gas_virial.py",
        "m2_material/gas_virial_sources.json", "m2_material/hydrogen_reference.py", "m2_material/hydrogen_reference.json",
        "m2_material/sossi_water_calibration.py", "m2_material/material_admission.py",
        "m2_material/water_data/Sossi2023_Table1.csv", "m2_material/water_data/sossi2023_provenance.json",
    ]]
    manifest = {"source_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "environment": {"python": platform.python_version(), "jax": jax.__version__,
                                "numpy": np.__version__, "matplotlib": matplotlib.__version__, "jax_enable_x64": True},
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
                "plots": plots}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(plots)} plots and their provenance manifest to {args.output_dir}")


if __name__ == "__main__":
    main()
