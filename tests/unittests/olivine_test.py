"""Native Fe(II) ternary liquid, olivine, and independent MELTS equilibria."""

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos.magma import (
    COMPONENTS, OLIVINE_COMPONENTS, R, TERNARY_COMPONENTS, fayalite_gibbs,
    forsterite_gibbs, liquid_gibbs, liquid_standard_gibbs, olivine_gibbs,
)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = json.loads((ROOT / "tests/reference/olivine_melts_v1.json").read_text())
spec = importlib.util.spec_from_file_location("magma_olivine", ROOT / "examples/magma_olivine.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.mark.parametrize("row", REFERENCE["properties"])
def test_mixed_and_endpoint_olivine_against_actual_melts(row):
    T, P, n = row["T_K"], row["P_Pa"], np.array(row["olivine_n_mol"])
    assert olivine_gibbs(T, P, n) == pytest.approx(row["olivine_g_J"], abs=1e-8, rel=0)
    if np.all(n > 0):
        np.testing.assert_allclose(jax.grad(olivine_gibbs, 2)(T, P, n),
                                   row["olivine_mu_J_mol"], atol=1e-8, rtol=0)


@pytest.mark.parametrize("row", [r for r in REFERENCE["properties"] if r["olivine_n_mol"] == [.8, .2]])
def test_ternary_liquid_against_actual_melts(row):
    T, P, n = row["T_K"], row["P_Pa"], row["liquid_n_mol"]
    np.testing.assert_allclose(liquid_standard_gibbs(T, P, include_fe=True),
                               row["liquid_standard_J_mol"], atol=2e-8, rtol=0)
    assert liquid_gibbs(T, P, n) == pytest.approx(row["liquid_g_J"], abs=3e-5, rel=0)
    np.testing.assert_allclose(jax.grad(liquid_gibbs, 2)(T, P, n),
                               row["liquid_mu_J_mol"], atol=1e-6, rtol=0)


@pytest.mark.parametrize("row", REFERENCE["equilibria"])
def test_independent_melts_partitioning_conservation_and_coexistence(row):
    ox = row["initial_oxide_moles"]
    r = example.equilibrium(row["T_K"], row["P_Pa"], ox["MgO"], ox["FeO"], ox["SiO2"])
    n = np.array(r["amounts_mol"])
    assert np.all(n >= 0)
    np.testing.assert_allclose(n, row["amounts_mol"], atol=1e-9, rtol=0)
    assert r["g_J"] == pytest.approx(row["g_J"], abs=3e-5, rel=0)
    # Rows Mg, Fe, Si, O; Fe is conserved as Fe(II), without oxygen exchange.
    formula = np.array([[0, 2, 0, 2, 0], [0, 0, 2, 0, 2],
                        [1, 1, 1, 1, 1], [2, 4, 4, 4, 4]])
    elements = np.array([ox["MgO"], ox["FeO"], ox["SiO2"],
                         ox["MgO"] + ox["FeO"] + 2*ox["SiO2"]])
    np.testing.assert_allclose(formula @ n, elements, rtol=0, atol=5e-16)
    masses = np.array([24.305, 55.845, 28.0855, 15.9994])
    assert (masses @ formula) @ n == pytest.approx(masses @ elements, rel=1e-15)
    if n[3:].sum() == 0:
        assert r["initial_insertion_J_mol"] >= 0
        assert r["olivine_fe_fraction"] is None and r["Kd_Fe_Mg"] is None
        assert row["Kd_Fe_Mg"] is None
    else:
        np.testing.assert_allclose(r["coexistence_residual_J_mol"], 0, atol=2e-8)
        assert r["Kd_Fe_Mg"] == pytest.approx(row["Kd_Fe_Mg"], abs=1e-9, rel=0)
        assert r["olivine_fe_fraction"] < r["liquid_fe_fraction"]


@pytest.mark.parametrize("function,n", [(liquid_gibbs, [.25, .6, .15]), (olivine_gibbs, [.8, .2])])
def test_jit_vmap_and_thermodynamic_derivatives(function, n):
    state = jnp.array([1850., 5e7, *n])
    def energy(z):
        return function(z[0], z[1], z[2:])
    value, derivative = jax.jit(jax.value_and_grad(energy))(state)
    hessian = jax.jit(jax.hessian(energy))(state)
    assert np.all(np.isfinite(hessian))
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-10, rtol=0)
    assert value == pytest.approx(float(state[2:] @ derivative[2:]), abs=1e-8)
    np.testing.assert_allclose(hessian[2:, 2:] @ state[2:], 0, atol=1e-9)
    assert function(state[0], state[1], 13*state[2:]) == pytest.approx(float(13*value), rel=2e-15)
    for i, step in enumerate([.02, 100., *([1e-5]*len(n))]):
        delta = jnp.eye(len(state))[i]*step
        fd = (energy(state+delta) - energy(state-delta))/(2*step)
        assert derivative[i] == pytest.approx(float(fd), abs=1e-6, rel=3e-7)
    batch = jnp.stack((state, state.at[0].set(1950.)))
    np.testing.assert_allclose(jax.jit(jax.vmap(energy))(batch), [energy(s) for s in batch], atol=1e-8)
    # S=-G_T, Cp=-T G_TT, V=G_P must retain the standard-state dependence.
    assert derivative[0] < 0 and derivative[1] > 0 and hessian[0, 0] < 0


def test_binary_compatibility_zero_phase_pure_limits_and_site_multiplicity():
    T, P = 1800., 5e7
    assert COMPONENTS == ("SiO2", "Mg2SiO4")
    assert TERNARY_COMPONENTS == ("SiO2", "Mg2SiO4", "Fe2SiO4")
    assert OLIVINE_COMPONENTS == ("Mg2SiO4", "Fe2SiO4")
    np.testing.assert_array_equal(liquid_standard_gibbs(T, P, include_fe=True)[:2], liquid_standard_gibbs(T, P))
    for n in ([1., 0.], [0., 1.], [.25, .75], [0., 0.]):
        assert liquid_gibbs(T, P, [*n, 0.]) == pytest.approx(float(liquid_gibbs(T, P, n)), abs=1e-8)
    for phase, size in ((liquid_gibbs, 3), (olivine_gibbs, 2)):
        zeros = jnp.zeros(size)
        assert phase(T, P, zeros) == 0.
        assert jax.grad(phase, 0)(T, P, zeros) == 0.
        assert jax.grad(phase, 1)(T, P, zeros) == 0.
    assert olivine_gibbs(T, P, [1., 0.]) == forsterite_gibbs(T, P)
    assert olivine_gibbs(T, P, [0., 1.]) == fayalite_gibbs(T, P)
    assert liquid_gibbs(T, P, [0., 0., 1.]) == liquid_standard_gibbs(T, P, include_fe=True)[2]
    assert liquid_standard_gibbs(1490., 1e5, include_fe=True)[2] == pytest.approx(float(fayalite_gibbs(1490., 1e5)), abs=1e-8)
    y = jnp.array([.8, .2])
    standards = jnp.array([forsterite_gibbs(T, P), fayalite_gibbs(T, P)])
    excess = (20300 + .015*(P/1e5-1))*np.prod(y)
    assert olivine_gibbs(T, P, y)-y@standards-excess == pytest.approx(float(2*R*T*jnp.sum(y*jnp.log(y))), abs=1e-8)
    # Excess pressure derivative: J/bar -> m3, including both sites' entropy.
    v_excess = jax.grad(lambda p: olivine_gibbs(T, p, y) - y @ jnp.array([forsterite_gibbs(T, p), fayalite_gibbs(T, p)]))(P)
    assert v_excess == pytest.approx(float(.015*np.prod(y)/1e5), abs=1e-19)


def test_restricted_minimum_against_feasible_grid_scaling_and_onset():
    T, P = 1900., 1e5
    bulk = example.component_amounts(1.2, .3, 1.)
    result = example.equilibrium(T)
    u, v = np.meshgrid(np.linspace(0, .6, 41), np.linspace(0, .15, 41))
    grid = jnp.array(np.c_[u.ravel(), v.ravel()])
    energies = jax.jit(jax.vmap(example.total_gibbs, in_axes=(None, None, 0, None)))(T, P, grid, bulk)
    assert result["g_J"] <= float(energies.min()) + 1e-8
    for scale in (.001, 13.):
        scaled = example.equilibrium(T, mgo_mol=1.2*scale, feo_mol=.3*scale, sio2_mol=scale)
        np.testing.assert_allclose(scaled["amounts_mol"], np.array(result["amounts_mol"])*scale, atol=1e-13)
        assert scaled["Kd_Fe_Mg"] == pytest.approx(result["Kd_Fe_Mg"], abs=1e-12)
        assert scaled["g_J"] == pytest.approx(scale*result["g_J"], rel=2e-15)
    lo, hi = 1900., 2100.
    for _ in range(40):
        mid = (lo+hi)/2
        if example.insertion(mid, P, bulk)[0] < 0:
            lo = mid
        else:
            hi = mid
    onset = (lo+hi)/2
    assert example.equilibrium(onset+.01)["Kd_Fe_Mg"] is None
    assert sum(example.equilibrium(onset-.01)["amounts_mol"][3:]) > 0


def test_validation_and_static_shape_contracts():
    for function, n in ((liquid_gibbs, [1.]), (olivine_gibbs, [1., 0., 0.])):
        with pytest.raises(ValueError):
            function(1900., 1e5, n)
    with pytest.raises(ValueError):
        fayalite_gibbs([1800.], 1e5)
    for T, P in ((1599., 1e5), (1800., 0.), (np.nan, 1e5), (1800., 6e8)):
        with pytest.raises(ValueError):
            example.equilibrium(T, P)
    for ox in ((0., .3, 1.), (1.2, -.1, 1.), (1.2, .3, .75), (np.nan, .3, 1.)):
        with pytest.raises(ValueError):
            example.component_amounts(*ox)
