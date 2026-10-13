"""Silica branches, MELTS comparisons, and conserved multiphase equilibrium."""

import importlib.util
import json
from pathlib import Path
import sys

import jax
import numpy as np
import pytest

from exoeos.magma import R, W, liquid_gibbs, liquid_standard_gibbs, silica_gibbs


ROOT = Path(__file__).resolve().parents[2]
BACKEND = json.loads((ROOT / "tests/reference/silica_melts_v1.json").read_text())
sys.path.insert(0, str(ROOT / "examples"))
try:
    spec = importlib.util.spec_from_file_location("magma_multiphase", ROOT / "examples/magma_multiphase.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
finally:
    sys.path.pop(0)

ELEMENTS = np.array([[0, 2, 2, 1, 0, 0, 0], [1]*7, [2, 4, 4, 3, 2, 2, 2]])
mu_liquid = jax.jit(jax.grad(liquid_gibbs, 2))
energies = jax.jit(jax.vmap(model.total_gibbs, in_axes=(None, None, 0)))


def test_silica_standards_match_melts_across_alpha_beta_pressure_branches():
    states = BACKEND["standards"]
    T, P = np.array([[s["T_K"], s["P_Pa"]] for s in states]).T
    actual = jax.jit(jax.vmap(silica_gibbs))(T, P)
    np.testing.assert_allclose(actual, [s["g_J_mol"] for s in states], atol=2e-8, rtol=0)


@pytest.mark.parametrize("s", BACKEND["derivatives"])
def test_silica_derivatives_match_native_finite_differences(s):
    T, P = s["T_K"], s["P_Pa"]
    np.testing.assert_allclose(-jax.jacfwd(silica_gibbs, 0)(T, P), s["s_J_mol_K"], atol=5e-7, rtol=0)
    np.testing.assert_allclose(jax.jacfwd(silica_gibbs, 1)(T, P), s["v_m3_mol"], atol=4e-12, rtol=0)
    cp = -T*jax.jacfwd(jax.jacfwd(silica_gibbs, 0), 0)(T, P)
    assert np.all(np.isfinite(cp)) and np.all(cp > 0)
    mixed1 = jax.jacfwd(jax.jacfwd(silica_gibbs, 0), 1)(T, P)
    mixed2 = jax.jacfwd(jax.jacfwd(silica_gibbs, 1), 0)(T, P)
    np.testing.assert_allclose(mixed1, mixed2, atol=1e-18, rtol=0)


@pytest.mark.parametrize("s", BACKEND["states"])
def test_amounts_and_remaining_liquid_match_independent_melts_roots(s):
    actual = model.equilibrium(s["T_K"], s["P_Pa"], s["MgO_mol"])
    np.testing.assert_allclose(actual["amounts_mol"], s["amounts_mol"], atol=1e-9, rtol=0)
    assert actual["g_J"] == pytest.approx(s["g_J"], abs=3e-5, rel=0)
    if s["liquid_x_F"] is None:
        assert actual["liquid_x_F"] is None and actual["liquid_MgO_SiO2"] is None
    else:
        assert actual["liquid_x_F"] == pytest.approx(s["liquid_x_F"], abs=1e-10, rel=0)


@pytest.mark.parametrize("M", [0., .1, .5, .9, 1., 1.5, 2.])
@pytest.mark.parametrize("T,P", [(500., 1e5), (1800., 1e5), (1900., 1e5),
                                  (2200., 1e5), (1400., 5e7), (2000., 5e8)])
def test_conservation_stability_and_global_energy(M, T, P):
    result = model.equilibrium(T, P, M)
    n = np.array(result["amounts_mol"])
    assert min(n) >= 0
    np.testing.assert_allclose(ELEMENTS@n, [M, 1., M+2], atol=2e-15, rtol=0)
    masses = np.array([24.305, 28.0855, 15.9994])
    assert (masses@ELEMENTS)@n == pytest.approx(masses@[M, 1., M+2], rel=2e-15)
    if np.all(n[:2] > 0):
        mu = mu_liquid(T, P, n[:2])
        slopes = np.asarray(model.solid_gibbs(T, P))-model.SOLID_COMPONENTS.T@mu
        assert min(slopes) >= -2e-7
        np.testing.assert_allclose(slopes[n[2:] > 0], 0., atol=2e-7, rtol=0)

    # Independent feasible states with simultaneous Fo, En and all three
    # silica solids, rather than a grid restricted to candidate edges.
    rng = np.random.default_rng(716)
    trial = np.zeros((1000, 7))
    trial[:, 3] = rng.uniform(size=1000)*min(M, 2-M)
    trial[:, 2] = rng.uniform(size=1000)*(M/2-trial[:, 3]/2)
    silica = rng.uniform(size=1000)*(1-M/2-trial[:, 3]/2)
    trial[:, 4:] = silica[:, None]*rng.dirichlet([1., 1., 1.], 1000)
    trial[:, 0] = np.maximum(0., 1-M/2-trial[:, 3]/2-silica)
    trial[:, 1] = np.maximum(0., M/2-trial[:, 3]/2-trial[:, 2])
    assert result["g_J"] <= float(energies(T, P, trial).min())+2e-8


@pytest.mark.parametrize("M", [.1, .5, .9])
def test_enstatite_silica_invariant_returns_equal_energy_endpoints(M):
    T = BACKEND["enstatite_tridymite_invariant"]["T_K"]
    result = model.equilibrium(T, mgo_mol=M)
    assert result["degenerate"]
    ends = np.array(result["equilibrium_endpoints_mol"])
    assert ends.shape == (2, 7)
    assert ends[0, :2].sum() == 0 and ends[1, :2].sum() > 0
    weights = np.linspace(0., 1., 11)[:, None]
    segment = weights*ends[0]+(1-weights)*ends[1]
    np.testing.assert_allclose(energies(T, 1e5, segment), result["g_J"], atol=3e-8, rtol=0)
    np.testing.assert_allclose(segment@ELEMENTS.T, np.tile([M, 1., M+2], (11, 1)), atol=2e-15)


def test_polymorph_tie_and_cooling_appearance_disappearance():
    T = BACKEND["quartz_tridymite_transition_T_K"]
    tie = model.equilibrium(T, mgo_mol=.5)
    assert tie["degenerate"] and len(tie["equilibrium_endpoints_mol"]) == 2
    assert model.equilibrium(T+1, mgo_mol=0.)["phases"] == ["tridymite"]
    assert model.equilibrium(T-1, mgo_mol=0.)["phases"] == ["quartz"]
    path = model.cooling_path([2200., 1925., 1900., 1800.], mgo_mol=.9)
    assert [s["phases"] for s in path["states"]] == [
        ["liquid"], ["liquid", "forsterite"], ["liquid", "enstatite"], ["enstatite", "tridymite"]]
    assert path["events"][-1] == {"T_bracket_K": [1800., 1900.],
                                  "appeared": ["tridymite"], "disappeared": ["liquid"]}


@pytest.mark.parametrize("scale", [1e-6, 3., 1e6])
def test_scaling_and_pure_melting_degeneracy(scale):
    for T, M in [(1900., .9), (1800., .5), (500., .5)]:
        base = model.equilibrium(T, mgo_mol=M)
        scaled = model.equilibrium(T, mgo_mol=M*scale, sio2_mol=scale)
        np.testing.assert_allclose(np.array(scaled["amounts_mol"])/scale, base["amounts_mol"], atol=2e-12, rtol=0)
        assert scaled["g_J"]/scale == pytest.approx(base["g_J"], abs=2e-8, rel=0)
    assert model.equilibrium(2163., mgo_mol=2.)["degenerate"]


def test_input_validation():
    for args in [{"T": W/(2*R)}, {"T": np.nan}, {"P": 0}, {"P": np.inf},
                 {"mgo_mol": -1.}, {"mgo_mol": 2.1}, {"sio2_mol": 0.}]:
        with pytest.raises(ValueError):
            model.equilibrium(**dict({"T": 1900.}, **args))
    for temperatures in [[], [1800.], [1800., 1900.], [1900., 1900.], [[1800., 1700.]]]:
        with pytest.raises(ValueError):
            model.cooling_path(temperatures)
    with pytest.raises(ValueError):
        silica_gibbs([1800.], 1e5)


def test_pure_silica_melting_preserves_both_endpoints():
    T = model._bisect_increasing(
        lambda t: float(np.min(silica_gibbs(t, 1e5))-liquid_standard_gibbs(t, 1e5)[0]),
        2200., 3000.)
    assert model.equilibrium(T+1, mgo_mol=0.)["phases"] == ["liquid"]
    assert model.equilibrium(T-1, mgo_mol=0.)["phases"] == ["tridymite"]
    result = model.equilibrium(T, mgo_mol=0.)
    assert result["degenerate"]
    np.testing.assert_array_equal(result["equilibrium_endpoints_mol"],
                                  [[0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0]])
