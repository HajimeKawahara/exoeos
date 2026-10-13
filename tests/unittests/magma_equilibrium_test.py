"""Conserved post-enstatite equilibrium, invariant ties and native references."""

import importlib.util
import json
from pathlib import Path
import sys

import jax
import numpy as np
import pytest

from exoeos.magma import R, W, enstatite_gibbs, forsterite_gibbs, liquid_gibbs


ROOT = Path(__file__).resolve().parents[2]
BACKEND = json.loads((ROOT / "tests/reference/magma_equilibrium_melts_v1.json").read_text())
sys.path.insert(0, str(ROOT / "examples"))
try:
    spec = importlib.util.spec_from_file_location("magma_equilibrium", ROOT / "examples/magma_equilibrium.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
finally:
    sys.path.pop(0)

# Rows Mg, Si, O; columns liquid Q, liquid F, solid Fo, solid En (MgSiO3).
ELEMENTS = np.array([[0, 2, 2, 1], [1, 1, 1, 1], [2, 4, 4, 3]])
liquid_mu = jax.jit(jax.grad(liquid_gibbs, 2))
energies = jax.jit(jax.vmap(model.total_gibbs, in_axes=(None, None, 0)))


@pytest.mark.parametrize("state", BACKEND["states"])
def test_equilibrium_matches_independent_native_melts_roots(state):
    inventory = state["initial_oxide_moles"]
    result = model.equilibrium(state["T_K"], state["P_Pa"], inventory["MgO"], inventory["SiO2"])
    np.testing.assert_allclose(result["amounts_mol"], state["amounts_mol"], rtol=0, atol=1e-9)
    assert result["g_J"] == pytest.approx(state["g_J"], abs=3e-5, rel=0)
    assert not result["degenerate"]
    if sum(state["amounts_mol"][:2]) == 0:
        # Native global supporting-line check rejects nucleation at any x_F.
        assert state["minimum_liquid_gap_J_mol_Si"] > 0
    else:
        assert min(state["solid_insertion_J_mol"]) >= -1e-6


@pytest.mark.parametrize("M", [.2, .9, 1., 1.5, 1.9])
@pytest.mark.parametrize("T,P", [(1400., 1e5), (1900., 1e5), (1910., 1e5),
                                  (2200., 1e5), (1800., 5e7), (2000., 5e8)])
def test_global_minimum_conservation_and_present_liquid_stability(M, T, P):
    result = model.equilibrium(T, P, M)
    n = np.array(result["amounts_mol"])
    assert np.all(n >= 0)
    np.testing.assert_allclose(ELEMENTS @ n, [M, 1., M + 2], atol=2e-15, rtol=0)
    masses = np.array([24.305, 28.0855, 15.9994])
    assert (masses @ ELEMENTS) @ n == pytest.approx(masses @ [M, 1., M + 2], rel=2e-15)
    if np.all(n[:2] > 0):
        mu = np.asarray(liquid_mu(T, P, n[:2]))
        slopes = np.array([float(forsterite_gibbs(T, P)) - mu[1],
                           float(enstatite_gibbs(T, P)) - mu.mean()])
        assert min(slopes) >= -2e-7
        np.testing.assert_allclose(slopes[n[2:] > 0], 0., atol=2e-7, rtol=0)

    # A feasible 2-D grid is independent of the candidate-edge construction.
    a, b = M/2, 1-M/2
    eta = np.repeat(np.linspace(0., 2*min(a, b), 31), 31)
    xi = np.tile(np.linspace(0., 1., 31), 31) * (a - eta/2)
    trials = np.column_stack([np.maximum(0., b - eta/2),
                              np.maximum(0., a - eta/2 - xi), xi, eta])
    assert result["g_J"] <= float(energies(T, P, trials).min()) + 2e-8


@pytest.mark.parametrize("M", [.9, 1., 1.5])
def test_invariant_returns_the_full_conserved_equal_energy_segment(M):
    T = model.invariant_temperature()
    assert T == pytest.approx(BACKEND["invariant"]["T_K"], abs=1e-7, rel=0)
    result = model.equilibrium(T, mgo_mol=M)
    ends = np.array(result["equilibrium_endpoints_mol"])
    assert result["degenerate"]
    assert ends.shape == (2, 4)
    np.testing.assert_array_equal(result["amounts_mol"], ends[0])
    assert ends[0, :2].sum() < ends[1, :2].sum()
    assert ends[0, 3] > 0 and ends[1, 3] == 0
    weight = np.linspace(0., 1., 11)[:, None]
    segment = weight * ends[0] + (1-weight) * ends[1]
    np.testing.assert_allclose(segment @ ELEMENTS.T, np.tile([M, 1., M+2], (11, 1)), atol=2e-15)
    np.testing.assert_allclose(energies(T, 1e5, segment), result["g_J"], atol=2e-8, rtol=0)
    assert np.all(segment[5] > 0)
    mu = np.asarray(liquid_mu(T, 1e5, segment[5, :2]))
    np.testing.assert_allclose([forsterite_gibbs(T, 1e5) - mu[1],
                               enstatite_gibbs(T, 1e5) - mu.mean()], 0., atol=2e-8)
    for delta in [-.01, .01]:
        assert not model.equilibrium(T+delta, mgo_mol=M)["degenerate"]


def test_post_enstatite_reaction_consumes_forsterite_and_can_exhaust_liquid():
    warm = np.array(model.equilibrium(1910.)["amounts_mol"])
    cold = np.array(model.equilibrium(1900.)["amounts_mol"])
    np.testing.assert_array_equal(cold, [0., 0., .5, .5])
    assert cold[2] < warm[2]
    restricted = np.array([.25, .75-model.crystal_amount(1900.), model.crystal_amount(1900.), 0.])
    assert model.total_gibbs(1900., 1e5, cold) < model.total_gibbs(1900., 1e5, restricted)
    silica_rich = np.array(model.equilibrium(1900., mgo_mol=.9)["amounts_mol"])
    assert np.all(silica_rich[:2] > 0) and silica_rich[2] == 0 and silica_rich[3] > 0


@pytest.mark.parametrize("scale", [1e-6, 3., 1e6])
@pytest.mark.parametrize("T,M", [(2000., 1.5), (1900., 1.5), (1900., .9)])
def test_amounts_and_energy_scale_with_inventory(T, M, scale):
    base = model.equilibrium(T, mgo_mol=M)
    scaled = model.equilibrium(T, mgo_mol=scale*M, sio2_mol=scale)
    np.testing.assert_allclose(np.array(scaled["amounts_mol"])/scale, base["amounts_mol"], atol=2e-12, rtol=0)
    assert scaled["g_J"]/scale == pytest.approx(base["g_J"], abs=2e-8, rel=0)


def test_pure_endpoints_and_pure_forsterite_melting_tie():
    np.testing.assert_array_equal(model.equilibrium(1800., mgo_mol=0.)["amounts_mol"], [1., 0., 0., 0.])
    np.testing.assert_array_equal(model.equilibrium(1800., mgo_mol=2.)["amounts_mol"], [0., 0., 1., 0.])
    np.testing.assert_array_equal(model.equilibrium(2200., mgo_mol=2.)["amounts_mol"], [0., 1., 0., 0.])
    result = model.equilibrium(2163., mgo_mol=2.)
    assert result["degenerate"]
    np.testing.assert_array_equal(result["equilibrium_endpoints_mol"], [[0., 0., 1., 0.], [0., 1., 0., 0.]])


@pytest.mark.parametrize("inputs", [{"T": W/(2*R)}, {"T": np.nan}, {"P": 0.}, {"P": np.inf},
                                    {"mgo_mol": -1.}, {"mgo_mol": 2.1}, {"sio2_mol": 0.}])
def test_invalid_inputs(inputs):
    with pytest.raises(ValueError):
        model.equilibrium(**dict({"T": 1900.}, **inputs))
