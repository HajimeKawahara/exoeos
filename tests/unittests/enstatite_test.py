"""Independent Mg orthopyroxene references and conserved insertion tests."""

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos.magma import enstatite_gibbs, forsterite_gibbs, liquid_gibbs


ROOT = Path(__file__).resolve().parents[2]
SOURCE = json.loads((ROOT / "tests/reference/enstatite_source_v1.json").read_text())
BACKEND = json.loads((ROOT / "tests/reference/enstatite_melts_v1.json").read_text())
spec = importlib.util.spec_from_file_location("magma_cooling", ROOT / "examples/magma_cooling.py")
cooling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cooling)


@pytest.mark.parametrize("state", SOURCE["states"])
def test_enstatite_g_s_cp_v_match_source_quadrature_and_site_endpoint(state):
    T, P = state["T_K"], state["P_Pa"]
    assert enstatite_gibbs(T, P) == pytest.approx(state["g_J_mol"], abs=2e-8, rel=0)
    assert -jax.grad(enstatite_gibbs, 0)(T, P) == pytest.approx(state["s_J_mol_K"], abs=1e-10, rel=0)
    assert -T * jax.grad(jax.grad(enstatite_gibbs, 0), 0)(T, P) == pytest.approx(
        state["cp_J_mol_K"], abs=1e-10, rel=0)
    assert jax.grad(enstatite_gibbs, 1)(T, P) == pytest.approx(state["v_m3_mol"], abs=1e-18, rel=0)


@pytest.mark.parametrize("state", BACKEND["states"])
def test_enstatite_matches_actual_melts_endpoint_with_mole_conversion(state):
    # The worker supplied one mol Mg2Si2O6; the API returns per mol MgSiO3.
    expected = state["orthopyroxene_g_J_per_mol_Mg2Si2O6"] / 2
    assert enstatite_gibbs(state["T_K"], state["P_Pa"]) == pytest.approx(expected, abs=2e-8, rel=0)


@pytest.mark.parametrize("T", [2200., 2000., 1950., 1900., 1800.])
def test_insertion_is_a_conserved_directional_derivative(T):
    P, xi = 1e5, cooling.crystal_amount(T)
    n = jnp.array([.25, .75 - xi])

    def energy(eta):
        return (liquid_gibbs(T, P, n - eta / 2) + xi * forsterite_gibbs(T, P)
                + eta * enstatite_gibbs(T, P))

    insertion = cooling.enstatite_insertion_energy(T, P, n)
    assert insertion == pytest.approx(float(jax.grad(energy)(0.)), abs=2e-8, rel=0)
    eta = 1e-5
    # Columns: liquid Q, liquid F, crystalline Fo, crystalline En (MgSiO3).
    elements = np.array([[0, 2, 2, 1], [1, 1, 1, 1], [2, 4, 4, 3]])
    amounts = np.r_[np.asarray(n) - eta / 2, xi, eta]
    np.testing.assert_allclose(elements @ amounts, [1.5, 1., 3.5], atol=1e-15, rtol=0)
    masses = np.array([24.305, 28.0855, 15.9994])
    assert (masses @ elements) @ amounts == pytest.approx(masses @ [1.5, 1., 3.5], rel=2e-15)
    assert float(energy(eta) - energy(0.)) * float(insertion) > 0
    assert (float(insertion) < 0) == (T <= 1900.)


@pytest.mark.parametrize("state", BACKEND["cooling"])
def test_restricted_path_and_insertion_match_independent_melts_bisection(state):
    T, P = state["T_K"], state["P_Pa"]
    xi = cooling.crystal_amount(T, P)
    assert xi == pytest.approx(state["forsterite_mol"], abs=1e-9, rel=0)
    insertion = cooling.enstatite_insertion_energy(T, P, jnp.array([.25, .75 - xi]))
    assert insertion == pytest.approx(state["insertion_J_mol_MgSiO3"], abs=1e-6, rel=0)
    if state["forsterite_mol"] > 0:
        assert state["forsterite_slope_J_mol"] == pytest.approx(0., abs=2e-6, rel=0)
    else:
        assert state["forsterite_slope_J_mol"] >= 0


def test_enstatite_crossing_on_the_restricted_cooling_path():
    def insertion(T):
        xi = cooling.crystal_amount(T)
        return float(cooling.enstatite_insertion_energy(T, 1e5, jnp.array([.25, .75 - xi])))

    assert insertion(1900.) < 0 < insertion(1950.)
    onset = cooling._bisect_increasing(insertion, 1900., 1950.)
    assert insertion(onset) == pytest.approx(0., abs=2e-8, rel=0)
    assert insertion(onset - .01) < 0 < insertion(onset + .01)


def test_enstatite_jit_vmap_and_joint_tp_derivatives():
    states = jnp.array([[1400., 1e5], [1800., 5e7], [2200., 5e8]])
    def energy(tp):
        return enstatite_gibbs(tp[0], tp[1])

    values, derivatives = jax.jit(jax.vmap(jax.value_and_grad(energy)))(states)
    np.testing.assert_allclose(values, [energy(tp) for tp in states], atol=2e-8, rtol=0)
    assert np.all(np.isfinite(derivatives))
    hessian = jax.jit(jax.hessian(energy))(states[1])
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-12, rtol=0)
    for i, step in enumerate([.01, 100.]):
        delta = jnp.eye(2)[i] * step
        finite = (energy(states[1] + delta) - energy(states[1] - delta)) / (2 * step)
        assert derivatives[1, i] == pytest.approx(float(finite), rel=1e-7, abs=1e-9)
    with pytest.raises(ValueError, match="scalars"):
        enstatite_gibbs(states[:, 0], 1e5)
