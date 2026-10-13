"""Saved standards, conserved cooling, and native JAX Gibbs derivatives."""

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos.magma import R, W, forsterite_gibbs, liquid_gibbs, liquid_standard_gibbs


ROOT = Path(__file__).resolve().parents[2]
SOURCE = json.loads((ROOT / "tests/reference/magma_source_v1.json").read_text())
SAVED_MELTS = json.loads((ROOT / "tests/reference/melts_silicate_v1.json").read_text())
BACKEND = json.loads((ROOT / "tests/reference/magma_melts_v1.json").read_text())
spec = importlib.util.spec_from_file_location("magma_cooling", ROOT / "examples/magma_cooling.py")
cooling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cooling)


def standards(T, P):
    return jnp.concatenate((liquid_standard_gibbs(T, P), jnp.atleast_1d(forsterite_gibbs(T, P))))


@pytest.mark.parametrize("state", SOURCE["states"])
def test_standards_match_independent_source_quadrature(state):
    np.testing.assert_allclose(standards(state["T_K"], state["P_Pa"]),
                               state["g_J_mol"], rtol=0, atol=2e-8)


@pytest.mark.parametrize("state", SAVED_MELTS["states"])
def test_liquid_standards_match_existing_saved_melts(state):
    np.testing.assert_allclose(liquid_standard_gibbs(state["T_K"], state["P_Pa"]),
                               np.array(state["liquid"]["mu0_J_mol"], float)[[0, 7]],
                               rtol=0, atol=2e-8)


@pytest.mark.parametrize("n", [[.25, .75], [.9, .1], [1e-12, 2.]])
def test_composition_gradient_euler_scaling_and_curvature(n):
    T, P = 2000., 5e7
    n = jnp.array(n)
    x = n / n.sum()
    mu = jax.grad(liquid_gibbs, argnums=2)(T, P, n)
    expected = liquid_standard_gibbs(T, P) + R * T * jnp.log(x) + W * x[::-1]**2
    np.testing.assert_allclose(mu, expected, rtol=0, atol=2e-8)
    energy = liquid_gibbs(T, P, n)
    assert energy == pytest.approx(float(n @ mu), abs=2e-8)
    assert liquid_gibbs(T, P, 13 * n) == pytest.approx(float(13 * energy), rel=2e-15)
    np.testing.assert_allclose(jax.grad(liquid_gibbs, argnums=2)(T, P, 13 * n), mu, atol=2e-8, rtol=0)
    hessian = jax.hessian(liquid_gibbs, argnums=2)(T, P, n)
    np.testing.assert_allclose(hessian @ n, 0, atol=1e-9)


@pytest.mark.parametrize("T", [2200., 2100., 2078., 2050., 2000., 1800.])
def test_cooling_conserves_elements_mass_and_minimizes_gibbs(T):
    xi = cooling.crystal_amount(T)
    liquid = np.array([.25, .75 - xi])
    # Rows: Mg, Si, O. Columns: liquid Q, liquid F, crystalline F.
    formula = np.array([[0, 2, 2], [1, 1, 1], [2, 4, 4]])
    expected_elements = np.array([1.5, 1., 3.5])
    elements = formula @ np.r_[liquid, xi]
    np.testing.assert_allclose(elements, expected_elements, rtol=0, atol=1e-15)
    atomic_masses = np.array([24.305, 28.0855, 15.9994])
    assert (atomic_masses @ formula) @ np.r_[liquid, xi] == pytest.approx(
        atomic_masses @ expected_elements, rel=2e-15)
    assert 0 <= xi < .75
    mu_f = jax.grad(liquid_gibbs, argnums=2)(T, 1e5, liquid)[1]
    slope = float(forsterite_gibbs(T, 1e5) - mu_f)
    assert cooling.crystallization_slope(T, 1e5, xi) == pytest.approx(slope, abs=1e-8)
    if xi == 0:
        assert slope >= 0
    else:
        assert slope == pytest.approx(0., abs=2e-8)
    grid = jax.vmap(lambda value: cooling.total_gibbs(T, 1e5, value))(jnp.linspace(0, .75, 101))
    assert float(cooling.total_gibbs(T, 1e5, xi)) <= float(grid.min()) + 1e-8


def test_onset_and_memo_cooling_values():
    assert cooling.crystal_amount(2100.) == 0.
    assert cooling.crystal_amount(2050.) == pytest.approx(.220, abs=.001)
    assert cooling.crystal_amount(2000.) == pytest.approx(.425, abs=.001)
    onset = cooling._bisect_increasing(lambda t: float(cooling.crystallization_slope(t, 1e5, 0.)), 2000., 2100.)
    assert onset == pytest.approx(2078., abs=1.)
    assert cooling.crystallization_slope(onset, 1e5, 0.) == pytest.approx(0., abs=2e-8)
    assert cooling.crystal_amount(onset + .01) == 0.
    assert cooling.crystal_amount(onset - .01) > 0.


@pytest.mark.parametrize("state", SOURCE["states"])
def test_temperature_pressure_derivatives_from_the_same_gibbs(state):
    T, P = state["T_K"], state["P_Pa"]
    entropy = -jax.jacfwd(standards, 0)(T, P)
    volume = jax.jacfwd(standards, 1)(T, P)
    cp = -T * jax.jacfwd(jax.jacfwd(standards, 0), 0)(T, P)
    np.testing.assert_allclose(entropy, state["s_J_mol_K"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(volume, state["v_m3_mol"], atol=1e-18, rtol=0)
    np.testing.assert_allclose(cp, state["cp_J_mol_K"], atol=1e-10, rtol=0)


def test_jit_vmap_and_joint_interior_derivatives():
    states = jnp.array([[1400., 1e5, .25, .75], [1480., 5e7, 1., 2.], [2100., 5e8, .8, .2]])
    def energy(state):
        return liquid_gibbs(state[0], state[1], state[2:])

    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(energy)))(states)
    np.testing.assert_allclose(values, [energy(s) for s in states], atol=2e-8, rtol=0)
    assert np.all(np.isfinite(gradients))
    hessian = jax.jit(jax.hessian(energy))(states[0])
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-10, rtol=0)
    # Interior finite differences cover all T, P and amount derivatives.
    for i, step in enumerate([.01, 100., 1e-4, 1e-4]):
        delta = jnp.eye(4)[i] * step
        finite_difference = (energy(states[0] + delta) - energy(states[0] - delta)) / (2 * step)
        assert gradients[0, i] == pytest.approx(float(finite_difference), rel=1e-7, abs=1e-6)


def test_continuous_energy_endpoints_and_temperature_branch():
    T, P = 2000., 1e5
    n = jnp.array([[0., 0.], [1., 0.], [0., 1.]])
    energies = jax.jit(jax.vmap(liquid_gibbs, in_axes=(None, None, 0)))(T, P, n)
    np.testing.assert_allclose(energies, np.r_[0., liquid_standard_gibbs(T, P)], atol=1e-8, rtol=0)
    assert liquid_gibbs(T, P, [0., 0.]) == 0.
    assert jax.grad(liquid_gibbs, 0)(T, P, n[0]) == 0.
    assert jax.grad(liquid_gibbs, 1)(T, P, n[0]) == 0.
    np.testing.assert_allclose(liquid_gibbs(T, P, [1e-12, 1.]), energies[2], atol=1e-4, rtol=0)
    for derivative in (standards, jax.jacfwd(standards, 0)):
        np.testing.assert_allclose(derivative(1480. - 1e-7, P), derivative(1480. + 1e-7, P), atol=1e-4, rtol=0)
    assert liquid_standard_gibbs(2163., P)[1] == pytest.approx(float(forsterite_gibbs(2163., P)), abs=1e-8)


def test_inventory_limits_and_eager_validation():
    np.testing.assert_array_equal(cooling.component_amounts(1.5, 1.), [.25, .75])
    assert cooling.crystal_amount(2000., mgo_mol=0.) == 0.
    assert cooling.crystal_amount(2000., mgo_mol=2.) == 1.
    assert cooling.crystal_amount(2200., mgo_mol=2.) == 0.
    for m, s in [(2.1, 1.), (-1., 1.), (1., 0.), (np.nan, 1.)]:
        with pytest.raises(ValueError):
            cooling.component_amounts(m, s)
    with pytest.raises(ValueError):
        cooling.crystal_amount(100.)
    with pytest.raises(ValueError):
        liquid_gibbs(2000., 1e5, [1.])
    with pytest.raises(ValueError):
        liquid_standard_gibbs([2000.], 1e5)


@pytest.mark.parametrize("state", BACKEND["states"])
def test_independent_pinned_backend_standards_mixture_and_potentials(state):
    T, P, n = state["T_K"], state["P_Pa"], np.array(state["n_mol"])
    present = n > 0
    np.testing.assert_allclose(np.asarray(liquid_standard_gibbs(T, P))[present],
                               np.array(state["liquid_standard_J_mol"], float)[present],
                               atol=2e-8, rtol=0)
    assert forsterite_gibbs(T, P) == pytest.approx(state["forsterite_gibbs_J_mol"], abs=2e-8, rel=0)
    # Preserve the observed binary differences separately from source agreement.
    assert liquid_gibbs(T, P, n) == pytest.approx(state["liquid_gibbs_J"], abs=3e-5, rel=0)
    if present.all():
        np.testing.assert_allclose(jax.grad(liquid_gibbs, 2)(T, P, n),
                                   state["liquid_mu_J_mol"], atol=3e-5, rtol=0)


def test_reduced_solution_convention_uses_the_same_gas_constant():
    from exoeos.solution_gibbs import total_solution_gibbs_RT

    class BinaryRegularSolution:
        def gex_RT(self, T, P, x):
            return W * x[0] * x[1] / (R * T)

    T, P, n = 2000., 5e7, jnp.array([.25, .75])
    reduced = total_solution_gibbs_RT(BinaryRegularSolution(), T, P, n,
                                     liquid_standard_gibbs(T, P) / (R * T))
    assert reduced == pytest.approx(float(liquid_gibbs(T, P, n) / (R * T)), abs=1e-12)
