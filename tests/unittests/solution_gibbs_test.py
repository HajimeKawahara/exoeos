"""Full solution energy checked against analytic and independent derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import (IdealSolution, MaFeSiOHLiquid, total_solution_gibbs_RT,
                    total_solution_state)


def test_ideal_solution_standards_and_exact_absence():
    standards = jnp.array([-2., 3.])
    amounts = jnp.array([0.6, 1.4])
    state = total_solution_state(IdealSolution(), 1800., 1e5, amounts, standards)
    expected_mu = standards + np.log(np.asarray(amounts) / 2)
    np.testing.assert_allclose(state.mu_RT, expected_mu)
    np.testing.assert_allclose(state.gibbs_RT, amounts @ expected_mu)
    absent = jax.jit(lambda n: total_solution_state(
        IdealSolution(), 1800., 1e5, n, standards))(jnp.zeros(2))
    assert absent.gibbs_RT == 0
    assert np.all(np.isnan(absent.mu_RT))
    edge = total_solution_state(IdealSolution(), 1800., 1e5,
                                jnp.array([2., 0.]), standards)
    assert edge.gibbs_RT == -4.
    assert edge.mu_RT[0] == -2.
    assert np.isneginf(edge.mu_RT[1])


def test_alloy_gradient_hessian_euler_scaling_and_zero_limit():
    model = MaFeSiOHLiquid()
    amounts = jnp.array([0.84, 0.07, 0.03, 0.06]) * 2.3
    standards = jnp.array([-1.3, 2.5, 0.7, -4.1])
    energy = lambda n: total_solution_gibbs_RT(model, 2100., 1e5, n, standards)
    state = total_solution_state(model, 2100., 1e5, amounts, standards)
    derivative = jax.grad(energy)(amounts)
    np.testing.assert_allclose(derivative, state.mu_RT, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state.gibbs_RT, amounts @ state.mu_RT, rtol=1e-12)
    finite_difference = []
    for i in range(4):
        step = np.eye(4)[i] * float(amounts[i]) * 1e-5
        finite_difference.append(float((energy(amounts + step) - energy(amounts - step)) / (2 * step[i])))
    np.testing.assert_allclose(finite_difference, state.mu_RT, atol=1e-9, rtol=1e-9)
    hessian = jax.hessian(energy)(amounts)
    np.testing.assert_allclose(hessian, hessian.T, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(hessian @ amounts, 0, atol=1e-12)
    for scale in (3.7, 1e-20):
        scaled = total_solution_state(model, 2100., 1e5, scale * amounts, standards)
        np.testing.assert_allclose(scaled.gibbs_RT, scale * state.gibbs_RT, rtol=1e-12)
        np.testing.assert_allclose(scaled.mu_RT, state.mu_RT, rtol=1e-12)
    assert energy(jnp.zeros(4)) == 0
    # Atomic standards are explicit; an elemental gauge adds a linear energy.
    gauge = jnp.array([0.2, -0.5, 0.7, 1.1])
    shifted = total_solution_state(model, 2100., 1e5, amounts, standards + gauge)
    np.testing.assert_allclose(shifted.gibbs_RT - state.gibbs_RT, amounts @ gauge)
    np.testing.assert_allclose(shifted.mu_RT - state.mu_RT, gauge)


def test_jit_vmap_and_supported_dry_alloy_boundary():
    model = MaFeSiOHLiquid()
    standards = jnp.zeros(4)
    amounts = jnp.array([0.9, 0.07, 0.03, 0.])
    evaluate = jax.jit(jax.vmap(lambda n: total_solution_state(model, 2000., 1e5, n, standards)))
    states = evaluate(jnp.stack((amounts, amounts * 3, jnp.zeros(4))))
    assert np.isneginf(states.mu_RT[0, 3])
    assert np.all(np.isfinite(states.mu_RT[0, :3]))
    np.testing.assert_allclose(states.gibbs_RT[1], 3 * states.gibbs_RT[0])
    assert states.gibbs_RT[2] == 0
    assert np.all(np.isnan(states.mu_RT[2]))


@pytest.mark.parametrize("n,standards", [([1., 2.], [0.]), ([], []), ([[1.]], [0.])])
def test_reject_mismatched_or_empty_vectors(n, standards):
    with pytest.raises(ValueError):
        total_solution_state(IdealSolution(), 1800., 1e5, n, standards)
