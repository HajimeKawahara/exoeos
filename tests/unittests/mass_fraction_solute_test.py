"""Mass-basis chemistry verified from independent amounts and scalar derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from exoeos import IdealSolution, mass_fraction_solute_state, total_solution_state


WEIGHTS = jnp.array([.0600843, .20377308])
H2_WEIGHT = .00201588


def mixture(amounts, weights=WEIGHTS):
    host = total_solution_state(IdealSolution(), 1800., 1e5, amounts[:2], jnp.array([-2., 3.]))
    return mass_fraction_solute_state(host.gibbs_RT, host.mu_RT, amounts[:2], weights,
                                      amounts[2], H2_WEIGHT, -4.)


def test_full_scalar_host_response_and_mass_fraction_equilibrium():
    n = jnp.array([1.1, .4, .03])
    state = mixture(n)
    mu = np.r_[state.host_mu_RT, state.solute_mu_RT]
    energy = lambda amounts: mixture(amounts).gibbs_RT
    np.testing.assert_allclose(jax.grad(energy)(n), mu, rtol=2e-12, atol=2e-12)
    finite = []
    for i in range(3):
        step = np.eye(3)[i] * float(n[i]) * 3e-5
        finite.append(float((energy(n + step) - energy(n - step)) / (2 * step[i])))
    np.testing.assert_allclose(finite, mu, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(n @ mu, state.gibbs_RT, rtol=2e-12)
    derivatives = jax.jacfwd(lambda a: jnp.r_[mixture(a).host_mu_RT, mixture(a).solute_mu_RT])(n)
    np.testing.assert_allclose(derivatives, derivatives.T, rtol=2e-12, atol=2e-12)
    w = n[2] * H2_WEIGHT / (WEIGHTS @ n[:2] + n[2] * H2_WEIGHT)
    assert state.solute_mu_RT == pytest.approx(-4. + np.log(w))
    assert abs(state.solute_mu_RT - (-4. + np.log(n[2] / n.sum()))) > 3.


def test_independent_grand_potential_minimum_uses_mass_not_endmember_moles():
    host_n = np.array([1.1, .4])
    requested_w = .003
    reservoir_mu = -4. + np.log(requested_w)
    grand_potential = lambda h: float(mixture(jnp.r_[host_n, h]).gibbs_RT) - reservoir_mu * h
    result = minimize_scalar(grand_potential, bounds=(.001, 1.), method="bounded",
                             options={"xatol": 1e-10})
    mass_fraction = result.x * H2_WEIGHT / (np.asarray(WEIGHTS) @ host_n + result.x * H2_WEIGHT)
    assert result.success
    assert mass_fraction == pytest.approx(requested_w, rel=2e-6)


def test_amount_scaling_host_basis_and_exact_zero():
    n = jnp.array([1.1, .4, .03])
    first = mixture(n)
    scaled = jax.jit(jax.vmap(mixture))(jnp.array([n, n * 7.]))
    np.testing.assert_allclose(scaled.gibbs_RT[1], 7 * first.gibbs_RT, rtol=2e-12)
    np.testing.assert_allclose(scaled.host_mu_RT[1], first.host_mu_RT, rtol=2e-12)
    # Splitting host endmembers changes their count, but not physical mass or dilution G.
    a = mass_fraction_solute_state(0., jnp.zeros(2), n[:2], WEIGHTS, n[2], H2_WEIGHT, -4.)
    b = mass_fraction_solute_state(0., jnp.zeros(2), 2 * n[:2], WEIGHTS / 2, n[2], H2_WEIGHT, -4.)
    assert a.gibbs_RT == pytest.approx(b.gibbs_RT)
    assert a.solute_mu_RT == pytest.approx(b.solute_mu_RT)
    np.testing.assert_allclose(a.host_mu_RT, 2 * b.host_mu_RT)
    zero = mixture(n.at[2].set(0.))
    host = total_solution_state(IdealSolution(), 1800., 1e5, n[:2], jnp.array([-2., 3.]))
    assert zero.gibbs_RT == host.gibbs_RT
    np.testing.assert_array_equal(zero.host_mu_RT, host.mu_RT)
    assert np.isneginf(zero.solute_mu_RT)


@pytest.mark.parametrize("n,weights,h,weight", [([0., 0.], [1., 2.], 1., 1.),
    ([1., -1.], [1., 2.], 1., 1.), ([1., 1.], [0., 2.], 1., 1.),
    ([1., 1.], [1., 2.], -1., 1.), ([1., 1.], [1., 2.], 1., 0.)])
def test_invalid_physical_inputs_remain_invalid(n, weights, h, weight):
    state = mass_fraction_solute_state(0., jnp.zeros(2), n, weights, h, weight, 0.)
    assert np.isnan(state.gibbs_RT) and np.all(np.isnan(state.host_mu_RT))
    assert np.isnan(state.solute_mu_RT)
