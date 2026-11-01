"""Material sensitivities checked against analytic laws and independent probes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import GibbsResponse, MaFeSiOHLiquid, gibbs_response, total_solution_gibbs_RT
from exoeos.constants import MOLAR_GAS_CONSTANT as R
from exoeos import phosphorus_mqm


def ideal_gibbs(t, p, n):
    standards = jnp.array([17., -29.]) * t + jnp.array([2e-6, 3e-6]) * p
    return n @ standards + R*t*jnp.sum(n*jnp.log(n/n.sum()))


def test_ideal_response_has_analytic_temperature_pressure_amount_derivatives():
    t, p, n = jnp.array(2100.), jnp.array(1e7), jnp.array([.6, 1.4])
    dt, dp, dn = jnp.array(13.), jnp.array(2e5), jnp.array([.03, -.02])
    evaluate = jax.jit(lambda t, p, n: gibbs_response(ideal_gibbs, t, p, n))
    state, response = jax.jvp(evaluate, (t, p, n), (dt, dp, dn))
    assert isinstance(state, GibbsResponse)
    a, v = np.array([17., -29.]), np.array([2e-6, 3e-6])
    logx = np.log(n/n.sum())
    expected_mu = a*t + v*p + R*t*logx
    expected_dmu = (a + R*logx)*dt + v*dp + R*t*(dn/n-dn.sum()/n.sum())
    np.testing.assert_allclose(state.chemical_potentials_j_mol, expected_mu, rtol=1e-13)
    np.testing.assert_allclose(response.chemical_potentials_j_mol, expected_dmu, rtol=1e-13)
    expected_dg = n @ (a+R*logx)*dt + n @ v*dp + expected_mu @ dn
    np.testing.assert_allclose(response.gibbs_j, expected_dg, rtol=1e-13)


def alloy_gibbs(t, p, n):
    standards = jnp.array([-1.3, 2.5, .7, -4.1]) + (p-1e5)*1e-10
    return R*t*total_solution_gibbs_RT(MaFeSiOHLiquid(), t, p, n, standards)


def test_nonideal_jvp_matches_finite_difference_and_transposed_response():
    primals = (jnp.array(2100.), jnp.array(1e5), jnp.array([.84, .07, .03, .06]))
    direction = (jnp.array(12.), jnp.array(30.), jnp.array([.02, -.01, .004, .003]))
    evaluate = lambda t, p, n: gibbs_response(alloy_gibbs, t, p, n)
    value, response = jax.jvp(evaluate, primals, direction)
    step = 1e-4
    plus = evaluate(*(x+step*dx for x, dx in zip(primals, direction)))
    minus = evaluate(*(x-step*dx for x, dx in zip(primals, direction)))
    for derivative, high, low in zip(response, plus, minus):
        np.testing.assert_allclose(derivative, (high-low)/(2*step), rtol=2e-8, atol=1e-6)
    weight = GibbsResponse(jnp.array(.3), jnp.array([.2, -.1, .7, .8]))
    _, pullback = jax.vjp(evaluate, *primals)
    cotangent = pullback(weight)
    lhs = sum(jnp.vdot(w, d) for w, d in zip(weight, response))
    rhs = sum(jnp.vdot(w, d) for w, d in zip(cotangent, direction))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)
    np.testing.assert_allclose(value.gibbs_j, primals[2] @ value.chemical_potentials_j_mol)


@pytest.mark.parametrize('scale', [1e-20, 3.7, 1e20])
def test_amount_scale_preserves_potentials_and_log_amount_response(scale):
    n, dn = jnp.array([.84, .07, .03, .06]), jnp.array([.02, -.01, .004, .003])
    evaluate = lambda amounts: gibbs_response(alloy_gibbs, 2100., 1e5, amounts)
    value, derivative = jax.jvp(evaluate, (n,), (dn,))
    scaled, scaled_derivative = jax.jvp(evaluate, (scale*n,), (scale*dn,))
    np.testing.assert_allclose(scaled.gibbs_j, scale*value.gibbs_j, rtol=2e-12)
    np.testing.assert_allclose(scaled.chemical_potentials_j_mol, value.chemical_potentials_j_mol, rtol=2e-12)
    np.testing.assert_allclose(scaled_derivative.chemical_potentials_j_mol,
                               derivative.chemical_potentials_j_mol, rtol=2e-12)


def test_pair_coordinate_basis_is_preserved_without_implicit_relaxation():
    # FeFe, PP, FeP are material coordinates, not single-atom mol amounts.
    callback = lambda t, p, n: phosphorus_mqm.gibbs_j(t, n, jnp.zeros(2), model='you2020')
    n = jnp.array([1.8, .15, .4])
    response = gibbs_response(callback, 2173.15, 1e5, n)
    step = 1e-5
    expected = []
    for vector in np.eye(3):
        expected.append((callback(2173.15, 1e5, n+step*vector)
                         -callback(2173.15, 1e5, n-step*vector))/(2*step))
    np.testing.assert_allclose(response.chemical_potentials_j_mol, expected, rtol=2e-8)
    np.testing.assert_allclose(response.gibbs_j, n @ response.chemical_potentials_j_mol, rtol=1e-12)


def test_captured_material_parameter_is_differentiable_and_batches_are_explicit():
    n = jnp.array([.6, 1.4])
    def evaluate(strength, amounts):
        callback = lambda t, p, q: ideal_gibbs(t, p, q) + strength*q[0]*q[1]/q.sum()
        return gibbs_response(callback, 2100., 1e5, amounts)
    derivative = jax.jacfwd(lambda s: evaluate(s, n).chemical_potentials_j_mol)(100.)
    np.testing.assert_allclose(derivative, np.array([n[1]**2, n[0]**2])/n.sum()**2)
    states = jax.jit(jax.vmap(lambda q: evaluate(100., q)))(jnp.stack([n, 3*n]))
    np.testing.assert_allclose(states.chemical_potentials_j_mol[0], states.chemical_potentials_j_mol[1])


@pytest.mark.parametrize('t,p,n', [([2100.], 1e5, [.5, .5]), (2100., [1e5], [.5, .5]),
                                     (2100., 1e5, []), (2100., 1e5, [[.5, .5]])])
def test_shape_contract(t, p, n):
    with pytest.raises(ValueError):
        gibbs_response(ideal_gibbs, t, p, n)


def test_callback_must_be_callable_and_scalar_valued():
    with pytest.raises(TypeError, match='callable'):
        gibbs_response(None, 2100., 1e5, [.5, .5])
    with pytest.raises(TypeError, match='scalar-output'):
        gibbs_response(lambda t, p, n: n, 2100., 1e5, [.5, .5])
