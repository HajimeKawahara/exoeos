"""Primary-parameter, reaction, reference and extensivity checks for finite P."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from exoeos import associated_metal as old
from exoeos import phosphorus_association as new
from exoeos._conditional_data import R

jax.config.update("jax_enable_x64", True)


def binary_reference(t, x):
    """Independent analytic reaction derivative of the Table 1 polynomial."""
    dg, l0, l1, m0, m1 = (-147357+27.537*t, -27783+8.676*t,
                           -29128+18.174*t, -31649, -37695)
    limit = min(x, (1-x)/2)

    def affinity(fraction):
        extent = fraction*limit
        n = np.array([1-x-2*extent, x-extent, extent])
        f, p, a = y = n/n.sum()
        excess = f*a*(l0+l1*(f-a))+a*p*(m0+m1*(a-p))
        grad = np.array([a*l0+l1*(2*f*a-a*a),
                         a*m0+m1*(a*a-2*a*p),
                         f*l0+l1*(f*f-2*f*a)+p*m0+m1*(2*a*p-p*p)])
        return (dg+grad @ (np.array([-2., -1., 1.])+2*y)-2*excess
                +R*t*(np.log(a)-2*np.log(f)-np.log(p)))

    extent = brentq(affinity, 1e-13, 1-1e-13, xtol=5e-15)*limit
    return np.array([1-x-2*extent, x-extent, extent])


def binary_energy(t, n):
    y = n/jnp.sum(n)
    return R*t*jnp.sum(n*jnp.log(y))+new.binary_nonideal_j(t, n)


@pytest.mark.parametrize("t,x", [(1673., .02), (1873., .1), (1873., .3),
                                  (2173.15, .02), (2173.15, .3), (2173.15, .7)])
def test_binary_reaction_and_atom_basis(t, x):
    n = binary_reference(t, x)
    np.testing.assert_allclose(new.binary_gibbs_j(t, n, jnp.zeros(2)), binary_energy(t, n), rtol=3e-14)
    mu = jax.grad(lambda v: binary_energy(t, v))(jnp.asarray(n))
    assert abs(float(mu[2]-2*mu[0]-mu[1])) < 3e-7
    np.testing.assert_allclose([n[0]+2*n[2], n[1]+n[2]], [1-x, x], atol=1e-15)
    np.testing.assert_allclose(mu @ n, binary_energy(t, n), rtol=3e-14)
    direction = jnp.array([-2., -1., 1.])
    assert float(direction @ jax.hessian(lambda v: binary_energy(t, v))(n) @ direction) > 0


@pytest.mark.parametrize("t", [1673., 1873., 2173.15])
def test_henry_reference_is_the_equilibrated_dilute_limit(t):
    n = binary_reference(t, 1e-9)
    mu = jax.grad(lambda v: binary_energy(t, v))(jnp.asarray(n))/(R*t)
    # The caller's Henry reference is exactly preserved after association.
    assert abs(float(mu[1]+new.henry_shift_rt(t)-np.log(1e-9))) < 1e-6


def test_parent_recovery_and_pure_phosphorus_endpoint():
    t, p = 2173.15, 4.4e7
    n = jnp.array([.9, .01, .002, .004, .012]+[.001]*15)
    standard = jnp.linspace(-20., 5., 20)
    eps, matrix = old.interaction_parameters(t)
    quad = lambda v: new.gibbs_j(t, p, v, standard, eps, matrix, model="quadratic")
    parent = lambda v: old.gibbs_j(t, p, v, standard, eps, matrix)
    np.testing.assert_allclose(quad(n), parent(n), rtol=3e-15)
    np.testing.assert_allclose(jax.grad(quad)(n), jax.grad(parent)(n), rtol=3e-14)
    no_p = n.at[4].set(0.)
    assoc = lambda v: new.gibbs_j(t, p, v, standard, eps, matrix, model="association")
    np.testing.assert_allclose(assoc(jnp.r_[no_p, 0.]), parent(no_p), rtol=3e-15)
    pure = jnp.zeros(20).at[4].set(1.)
    np.testing.assert_allclose(quad(pure)/(R*t), standard[4]+.5*eps[0])
    np.testing.assert_allclose(assoc(jnp.r_[pure, 0.])/(R*t), standard[4]+new.henry_shift_rt(t))
    assert float(quad(jnp.zeros(20))) == 0
    assert float(assoc(jnp.zeros(21))) == 0


def test_full_scalar_extensivity_reciprocity_and_elemental_gauge():
    t, p = 2173.15, 4.4e7
    n = jnp.array([.7, .01, .002, .004, .012]+[.001]*15+[.08])
    standard = jnp.linspace(-20., 5., 20)
    eps, matrix = old.interaction_parameters(t)
    g = jax.jit(lambda v, s: new.gibbs_j(t, p, v, s, eps, matrix, model="association"))
    grad = jax.grad(g)(n, standard)
    hess = jax.hessian(g)(n, standard)
    np.testing.assert_allclose(g(7.3*n, standard), 7.3*g(n, standard), rtol=3e-14)
    np.testing.assert_allclose(grad @ n, g(n, standard), rtol=3e-14)
    np.testing.assert_allclose(hess, hess.T, atol=2e-9, rtol=3e-12)
    elements = sorted({e for row in new.FORMULAS for e in row})
    atoms = np.array([[row.get(e, 0.) for row in new.FORMULAS] for e in elements])
    shift = np.linspace(-.3, .8, len(elements))
    expected = R*t*(shift @ atoms @ n)
    np.testing.assert_allclose(g(n, standard+shift @ atoms[:, :20])-g(n, standard), expected, rtol=1e-13)
    np.testing.assert_allclose(atoms @ n, atoms[:, :20] @ new.collapse_amounts(n), rtol=3e-15)


def test_binary_limit_of_full_scalar_has_one_ideal_term():
    t = 2173.15
    binary = jnp.asarray(binary_reference(t, .2))
    n = jnp.zeros(21).at[jnp.array([0, 4, 20])].set(binary)
    eps, matrix = old.interaction_parameters(t)
    g = new.gibbs_j(t, 1e5, n, jnp.zeros(20), eps, matrix, model="association")
    expected = binary_energy(t, binary)+R*t*.2*new.henry_shift_rt(t)
    np.testing.assert_allclose(g, expected, rtol=3e-14)
    # Dynamic temperature dependence includes both the fit and reference shift.
    scalar = lambda temp: new.gibbs_j(temp, 1e5, n, jnp.zeros(20), eps, matrix, model="association")
    numerical = (scalar(t+.01)-scalar(t-.01))/.02
    np.testing.assert_allclose(jax.grad(scalar)(t), numerical, atol=1e-6)


def test_invalid_static_contracts_are_rejected():
    with pytest.raises(ValueError):
        new.binary_nonideal_j(2173.15, jnp.ones(4))
    with pytest.raises(ValueError):
        new.gibbs_j(2173.15, 1e5, jnp.ones(20), jnp.zeros(20), jnp.ones(4), jnp.zeros((18, 18)), model="association")
