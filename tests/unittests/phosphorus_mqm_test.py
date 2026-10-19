"""Atom basis, full MQM entropy, absolute reference and derivative checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import phosphorus_mqm as p
from exoeos._conditional_data import R

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("model", p.MODELS)
def test_source_entropy_atom_basis_and_extensivity(model):
    t, n = 1873.15, np.array([1.3, .17, .8])
    z = 6 if model.endswith("II") else 3
    atoms = np.array([n[0]/3+n[2]/z, n[1]/3+n[2]/6])
    x, y = n/n.sum(), atoms/atoms.sum()
    yf, yp = x[0]+x[2]/2, x[1]+x[2]/2
    # Direct source logarithmic ratios, independently of the provider's
    # cancellation-safe entropy form.
    entropy = atoms @ np.log(y)+n @ np.log(x/[yf*yf, yp*yp, 2*yf*yp])
    coeff = np.asarray(p.parameters_j_mol(t, model=model))
    standard = jnp.array([-12., 7.])
    expected = R*t*(atoms @ standard+entropy)+.5*n[2]*(coeff @ [1, x[0], x[0]**2, x[1]])
    g = jax.jit(lambda v, s: p.gibbs_j(t, v, s, model=model))
    np.testing.assert_allclose(p.atom_matrix(model=model) @ n, atoms, atol=1e-15)
    np.testing.assert_allclose(g(n, standard), expected, rtol=2e-14)
    np.testing.assert_allclose(g(7*n, standard), 7*expected, rtol=2e-14)
    mu = jax.grad(g)(n, standard)
    np.testing.assert_allclose(mu @ n, expected, rtol=2e-14)
    shift = np.array([.2, -.7])
    np.testing.assert_allclose(g(n, standard+shift)-expected, R*t*(atoms @ shift), atol=1e-9, rtol=0)
    h = jax.hessian(g)(n, standard)
    np.testing.assert_allclose(h, h.T, atol=2e-9)
    step = 1e-5
    fd = [(g(n+step*v, standard)-g(n-step*v, standard))/(2*step) for v in np.eye(3)]
    np.testing.assert_allclose(mu, fd, atol=2e-5)
    assert float(g(jnp.zeros(3), standard)) == 0.
    np.testing.assert_allclose(g(jnp.array([3., 0., 0.]), standard), R*t*standard[0])
    np.testing.assert_allclose(g(jnp.array([0., 3., 0.]), standard), R*t*standard[1])


@pytest.mark.parametrize("model", p.MODELS)
@pytest.mark.parametrize("t", [1473.15, 1873.15, 2173.15])
def test_dilute_limit_from_an_independent_feasible_path(model, t):
    # The FeP-saturated path differs from the true minimum by O(x_P**2).
    # Its derivative is therefore an independent check of the Henry limit.
    z = 6 if model.endswith("II") else 3
    def path(x):
        return p.gibbs_j(t, jnp.array([3*(1-x)-18*x/z, 0., 6*x]), jnp.zeros(2), model=model)
    x = 1e-9
    relative_mu = jax.grad(path)(x)/(R*t)-np.log(x)
    np.testing.assert_allclose(relative_mu, p.henry_log_gamma(t, model=model), atol=3e-7)
    if model == "you2020":
        # Eq.3.24 in the source is rounded and uses an older gas constant.
        assert abs(float(relative_mu)-(-22871/t+4.3704)) < .001


def test_sgte_reference_range_and_temperature_derivative():
    t = 1873.15
    np.testing.assert_allclose(p.sgte_liquid_p_j(t), -7232.449+133.291873*t-26.326*t*np.log(t))
    np.testing.assert_allclose(jax.grad(p.sgte_liquid_p_j)(t), 133.291873-26.326*(np.log(t)+1))
    assert np.all(np.isnan(p.sgte_liquid_p_j(jnp.array([300., 3001.]))))


def test_static_contracts():
    with pytest.raises(ValueError):
        p.atom_matrix(model="unknown")
    with pytest.raises(ValueError):
        p.gibbs_j(1873., jnp.ones(4), jnp.zeros(2), model="you2020")
