"""Independent quadrature and thermodynamic checks for two-power water."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from exoeos import magma, magma_volatiles as liquid
from exoeos._conditional_data import R
from exoeos._water_isotherm import two_power_water_g_rt


@pytest.mark.parametrize("w", [1e-12, 1e-5, .001, .1, .375, .5])
@pytest.mark.parametrize("a,c", [([.807805, .449387], [5e-5, 6e-4]),
                                ([.25, 1.], [.003, .0002]), ([.5, .5], [.001, .002])])
def test_inverse_derivative_and_independent_scalar_integral(w, a, c):
    a, c = np.asarray(a), np.asarray(c)
    s, h = 2.3, 2.3*w/(1-w)
    z = brentq(lambda z: np.logaddexp.reduce(np.log(c)+a*z)-np.log(w), -200., 200.)
    # Integrate over log pressure, separate from the implementation's power transform.
    integral = quad(lambda y: np.sum(c*np.exp(a*y))/(1-np.sum(c*np.exp(a*y))),
                    z-200., z, epsabs=1e-14, epsrel=2e-12)[0]
    fn = lambda h: two_power_water_g_rt(h, s, c, a)
    assert float(fn(h)) == pytest.approx(h*z-s*integral, abs=2e-13, rel=2e-12)
    assert float(jax.grad(fn)(h)) == pytest.approx(z, abs=3e-12)
    expected_curvature = s/(s+h)**2/np.sum(a*c*np.exp(a*z))
    assert float(jax.grad(jax.grad(fn))(h)) == pytest.approx(expected_curvature, rel=2e-11)


def test_full_scalar_composition_temperature_pressure_and_h2_response():
    t, p = 2173.15, 2.7e7
    dry = jnp.linspace(.01, .05, 11)
    masses = jnp.asarray(liquid.water_parameters()["dry_molar_masses_kg_mol"])
    gas = jnp.array([-30., -20., -15.])
    n = jnp.r_[dry, .004, .0002, 1e-6]

    def fun(n, t=t, p=p, gas=gas):
        # Synthetic, explicitly composition-dependent coefficients (not a fitted law).
        c = jnp.array([5e-5*jnp.exp(n[0]/jnp.sum(n[:11])+100/t), 6e-4*jnp.exp(p/1e9)])
        return liquid.gibbs_j(t, p, n, gas, masses, water_model="two_power_law",
                             water_capacity_1bar=c, water_pressure_exponent=jnp.array([.8, .45]))/(R*t)

    mu, hessian = jax.jit(jax.grad(fun))(n), jax.jit(jax.hessian(fun))(n)
    assert float(mu @ n) == pytest.approx(float(fun(n)), abs=5e-13)
    np.testing.assert_allclose(hessian @ n, 0., atol=4e-12)
    np.testing.assert_allclose(hessian, hessian.T, atol=4e-12)
    for direction in (jnp.eye(14)[0], jnp.eye(14)[11], n*jnp.linspace(-.2, .3, 14)):
        step = 1e-5
        numeric = (jax.grad(fun)(n+step*direction)-jax.grad(fun)(n-step*direction))/(2*step)
        np.testing.assert_allclose(hessian @ direction, numeric, atol=2e-7, rtol=3e-6)
    shifted = jax.grad(lambda n: fun(n, gas=gas+jnp.array([3., 0., 0.])))(n)
    np.testing.assert_allclose(shifted-mu, jnp.eye(14)[11]*3, atol=2e-13)
    baseline = jax.grad(lambda n: liquid.gibbs_j(t, p, n, gas, masses)/(R*t))(n)
    np.testing.assert_allclose(mu[-2:], baseline[-2:], atol=2e-13)
    for x, f in ((t, lambda x: fun(n, t=x)), (p, lambda x: fun(n, p=x))):
        eps = x*1e-4
        assert float(jax.grad(f)(x)) == pytest.approx(float((f(x+eps)-f(x-eps))/(2*eps)), rel=3e-7)


def test_single_exponent_limit_zero_and_numerical_domain():
    s, h, alpha = 1.5, .03, .5
    expected = (-s*np.log1p(h/s)+h*np.log(h/(s+h)))/alpha-h*np.log(.003)/alpha
    assert float(two_power_water_g_rt(h, s, [.001, .002], [alpha, alpha])) == pytest.approx(expected, abs=1e-14)
    assert two_power_water_g_rt(0., s, [.001, .002], [.8, .45]) == 0.
    for water, equivalent, c, a in [(-.1, s, [.001, .002], [.8, .45]),
                                   (.1, 0., [.001, .002], [.8, .45]),
                                   (.1, s, [-.001, .002], [.8, .45]),
                                   (.1, s, [.001, .002], [.1, .45]),
                                   (2., 1., [.001, .002], [.8, .45])]:
        assert np.isnan(two_power_water_g_rt(water, equivalent, c, a))
    with pytest.raises(ValueError, match="two water"):
        two_power_water_g_rt(h, s, [.001], [.5])
    # The existing host-oxygen constraint is stricter than w<=0.5 everywhere.
    data = liquid.water_parameters()
    oxygen = magma.element_matrix(magma.DRY11_COMPONENTS)[magma.ELEMENTS.index("O")]
    ratio = oxygen*data["water_molar_mass_kg_mol"]/data["dry_molar_masses_kg_mol"]
    assert np.max(ratio/(1+ratio)) < .5
