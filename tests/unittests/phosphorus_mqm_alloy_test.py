"""Conservation, limiting thermodynamics and common-reference checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from exoeos import associated_metal, phosphorus_association
from exoeos import phosphorus_mqm as binary
from exoeos import phosphorus_mqm_alloy as alloy
from exoeos import phosphorus_phosphides as solid
from exoeos._conditional_data import R

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("model", binary.MODELS)
def test_binary_recovery_and_no_phosphorus_parent_limit(model):
    t, p = 2173.15, 5e7
    eps, matrix = associated_metal.interaction_parameters(t)
    standard = jnp.linspace(-10., 4., 20)
    pairs = jnp.array([2., .2, .5])
    n = jnp.zeros(21).at[jnp.array(alloy.PAIR_INDICES)].set(pairs)
    g = lambda n: alloy.gibbs_j(t, p, n, standard, eps, matrix,
                               model=model, delta_v_p_m3_mol=0.)
    np.testing.assert_allclose(g(n), binary.gibbs_j(t, pairs, standard[jnp.array([0, 4])], model=model))
    n = jnp.ones(21)*.0001
    n = n.at[0].set(2.).at[4].set(0.).at[20].set(0.)
    collapsed = alloy.collapse_matrix(model=model) @ n
    old = phosphorus_association.gibbs_j(t, p, collapsed, standard, eps, matrix, model="quadratic")
    np.testing.assert_allclose(g(n), old, rtol=2e-14)
    np.testing.assert_allclose(g(jnp.zeros(21)), 0.)


def test_extensivity_atomic_gauge_pressure_and_derivatives():
    t, p = 2173.15, 5e7
    eps, matrix = associated_metal.interaction_parameters(t)
    n = jnp.linspace(.001, .004, 21).at[0].set(1.8)
    s = jnp.linspace(-10., 2., 20)
    c = alloy.collapse_matrix(model="you2020")
    def g(n, p=p, s=s, v=2e-6):
        return alloy.gibbs_j(t, p, n, s, eps, matrix, model="you2020", delta_v_p_m3_mol=v)
    mu = jax.grad(g)(n)
    np.testing.assert_allclose(mu @ n, g(n), rtol=2e-14)
    np.testing.assert_allclose(g(n*7), 7*g(n), rtol=2e-14)
    h = jax.hessian(g)(n)
    np.testing.assert_allclose(h, h.T, atol=2e-7)
    for i in alloy.PAIR_INDICES:
        step = np.eye(21)[i]*1e-7
        np.testing.assert_allclose(mu[i], (g(n+step)-g(n-step))/2e-7, rtol=2e-7)
    shift = jnp.linspace(-.2, .3, 20)
    np.testing.assert_allclose(g(n,s=s+shift)-g(n), R*t*((c @ n) @ shift), atol=1e-10)
    # The pressure-dependent host cancels between the two controls.
    delta = lambda pressure: g(n, p=pressure)-g(n, p=pressure, v=0.)
    np.testing.assert_allclose(jax.grad(delta)(p), (c @ n)[4]*2e-6, rtol=1e-13)
    np.testing.assert_allclose(delta(1e5), 0., atol=1e-12)


def test_one_bar_reference_and_solid_gauge_are_consistent():
    t, p = 2173.15, 4e6
    fe, p2 = -12., -28.
    vp = jnp.zeros(4)
    g = lambda fe, p2: solid.common_gibbs_j(t, p, fe, p2,
        delta_volumes_m3_mol_formula=vp, high_temperature_continuation=True)
    a = np.array(solid.ATOM_MATRIX)
    shift = np.array([.4, -.3])
    np.testing.assert_allclose(g(fe+shift[0],p2+2*shift[1])-g(fe,p2), R*t*(shift @ a), atol=1e-9)
    pure_p = binary.pure_p_standard_rt(t,p2)
    # Same reaction residual in source SER and native gauges.
    reaction = g(fe,p2)-R*t*(np.array([fe,pure_p]) @ a)
    expected = solid.standard_gibbs_j(t,high_temperature_continuation=True)-np.array(
        [binary.sgte_liquid_fe_j(t),binary.sgte_liquid_p_j(t)]) @ a
    np.testing.assert_allclose(reaction, expected, atol=1e-9)
    np.testing.assert_allclose(binary.pure_p_standard_rt(t,p2+.6)-pure_p,.3,atol=1e-14)


@pytest.mark.parametrize("t", [900., 1800., 2173.15])
def test_phosphide_heat_capacity_integration(t):
    # Independent numerical quadrature of source Cp, without provider helpers.
    rows = [(1,-157643.,83.12,lambda x:71.555+.02375*x-370000/x**2),
            (2,-126100.,47.77,lambda x:43.7878+.01985*x-232000/x**2),
            (3,-191100.,51.05,lambda x:77.52563+.009348*x-443846/x**2-1.1e-6*x*x)]
    g = lambda t: solid.standard_gibbs_j(t,high_temperature_continuation=True)
    for i,hf,s,cp in rows:
        expected = hf+quad(cp,298.15,t)[0]-t*(s+quad(lambda x:cp(x)/x,298.15,t)[0])
        np.testing.assert_allclose(g(t)[i],expected,atol=2e-8)
        np.testing.assert_allclose(-t*jax.hessian(lambda v:g(v)[i])(t),cp(t),atol=1e-10)
    cp3 = lambda x: 106.76+.0108*x-780000/x**2 if x < 727 else 109.84+.0287*x-8000000/x**2
    h = quad(cp3,298.15,t,points=[727.])[0]
    s = quad(lambda x:cp3(x)/x,298.15,t,points=[727.])[0]
    tau, frac = t/713., .28
    d = 518/1125+11692/15975*(1/frac-1)
    magnetic = -4*R*t*np.log(1.6)*(tau**-5/10+tau**-15/315+tau**-25/1500)/d
    expected = -164830+h-t*(110.15+s)+1000*(1-t/698)+magnetic
    np.testing.assert_allclose(g(t)[0],expected,atol=2e-8)


def test_explicit_continuation_and_input_guards():
    assert np.all(np.isnan(solid.standard_gibbs_j(2173.15)))
    assert np.all(np.isfinite(solid.standard_gibbs_j(2173.15,high_temperature_continuation=True)))
    assert np.all(np.isnan(solid.standard_gibbs_j(3001.,high_temperature_continuation=True)))
    assert np.isnan(binary.nist_p2_gas_j(297.))
    assert np.isnan(binary.sgte_liquid_fe_j(6001.))
    with pytest.raises(ValueError):
        solid.common_gibbs_j(2173.,1e5,0.,0.,delta_volumes_m3_mol_formula=[0.])
    with pytest.raises(ValueError):
        alloy.gibbs_j(2173.,1e5,jnp.ones(20),jnp.zeros(20),jnp.zeros(4),jnp.zeros((18,18)),
                      model="you2020",delta_v_p_m3_mol=0.)
