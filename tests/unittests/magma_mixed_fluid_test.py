"""Binary fluid references, amount derivatives and liquid/fluid transfer."""

from functools import partial
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import IdealEOS, ZhangDuanEOS, state_tp
from exoeos.constants import MOLAR_GAS_CONSTANT as R
from exoeos.magma import element_matrix
from exoeos import magma_carbonate as liquid
from exoeos.magma_fluid import (FLUID_COMPONENTS, mixed_fluid_gibbs,
                               water_fluid_gibbs, water_gas_standard_gibbs,
                               co2_gas_standard_gibbs)


REFERENCE = json.loads((Path(__file__).resolve().parents[1]/"reference/carbonate_source_v1.json").read_text())
FIELDS = ("g_J", "s_J_K", "cp_J_K", "v_m3", "dvdt_m3_K", "dvdp_m3_Pa")
EOS = ZhangDuanEOS.from_species(FLUID_COMPONENTS)
FUN = partial(mixed_fluid_gibbs, eos=EOS)
G, MU = jax.jit(FUN), jax.jit(jax.grad(FUN, 2))


def properties(t, p, n):
    return jnp.array([FUN(t, p, n), -jax.grad(FUN, 0)(t, p, n),
                      -t*jax.grad(jax.grad(FUN, 0), 0)(t, p, n),
                      jax.grad(FUN, 1)(t, p, n),
                      jax.grad(jax.grad(FUN, 1), 0)(t, p, n),
                      jax.grad(jax.grad(FUN, 1), 1)(t, p, n)])


PROPERTIES = jax.jit(properties)


@pytest.mark.parametrize("row", REFERENCE["fluid_states"])
def test_binary_fluid_against_independent_zd09_pressure_integral(row):
    t, p, n = row["T_K"], row["P_Pa"], jnp.array(row["n_mol"])
    error = np.asarray(PROPERTIES(t, p, n))-[row[k] for k in FIELDS]
    assert np.all(abs(error) < [3e-7, 2e-7, 4e-5, 2e-13, 2e-14, 2e-17]), error
    if np.all(n > 0):
        np.testing.assert_allclose(MU(t, p, n), row["mu_J_mol"], rtol=0, atol=1e-4)


@pytest.mark.parametrize("eos", [IdealEOS(), EOS])
def test_joint_gibbs_derivatives_extensivity_and_fugacities(eos):
    fun = partial(mixed_fluid_gibbs, eos=eos)
    g, mu = jax.jit(fun), jax.jit(jax.grad(fun, 2))
    t, p, n = 1700., 5e7, jnp.array([.3, .7])
    expected = jnp.array([water_gas_standard_gibbs(t), co2_gas_standard_gibbs(t)])
    state = state_tp(eos, t, p, n/n.sum())
    expected += R*t*(jnp.log(n/n.sum())+jnp.log(p/1e5)+state.lnphi)
    np.testing.assert_allclose(mu(t, p, n), expected, rtol=3e-15)
    np.testing.assert_allclose(jax.grad(fun, 1)(t, p, n), n.sum()/state.rho, rtol=3e-12)
    np.testing.assert_allclose(n@mu(t, p, n), g(t, p, n), rtol=2e-15)
    for scale in (.01, 7.):
        np.testing.assert_allclose(g(t, p, scale*n), scale*g(t, p, n), rtol=2e-15)
        np.testing.assert_allclose(mu(t, p, scale*n), mu(t, p, n), rtol=2e-15)

    scaled = lambda z: fun(z[0]*1000, z[1]*1e8, z[2:])
    z = jnp.r_[t/1000, p/1e8, n]
    grad, hess = jax.jit(jax.grad(scaled)), jax.jit(jax.hessian(scaled))
    h = np.asarray(hess(z))
    np.testing.assert_allclose(h, h.T, rtol=2e-11, atol=1e-8)
    np.testing.assert_allclose(h[2:, 2:]@n, 0, rtol=0, atol=2e-10)
    for i in range(4):
        delta = jnp.eye(4)[i]*1e-4
        np.testing.assert_allclose(h[:, i], (grad(z+delta)-grad(z-delta))/2e-4, rtol=3e-6, atol=1e-3)


def test_pure_endpoints_zero_phase_and_batching():
    t, p = 1700., 5e7
    pure_water = partial(water_fluid_gibbs, eos=ZhangDuanEOS.from_species(("H2O",)))
    np.testing.assert_allclose(G(t, p, jnp.array([.4, 0.])), pure_water(t, p, .4), rtol=2e-15)
    co2_eos = ZhangDuanEOS.from_species(("CO2",))
    expected = .4*(co2_gas_standard_gibbs(t)+R*t*(jnp.log(p/1e5)+state_tp(co2_eos, t, p, jnp.ones(1)).gres_RT))
    np.testing.assert_allclose(G(t, p, jnp.array([0., .4])), expected, rtol=2e-15)
    np.testing.assert_array_equal(PROPERTIES(t, p, jnp.zeros(2)), np.zeros(6))
    np.testing.assert_array_equal(MU(t, p, jnp.zeros(2)), np.zeros(2))
    ns = jnp.array([[.3,.7],[0.,0.],[1.,0.],[0.,1.]])
    got = jax.jit(jax.vmap(FUN, in_axes=(None, None, 0)))(t, p, ns)
    np.testing.assert_allclose(got, [G(t, p, n) for n in ns], rtol=2e-15)
    for n in (jnp.array([1., 1e-12]), jnp.array([1e-12, 1.])):
        assert np.all(np.isfinite(MU(t, p, n)))
    with pytest.raises(ValueError, match="shape"):
        G(t, p, jnp.ones(1))
    with pytest.raises(ValueError, match="shape"):
        mixed_fluid_gibbs(t, p, jnp.ones(2), eos=co2_eos)
    with pytest.raises(ValueError, match="scalar"):
        G([t], p, jnp.ones(2))


def test_closed_water_and_carbon_transfer_uses_full_liquid_relaxation():
    t, p = 1700., 5e7
    nl = jnp.array([.5,.15,.05,.15,.1,.01,.01,.002,.02,.01,.002,.05,.01])
    nf = jnp.array([.03, .02])

    def total(xi):
        return liquid.liquid_gibbs(t, p, nl.at[-2:].add(-xi))+FUN(t, p, nf+xi)

    zero = jnp.zeros(2)
    derivative = jax.jit(jax.grad(total))(zero)
    expected = MU(t, p, nf)-jax.jit(jax.grad(liquid.liquid_gibbs, 2))(t, p, nl)[-2:]
    np.testing.assert_allclose(derivative, expected, rtol=3e-14, atol=2e-9)
    for i in range(2):
        dx = jnp.eye(2)[i]*1e-6
        finite = (jax.jit(total)(dx)-jax.jit(total)(-dx))/2e-6
        np.testing.assert_allclose(finite, derivative[i], rtol=2e-7, atol=5e-4)
    al, af = element_matrix(liquid.COMPONENTS), element_matrix(FLUID_COMPONENTS)
    xi = jnp.array([.01, .003])
    np.testing.assert_allclose(al@nl+af@nf, al@nl.at[-2:].add(-xi)+af@(nf+xi), rtol=2e-15)
