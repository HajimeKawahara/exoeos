"""Pure H2O standard alignment, EOS integration and joint Gibbs derivatives."""

from functools import partial
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import IdealEOS, PengRobinsonEOS, ZhangDuanEOS, state_tp
from exoeos.constants import MOLAR_GAS_CONSTANT as R
from exoeos.magma import HYDROUS_COMPONENTS, element_matrix, liquid_gibbs
from exoeos.magma_fluid import water_fluid_gibbs, water_gas_standard_gibbs


REFERENCE = json.loads((Path(__file__).resolve().parents[1]/"reference/magma_fluid_v1.json").read_text())
FIELDS = ("g_J_mol", "s_J_mol_K", "cp_J_mol_K", "v_m3_mol",
          "dvdt_m3_mol_K", "dvdp_m3_mol_Pa")
ZD = ZhangDuanEOS.from_species(("H2O",))
G = partial(water_fluid_gibbs, eos=ZD)


def properties(fun, t, p, n):
    return jnp.array([fun(t, p, n), -jax.grad(fun, 0)(t, p, n),
                      -t*jax.grad(jax.grad(fun, 0), 0)(t, p, n),
                      jax.grad(fun, 1)(t, p, n),
                      jax.grad(jax.grad(fun, 1), 0)(t, p, n),
                      jax.grad(jax.grad(fun, 1), 1)(t, p, n)])


PROPERTIES = jax.jit(partial(properties, G))
STANDARD = jax.jit(lambda t: jnp.array([
    water_gas_standard_gibbs(t), -jax.grad(water_gas_standard_gibbs)(t),
    -t*jax.grad(jax.grad(water_gas_standard_gibbs))(t)]))


@pytest.mark.parametrize("row", REFERENCE["standards"])
def test_ideal_standard_against_compiled_haar_dilute_limit(row):
    np.testing.assert_allclose(STANDARD(row["T_K"]), [row[k] for k in FIELDS[:3]],
                               rtol=0, atol=8e-8)
    # The finite-pressure Haar vapor contains a residual contribution.
    assert abs(float(STANDARD(row["T_K"])[0])-row["haar_real_1bar_J_mol"]) > .01


@pytest.mark.parametrize("row", REFERENCE["states"])
def test_full_fluid_against_independent_pressure_integral_and_thermal_reference(row):
    t, p = row["T_K"], row["P_Pa"]
    expected = np.array([row[k] for k in FIELDS])
    error = np.asarray(PROPERTIES(t, p, 1.))-expected
    # Independent five-point derivatives include truncation and roundoff.
    assert np.all(abs(error) < [2e-7, 1e-7, 2e-5, 1e-13, 1e-14, 1e-17]), error


@pytest.mark.parametrize("eos", [IdealEOS(), ZD, PengRobinsonEOS([647.096], [22.064e6], [.344])])
def test_eos_volume_potential_extensivity_and_joint_hessian(eos):
    fun = partial(water_fluid_gibbs, eos=eos)
    g, mu = jax.jit(fun), jax.jit(jax.grad(fun, 2))
    t, p, n = 1700., 5e7, .37
    value = g(t, p, n)
    np.testing.assert_allclose(n*mu(t, p, n), value, rtol=2e-15)
    np.testing.assert_allclose(g(t, p, 7*n), 7*value, rtol=2e-15)
    np.testing.assert_allclose(mu(t, p, 7*n), mu(t, p, n), rtol=2e-15)
    state = state_tp(eos, t, p, jnp.ones(1))
    np.testing.assert_allclose(jax.grad(fun, 1)(t, p, n), n/state.rho, rtol=2e-12)
    np.testing.assert_allclose(mu(t, p, n), water_gas_standard_gibbs(t)+R*t*(np.log(p/1e5)+state.lnphi[0]), rtol=2e-15)

    # Scale coordinates to compare all mixed second derivatives meaningfully.
    def scaled(z):
        return fun(z[0]*1000, z[1]*1e8, z[2])
    z = jnp.array([t/1000, p/1e8, n])
    gradient = jax.jit(jax.grad(scaled))
    hessian = np.asarray(jax.jit(jax.hessian(scaled))(z))
    np.testing.assert_allclose(hessian, hessian.T, rtol=2e-12, atol=1e-8)
    assert hessian[2, 2] == 0
    for i in range(3):
        delta = jnp.eye(3)[i]*1e-4
        finite = (gradient(z+delta)-gradient(z-delta))/(2e-4)
        np.testing.assert_allclose(hessian[:, i], finite, rtol=3e-6, atol=2e-4)
    batch = jax.jit(jax.vmap(g, in_axes=(0, None, 0)))(jnp.array([1200., t, 2200.]), p, jnp.array([n, 0., 2*n]))
    np.testing.assert_allclose(batch, [g(1200., p, n), 0., g(2200., p, 2*n)], rtol=2e-15)


def test_ideal_control_pressure_term_and_zero_phase_semantics():
    fun = partial(water_fluid_gibbs, eos=IdealEOS())
    t, p = 1700., 5e7
    expected = STANDARD(t)+jnp.array([R*t*np.log(p/1e5), -R*np.log(p/1e5), 0.])
    np.testing.assert_allclose(properties(fun, t, p, 1.)[:3], expected, rtol=1e-12)
    np.testing.assert_array_equal(PROPERTIES(t, p, 0.), np.zeros(6))
    # The guarded all-zero derivative is computational, not an insertion mu.
    assert jax.jit(jax.grad(G, 2))(t, p, 0.) == 0.
    assert jax.jit(jax.grad(G, 2))(t, p, 1.) < -1e5


@pytest.mark.parametrize("t", [1479., 1481., 1700.])
def test_liquid_fluid_transfer_derivative_and_conserved_water(t):
    dry = jnp.linspace(.02, .2, 11)
    water, fluid, p = .25, .1, 5e7

    def total(xi):
        return liquid_gibbs(t, p, jnp.r_[dry, water-xi])+G(t, p, fluid+xi)

    liquid_mu = jax.grad(lambda w: liquid_gibbs(t, p, jnp.r_[dry, w]))(water)
    fluid_mu = jax.jit(jax.grad(G, 2))(t, p, fluid)
    derivative = jax.jit(jax.grad(total))(0.)
    np.testing.assert_allclose(derivative, fluid_mu-liquid_mu, rtol=2e-14, atol=1e-9)
    finite = (jax.jit(total)(1e-5)-jax.jit(total)(-1e-5))/2e-5
    np.testing.assert_allclose(derivative, finite, rtol=2e-8, atol=1e-4)
    a = element_matrix(HYDROUS_COMPONENTS)
    def atoms(xi):
        return a@jnp.r_[dry, water-xi] + a[:, -1]*(fluid+xi)
    np.testing.assert_allclose(atoms(.05), atoms(0.), rtol=2e-15)
    # Supplied dry face and pure-water component endpoint have finite G.
    # Only positive water is differentiated; no phase-boundary root is solved.
    assert np.isfinite(liquid_gibbs(t, p, jnp.r_[dry, 0.]))
    assert np.isfinite(liquid_gibbs(t, p, jnp.r_[jnp.zeros(11), 1.]))


def test_shapes_and_pure_eos_component_count():
    assert water_gas_standard_gibbs(1700) == water_gas_standard_gibbs(1700.)
    # Explicit float32 inputs under x64 must not break the zero-amount cond.
    assert np.isfinite(G(jnp.float32(1700), jnp.float32(1e5), jnp.float32(1)))
    with pytest.raises(ValueError, match="scalar"):
        water_gas_standard_gibbs([1700.])
    for t, p, n in (([1700.], 1e5, 1.), (1700., [1e5], 1.), (1700., 1e5, [1.])):
        with pytest.raises(ValueError, match="scalar"):
            G(t, p, n)
    with pytest.raises(ValueError, match="shape"):
        water_fluid_gibbs(1700., 1e5, 1., eos=ZhangDuanEOS.from_species(("H2O", "CO2")))
