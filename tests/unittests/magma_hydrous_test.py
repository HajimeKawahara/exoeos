"""Hydrous MELTS standards, mixtures, dry limits and JAX derivatives."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos.magma import (
    DRY11_COMPONENTS, HYDROUS_COMPONENTS, HYDROUS_OXIDES, ELEMENTS, R,
    element_matrix, liquid_gibbs, liquid_standard_gibbs,
    liquid_moles_from_oxides, oxide_moles_from_liquid,
)
from exoeos._magma_water import _haar_density, _haar_helmholtz


ROOT = Path(__file__).resolve().parents[1]/"reference"
SOURCE = json.loads((ROOT/"hydrous_source_v1.json").read_text())
MELTS = json.loads((ROOT/"hydrous_melts_v1.json").read_text())
FIELDS = ("g_J", "s_J_K", "cp_J_K", "v_m3", "dvdt_m3_K", "dvdp_m3_Pa")
G = jax.jit(liquid_gibbs)
MU = jax.jit(jax.grad(liquid_gibbs, 2))
STANDARD = jax.jit(lambda t, p: liquid_standard_gibbs(t, p, include_h2o=True))


def properties(fun, t, p, *args):
    return jnp.array([fun(t, p, *args), -jax.jacfwd(fun, 0)(t, p, *args),
                      -t*jax.jacfwd(jax.jacfwd(fun, 0), 0)(t, p, *args),
                      jax.jacfwd(fun, 1)(t, p, *args),
                      jax.jacfwd(jax.jacfwd(fun, 1), 0)(t, p, *args),
                      jax.jacfwd(jax.jacfwd(fun, 1), 1)(t, p, *args)])


STANDARD_PROPERTIES = jax.jit(lambda t, p: properties(STANDARD, t, p))
LIQUID_PROPERTIES = jax.jit(lambda t, p, n: properties(liquid_gibbs, t, p, n))


@pytest.mark.parametrize("row", SOURCE["states"])
def test_water_standard_against_compiled_upstream_source(row):
    actual = np.asarray(STANDARD_PROPERTIES(row["T_K"], row["P_Pa"]))[:, -1]
    expected = np.array([row[key] for key in FIELDS])
    # Haar's high-order ideal polynomial loses digits by cancellation at
    # 2500 K; allow ~1e-10 relative Cp rounding across JAX/XLA versions.
    assert np.all(abs(actual-expected) < np.array([8e-8, 3e-10, 5e-9, 1e-18, 1e-21, 1e-26])), actual-expected


@pytest.mark.parametrize("row", MELTS["states"])
def test_hydrous_liquid_against_independent_native_properties(row):
    t, p, n = row["T_K"], row["P_Pa"], np.array(row["n_mol"])
    actual = np.asarray(LIQUID_PROPERTIES(t, p, n))
    expected = np.array([row[key] for key in FIELDS])
    tolerance = np.array([5e-5, 1e-8, 1e-8, 1e-14, 2e-19, 2e-24])*max(1., n.sum())
    assert np.all(abs(actual-expected) < tolerance), actual-expected
    present = n > 0
    np.testing.assert_allclose(np.asarray(STANDARD(t, p))[present],
                               np.array(row["standard_J_mol"], float)[present], rtol=0, atol=8e-8)
    if np.all(present):
        np.testing.assert_allclose(MU(t, p, n), row["mu_J_mol"], rtol=0, atol=2e-5)
    np.testing.assert_allclose(row["returned_component_fractions"], n/n.sum(), rtol=0, atol=1e-10)


def test_all_eleven_water_pairs_and_extra_entropy():
    assert {pair["component"] for pair in SOURCE["pairs"]} == set(DRY11_COMPONENTS)
    for pair in SOURCE["pairs"]:
        n = np.zeros(12)
        n[DRY11_COMPONENTS.index(pair["component"])], n[-1] = .7, .3
        for t, p in ((1200., 1e5), (1700., 5e7), (2200., 5e8)):
            # On a dry-component/water edge, MELTS has twice the usual
            # binary ideal mixing entropy, not just one ideal term.
            expected = .21*(pair["H_J_mol"]-t*pair["S_J_mol_K"]+(p/1e5-1)*pair["V_J_mol_bar"])
            expected += 2*R*t*(.7*np.log(.7)+.3*np.log(.3))
            np.testing.assert_allclose(G(t, p, n)-n@STANDARD(t, p), expected, rtol=0, atol=3e-9)


def test_dry_face_pure_water_and_absent_phase_have_continuous_energies():
    dry = jnp.linspace(.02, .2, 11)
    wet = jnp.r_[dry, 0.]
    for t, p in ((700., 1e5), (1200., 5e7), (2200., 5e8)):
        np.testing.assert_allclose(LIQUID_PROPERTIES(t, p, wet), LIQUID_PROPERTIES(t, p, dry), rtol=2e-14, atol=1e-20)
        np.testing.assert_allclose(STANDARD(t, p)[:11], liquid_standard_gibbs(t, p, include_dry11=True), rtol=2e-15)
        np.testing.assert_allclose(LIQUID_PROPERTIES(t, p, jnp.eye(12)[-1]), STANDARD_PROPERTIES(t, p)[:, -1], rtol=2e-14, atol=1e-20)
        np.testing.assert_array_equal(LIQUID_PROPERTIES(t, p, jnp.zeros(12)), np.zeros(6))
    old = MU(1700., 5e7, dry)
    errors = [float(max(abs(MU(1700., 5e7, wet.at[-1].set(eps))[:11]-old))) for eps in (1e-4, 1e-7, 1e-10)]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-4
    mu1 = MU(1700., 5e7, wet.at[-1].set(1e-8))[-1]
    mu2 = MU(1700., 5e7, wet.at[-1].set(1e-10))[-1]
    np.testing.assert_allclose(mu2-mu1, 2*R*1700*np.log(.01), rtol=0, atol=1e-3)


def test_hydrous_inventory_conversion_and_amount_derivatives():
    oxide = jnp.array([2., 1., .4, .5, .3, .1, .07, .02, .08, .04, .03, .25])
    n = jax.jit(liquid_moles_from_oxides)(oxide)
    np.testing.assert_allclose(oxide_moles_from_liquid(n), oxide, rtol=0, atol=4e-16)
    a, ao = element_matrix(HYDROUS_COMPONENTS), element_matrix(HYDROUS_OXIDES)
    expected = [7.82, 1., 2., .6, .6, .5, .16, .08, .07, .04, .06, .5]
    assert ELEMENTS[-1] == "H"
    np.testing.assert_allclose(a@n, expected, rtol=0, atol=2e-15)
    np.testing.assert_allclose(ao@oxide, expected, rtol=0, atol=2e-15)
    b = jax.jacfwd(liquid_moles_from_oxides)(oxide)
    np.testing.assert_array_equal(a@b, ao)
    np.testing.assert_array_equal(jax.jacfwd(oxide_moles_from_liquid)(n)@b, np.eye(12))
    np.testing.assert_allclose(jax.grad(lambda m: G(1700., 5e7, liquid_moles_from_oxides(m)))(oxide), b.T@MU(1700., 5e7, n), rtol=2e-15)


def test_implicit_haar_root_and_joint_jax_derivatives():
    for t in (700., 1000., 1700., 2500.):
        rho = _haar_density(t)
        np.testing.assert_allclose(rho*rho*jax.grad(_haar_helmholtz, 1)(t, rho), 1., rtol=0, atol=2e-14)
    t, p, n = 1700., 5e7, jnp.linspace(.03, .25, 12)
    mu, g = MU(t, p, n), G(t, p, n)
    np.testing.assert_allclose(n@mu, g, rtol=2e-15)
    np.testing.assert_allclose(G(t, p, 7*n), 7*g, rtol=2e-15)
    h = jax.jit(jax.hessian(liquid_gibbs, 2))(t, p, n)
    np.testing.assert_allclose(h, h.T, rtol=0, atol=2e-9)
    np.testing.assert_allclose(h@n, 0., rtol=0, atol=2e-10)
    z = jnp.r_[t, p, n]
    def fun(z):
        return liquid_gibbs(z[0], z[1], z[2:])
    gradient, hessian = jax.jit(jax.grad(fun)), jax.jit(jax.hessian(fun))
    joint = hessian(z)
    np.testing.assert_allclose(joint, joint.T, rtol=2e-12, atol=2e-9)
    for index, step in ((0, .02), (1, 1e3), (13, 1e-6)):
        delta = jnp.eye(14)[index]*step
        finite = (gradient(z+delta)-gradient(z-delta))/(2*step)
        np.testing.assert_allclose(joint[:, index], finite, rtol=3e-6, atol=5e-4)
    batched = jax.jit(jax.vmap(liquid_gibbs, in_axes=(0, None, None)))(jnp.array([1200., t, 2200.]), p, n)
    np.testing.assert_allclose(batched, [G(ti, p, n) for ti in (1200., t, 2200.)], rtol=2e-15)


def test_reference_inventory_and_boundary_accounting():
    assert MELTS["component_order"] == list(HYDROUS_COMPONENTS)
    assert MELTS["oxide_order"] == list(HYDROUS_OXIDES)
    names = {row["name"] for row in MELTS["states"]}
    names.update(row["samples"][0]["name"] for row in MELTS["unavailable_cases"])
    assert {f"water_pair_{i}" for i in range(11)} | {"pure_water"} <= names
    assert {r["water_wt_percent"] for r in MELTS["recipes"]} == {.1, 2., 5.}
