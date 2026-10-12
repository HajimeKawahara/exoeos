"""Independent dry-liquid standards, all 55 pairs, atom ledgers and faces."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos.magma import (
    COMPONENTS, TERNARY_COMPONENTS, CMAS_COMPONENTS, CMFAS_COMPONENTS,
    FERRIC_COMPONENTS, DRY11_COMPONENTS, DRY11_OXIDES, ELEMENTS, R,
    element_matrix, liquid_gibbs, liquid_standard_gibbs,
    liquid_moles_from_oxides, oxide_moles_from_liquid,
)


REFERENCE = Path(__file__).resolve().parents[1]/"reference"
SOURCE = json.loads((REFERENCE/"dry11_source_v1.json").read_text())
MELTS = json.loads((REFERENCE/"dry11_melts_v1.json").read_text())
STANDARD = jax.jit(lambda t, p: liquid_standard_gibbs(t, p, include_dry11=True))
MU = jax.jit(jax.grad(liquid_gibbs, 2))
G = jax.jit(liquid_gibbs)
FIELDS = ("g_J", "s_J_K", "cp_J_K", "v_m3", "dvdt_m3_K", "dvdp_m3_Pa")


def properties(fun, t, p, *args):
    return jnp.array([fun(t, p, *args), -jax.jacfwd(fun, 0)(t, p, *args),
                      -t*jax.jacfwd(jax.jacfwd(fun, 0), 0)(t, p, *args),
                      jax.jacfwd(fun, 1)(t, p, *args),
                      jax.jacfwd(jax.jacfwd(fun, 1), 0)(t, p, *args),
                      jax.jacfwd(jax.jacfwd(fun, 1), 1)(t, p, *args)])


STANDARD_PROPERTIES = jax.jit(lambda t, p: properties(STANDARD, t, p))
LIQUID_PROPERTIES = jax.jit(lambda t, p, n: properties(liquid_gibbs, t, p, n))


@pytest.mark.parametrize("row", SOURCE["states"])
def test_new_standards_against_independent_source_quadrature(row):
    expected = [row[k] for k in ("g_J_mol", "s_J_mol_K", "cp_J_mol_K",
                                "v_m3_mol", "dvdt_m3_mol_K", "dvdp_m3_mol_Pa")]
    actual = STANDARD_PROPERTIES(row["T_K"], row["P_Pa"])[:, DRY11_COMPONENTS.index(row["component"])]
    np.testing.assert_allclose(actual, expected, rtol=4e-12, atol=1e-20)


@pytest.mark.parametrize("row", MELTS["states"])
def test_supplied_native_liquid_properties(row):
    t, p, n = row["T_K"], row["P_Pa"], np.array(row["n_mol"])
    actual = np.asarray(LIQUID_PROPERTIES(t, p, n))
    expected = np.array([row[k] for k in FIELDS])
    tolerance = np.array([5e-5, 1e-8, 1e-8, 1e-14, 2e-19, 2e-24])
    # Native oxide round trips perturb the fractions by ~1e-11. Extensive
    # errors scale with amount; the sevenfold recipe must use the same bound.
    tolerance *= max(1., n.sum())
    assert np.all(abs(actual-expected) <= tolerance), actual-expected
    present = n > 0
    np.testing.assert_allclose(np.asarray(STANDARD(t, p))[present],
                               np.array(row["standard_J_mol"], float)[present], rtol=0, atol=3e-8)
    if np.all(present):
        np.testing.assert_allclose(MU(t, p, n), row["mu_J_mol"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(row["returned_component_fractions"], n/n.sum(), rtol=0, atol=1e-10)


def test_fixture_covers_all_pairs_and_non_bse_compositions():
    pairs = {frozenset(row["components"]) for row in SOURCE["pairs"]}
    assert pairs == {frozenset((a, b)) for i, a in enumerate(DRY11_COMPONENTS) for b in DRY11_COMPONENTS[i+1:]}
    assert len(SOURCE["pairs"]) == 55
    assert MELTS["component_order"] == list(DRY11_COMPONENTS)
    assert {r["name"] for r in MELTS["recipes"]} == {"BSE", "basalt_like", "rhyolite_like", "alkali_rich"}
    # Every attempted native boundary must be saved or explicitly unavailable.
    edges = {r["name"] for r in MELTS["states"] if r["name"].startswith("edge_")}
    edges.update(r["samples"][0]["name"] for r in MELTS["unavailable_cases"])
    assert edges == {f"edge_{i}_{j}" for i in range(11) for j in range(i, 11)}


@pytest.mark.parametrize("row", SOURCE["pairs"])
def test_each_binary_isolates_its_published_interaction(row):
    indices = [DRY11_COMPONENTS.index(name) for name in row["components"]]
    n = np.zeros(11)
    n[indices] = [.7, .3]
    for t, p in ((1200., 1e5), (1700., 5e7), (2200., 5e8)):
        ideal = R*t*np.dot(n[indices], np.log(n[indices]))
        excess = G(t, p, n)-n@STANDARD(t, p)-ideal
        w = row["H_J_mol"]-t*row["S_J_mol_K"]+(p/1e5-1)*row["V_J_mol_bar"]
        np.testing.assert_allclose(excess, .21*w, rtol=0, atol=2e-9)


@pytest.mark.parametrize("basis", [COMPONENTS, TERNARY_COMPONENTS, CMAS_COMPONENTS, CMFAS_COMPONENTS, FERRIC_COMPONENTS])
def test_lower_dimensional_faces_and_dilution_limits(basis):
    indices = np.array([DRY11_COMPONENTS.index(name) for name in basis])
    active = np.zeros(11, dtype=bool)
    active[indices] = True
    n = jnp.linspace(.07, .43, len(basis))
    full = np.zeros(11)
    full[indices] = n
    for t, p in ((1200., 1e5), (1600., 5e7), (2200., 5e8)):
        np.testing.assert_allclose(LIQUID_PROPERTIES(t, p, full), LIQUID_PROPERTIES(t, p, n),
                                   rtol=4e-15, atol=1e-20)
        old_mu = MU(t, p, n)
        errors = []
        for epsilon in (1e-4, 1e-7, 1e-10):
            errors.append(max(abs(np.asarray(MU(t, p, full+epsilon*(~active)))[indices]-old_mu)))
        assert errors[2] < errors[1] < errors[0]
        assert errors[2] < 5e-4


def test_basis_transform_conserves_atoms_mass_and_composition_derivatives():
    # An independently chosen oxide inventory activates all eleven components.
    oxide = jnp.array([2., 1., .4, .5, .3, .1, .07, .02, .08, .04, .03])
    expected = jnp.array([.74, .49, .2, .41, .26, .1, .07, .02, .08, .08, .03])
    n = jax.jit(liquid_moles_from_oxides)(oxide)
    np.testing.assert_allclose(n, expected, rtol=0, atol=3e-16)
    np.testing.assert_allclose(oxide_moles_from_liquid(n), oxide, rtol=0, atol=3e-16)
    a, ao = element_matrix(DRY11_COMPONENTS), element_matrix(DRY11_OXIDES)
    # O, Mg, Si, Fe, Al, Ca, Na, K, Ti, Cr, P, from the oxide formulas.
    atoms = [7.57, 1., 2., .6, .6, .5, .16, .08, .07, .04, .06]
    np.testing.assert_allclose(a@n, atoms, rtol=0, atol=2e-15)
    np.testing.assert_allclose(ao@oxide, atoms, rtol=0, atol=2e-15)
    b = jax.jacfwd(liquid_moles_from_oxides)(oxide)
    c = jax.jacfwd(oxide_moles_from_liquid)(n)
    np.testing.assert_array_equal(a@b, ao)
    np.testing.assert_array_equal(c@b, np.eye(11))
    atomic_g = jnp.array([15.9994, 24.305, 28.0855, 55.845, 26.9815386, 40.078,
                          22.98976928, 39.0983, 47.867, 51.9961, 30.973762])
    np.testing.assert_allclose(atomic_g@a@n, atomic_g@ao@oxide, rtol=2e-16)
    def fun(m):
        return liquid_gibbs(1700., 5e7, liquid_moles_from_oxides(m))
    np.testing.assert_allclose(jax.grad(fun)(oxide), b.T@MU(1700., 5e7, n), rtol=2e-15)
    np.testing.assert_allclose(jax.hessian(fun)(oxide), b.T@jax.hessian(liquid_gibbs, 2)(1700., 5e7, n)@b,
                               rtol=2e-13, atol=1e-9)
    assert a.shape == (len(ELEMENTS), 11)


def test_recipe_roundtrips_match_independent_native_conversion():
    for recipe in MELTS["recipes"]:
        np.testing.assert_allclose(liquid_moles_from_oxides(recipe["oxide_mol"]), recipe["n_mol"], rtol=2e-15, atol=1e-16)
        np.testing.assert_allclose(oxide_moles_from_liquid(recipe["n_mol"]), recipe["oxide_mol"], rtol=2e-15)


def test_infeasible_oxide_residuals_are_not_hidden():
    for oxide_index, component_index in ((1, 0), (7, 1), (9, 4), (10, 3)):
        oxide = jnp.eye(11)[oxide_index]
        n = liquid_moles_from_oxides(oxide)
        assert n[component_index] < 0
        np.testing.assert_array_equal(oxide_moles_from_liquid(n), oxide)


def test_extensivity_joint_derivatives_jit_vmap_and_endpoints():
    t, p, n = 1700., 5e7, jnp.linspace(.04, .24, 11)
    g, mu = liquid_gibbs(t, p, n), MU(t, p, n)
    np.testing.assert_allclose(n@mu, g, rtol=2e-15)
    np.testing.assert_allclose(liquid_gibbs(t, p, 7*n), 7*g, rtol=2e-15)
    h = jax.jit(jax.hessian(liquid_gibbs, 2))(t, p, n)
    np.testing.assert_allclose(h, h.T, rtol=0, atol=2e-9)
    np.testing.assert_allclose(h@n, 0., rtol=0, atol=2e-10)
    joint = jnp.r_[t, p, n]
    def fun(z):
        return liquid_gibbs(z[0], z[1], z[2:])
    hj = jax.jit(jax.hessian(fun))(joint)
    assert np.all(np.isfinite(hj))
    np.testing.assert_allclose(hj, hj.T, rtol=1e-12, atol=2e-9)
    for i, step in ((0, .01), (1, 1e3), (8, 1e-5), (11, 1e-5)):
        delta = jnp.eye(13)[i]*step
        fd = (jax.grad(fun)(joint+delta)-jax.grad(fun)(joint-delta))/(2*step)
        np.testing.assert_allclose(hj[:, i], fd, rtol=3e-7, atol=2e-3)
    batch = jax.jit(jax.vmap(liquid_gibbs, in_axes=(0, None, None)))(jnp.array([1600., t, 1800.]), p, n)
    np.testing.assert_allclose(batch[1], g, rtol=2e-15)
    np.testing.assert_array_equal(LIQUID_PROPERTIES(t, p, jnp.zeros(11)), np.zeros(6))
    pure = jax.jit(jax.vmap(liquid_gibbs, in_axes=(None, None, 0)))(t, p, jnp.eye(11))
    np.testing.assert_allclose(pure, STANDARD(t, p), rtol=0, atol=1e-9)


@pytest.mark.parametrize("size", [0, 7, 10, 12])
def test_static_shape_validation(size):
    with pytest.raises(ValueError, match="shape"):
        liquid_gibbs(1700., 1e5, jnp.ones(size))
    for function in (liquid_moles_from_oxides, oxide_moles_from_liquid):
        with pytest.raises(ValueError, match="shape"):
            function(jnp.ones(size))
