"""Mode-4 reference properties, conserved internal reaction and relaxed AD."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import magma, magma_carbonate as model
from exoeos.magma_fluid import co2_gas_standard_gibbs


REFERENCE = Path(__file__).resolve().parents[1]/"reference"
SOURCE = json.loads((REFERENCE/"carbonate_source_v1.json").read_text())
MELTS = json.loads((REFERENCE/"carbonate_melts_v1.json").read_text())
FIELDS = ("g_J", "s_J_K", "cp_J_K", "v_m3", "dvdt_m3_K", "dvdp_m3_Pa")
N = jnp.array([.5,.15,.05,.15,.1,.01,.01,.002,.02,.01,.002,.05,.001])
G = jax.jit(model.liquid_gibbs)
MU = jax.jit(jax.grad(model.liquid_gibbs, 2))
SPECIES = jax.jit(model.liquid_species_moles)


def properties(fun, t, p, n):
    return jnp.array([fun(t, p, n), -jax.grad(fun, 0)(t, p, n),
                      -t*jax.grad(jax.grad(fun, 0), 0)(t, p, n),
                      jax.grad(fun, 1)(t, p, n),
                      jax.grad(jax.grad(fun, 1), 0)(t, p, n),
                      jax.grad(jax.grad(fun, 1), 1)(t, p, n)])


PROPERTIES = jax.jit(lambda t, p, n: properties(model.liquid_gibbs, t, p, n))
STANDARDS = jax.jit(lambda t, p: model.liquid_standard_gibbs(t, p))
STANDARD_PROPERTIES = jax.jit(lambda t, p, n: properties(lambda t, p, n: model.liquid_standard_gibbs(t, p)@n, t, p, n))


def test_all_species_interactions_against_pinned_header():
    expected = np.zeros((14, 14))
    for row in SOURCE["pairs"]:
        i, j = [model.SPECIES.index(c) for c in row["components"]]
        expected[i, j] = expected[j, i] = row["H_J_mol"]
        assert row["S_J_mol_K"] == row["V_J_mol_bar"] == 0
    np.testing.assert_array_equal(model._interaction_matrix(), expected)
    nu = np.asarray(model._REACTION)
    np.testing.assert_allclose(nu@expected@nu, 113246.6, rtol=2e-15)
    np.testing.assert_array_equal(magma.element_matrix(model.SPECIES)@nu, np.zeros(len(magma.ELEMENTS)))


@pytest.mark.parametrize("row", SOURCE["standards"])
def test_standards_against_directly_compiled_upstream(row):
    t, p = row["T_K"], row["P_Pa"]
    if row["component"] == "CO2_gas":
        f = co2_gas_standard_gibbs
        got = jnp.array([f(t), -jax.grad(f)(t), -t*jax.grad(jax.grad(f))(t)])
        np.testing.assert_allclose(got, [row[k] for k in FIELDS[:3]], rtol=0, atol=1e-8)
    else:
        n = jnp.eye(13)[model.COMPONENTS.index(row["component"])]
        error = np.asarray(STANDARD_PROPERTIES(t, p, n))-[row[k] for k in FIELDS]
        assert np.all(abs(error) < [1e-7, 2e-9, 2e-8, 1e-18, 1e-22, 1e-28]), error


@pytest.mark.parametrize("row", MELTS["states"])
def test_relaxed_liquids_against_pinned_mode4(row):
    t, p, n = row["T_K"], row["P_Pa"], jnp.array(row["n_mol"])
    error = np.asarray(PROPERTIES(t, p, n))-[row[k] for k in FIELDS]
    assert np.all(abs(error) < [1e-4, 2e-7, 2e-7, 3e-13, 2e-16, 2e-21]), error
    np.testing.assert_allclose(SPECIES(t, p, n), row["species_mol"], rtol=0, atol=2e-10)
    present = np.asarray(n) > 0
    np.testing.assert_allclose(STANDARDS(t, p)[present], np.array(row["standard_J_mol"], float)[present], rtol=0, atol=3e-8)
    # Differentiate only the live coordinates on a fixed composition face.
    indices = np.flatnonzero(present)
    live = lambda m: model.liquid_gibbs(t, p, jnp.zeros(13).at[indices].set(m))
    mu = jax.jit(jax.grad(live))(n[present]) if not np.all(present) else MU(t, p, n)
    if row["name"] == "trace_carbon":
        # The pinned runtime leaves carbonate at zero at 1e-9 mol C. Its
        # trace-carbon potential is retained as a diagnostic, not a target.
        assert row["species_mol"][-1] == 0
        assert SPECIES(t, p, n)[-1] > 0
        assert abs(mu[-1]-row["mu_J_mol"][-1]) > 100
        np.testing.assert_allclose(mu[:-1], np.array(row["mu_J_mol"][:-1]), rtol=0, atol=1e-4)
    else:
        np.testing.assert_allclose(mu, np.array(row["mu_J_mol"], float)[present], rtol=0, atol=1e-4)


def test_amount_basis_and_conserved_internal_species():
    oxide = jnp.array([2.,1.,.4,.5,.3,.1,.07,.02,.08,.04,.03,.25,.02])
    n = jax.jit(model.liquid_moles_from_oxides)(oxide)
    np.testing.assert_allclose(model.oxide_moles_from_liquid(n), oxide, rtol=0, atol=4e-16)
    a, ao = magma.element_matrix(model.COMPONENTS), magma.element_matrix(model.OXIDES)
    b = jax.jacfwd(model.liquid_moles_from_oxides)(oxide)
    np.testing.assert_array_equal(a@b, ao)
    expected = {"O": 7.86, "Mg": 1., "Si": 2., "Fe": .6, "Al": .6,
                "Ca": .5, "Na": .16, "K": .08, "Ti": .07, "Cr": .04, "P": .06, "H": .5, "C": .02}
    np.testing.assert_allclose(a@n, [expected[e] for e in magma.ELEMENTS], rtol=0, atol=3e-15)
    y = SPECIES(1700., 5e7, n)
    np.testing.assert_allclose(magma.element_matrix(model.SPECIES)@y, a@n, rtol=0, atol=2e-15)
    np.testing.assert_allclose(y.sum(), n.sum(), rtol=2e-15)
    np.testing.assert_allclose(jax.grad(lambda m: G(1700., 5e7, model.liquid_moles_from_oxides(m)))(oxide), b.T@MU(1700., 5e7, n), rtol=2e-15)


@pytest.mark.parametrize("t", [1200., 1479., 1481., 2200.])
def test_relaxed_joint_hessian_and_schur_correction(t):
    # Scaled T/P permit comparisons of thermal, pressure and composition blocks.
    def energy(z):
        return model.liquid_gibbs(1000*z[0], 1e8*z[1], z[2:])

    z = jnp.r_[t/1000, .5, N]
    gradient, hessian = jax.jit(jax.grad(energy)), jax.jit(jax.hessian(energy))
    h = np.asarray(hessian(z))
    np.testing.assert_allclose(h, h.T, rtol=2e-11, atol=3e-8)
    for i in (0, 1, 2, 5, 13, 14):
        step = 2e-4*float(z[i])
        delta = jnp.eye(15)[i]*step
        fd = (gradient(z-2*delta)-8*gradient(z-delta)+8*gradient(z+delta)-gradient(z+2*delta))/(12*step)
        np.testing.assert_allclose(h[:, i], fd, rtol=2e-6, atol=3e-4)
    np.testing.assert_allclose(h[2:, 2:]@N, 0, rtol=0, atol=2e-9)

    # Explicitly eliminate the reaction coordinate from the unrelaxed Hessian.
    w = model._interaction_matrix()
    def unrelaxed(v):
        tt, pp, n, extent = 1000*v[0], 1e8*v[1], v[2:-1], v[-1]
        return (n@model.liquid_standard_gibbs(tt, pp)
                + n.sum()*model._mixing_gibbs(tt, pp, n/n.sum(), extent/n.sum(), w))
    extent = SPECIES(t, 5e7, N)[-1]
    full = np.asarray(jax.jit(jax.hessian(unrelaxed))(jnp.r_[z, extent]))
    correction = np.outer(full[:-1, -1], full[-1, :-1])/full[-1, -1]
    assert full[-1, -1] > 0
    assert np.linalg.norm(correction) > 1
    np.testing.assert_allclose(h, full[:-1, :-1]-correction, rtol=1e-10, atol=3e-8)
    assert abs(jax.grad(unrelaxed)(jnp.r_[z, extent])[-1]) < 2e-9


def test_scaling_jit_vmap_and_zero_carbon_model_identity():
    t, p = 1700., 5e7
    np.testing.assert_allclose(N@MU(t, p, N), G(t, p, N), rtol=2e-15)
    for scale in (.01, 7.):
        np.testing.assert_allclose(G(t, p, scale*N), scale*G(t, p, N), rtol=2e-15)
        np.testing.assert_allclose(MU(t, p, scale*N), MU(t, p, N), rtol=2e-15)
        np.testing.assert_allclose(SPECIES(t, p, scale*N), scale*SPECIES(t, p, N), rtol=2e-15)
    ns = jnp.stack((N, N.at[-1].set(0), jnp.zeros(13)))
    values = jax.jit(jax.vmap(model.liquid_gibbs, in_axes=(None, None, 0)))(t, p, ns)
    np.testing.assert_allclose(values, [G(t, p, n) for n in ns], rtol=2e-15)
    assert abs(values[1]-magma.liquid_gibbs(t, p, N[:12])) > 1
    dry = N.at[-2:].set(0)
    np.testing.assert_allclose(G(t, p, dry), magma.liquid_gibbs(t, p, dry[:11]), rtol=2e-15)


def test_faces_temperature_branch_and_zero_phase_semantics():
    t, p = 1700., 5e7
    np.testing.assert_array_equal(PROPERTIES(t, p, jnp.zeros(13)), np.zeros(6))
    np.testing.assert_array_equal(SPECIES(t, p, jnp.zeros(13)), np.zeros(14))
    # The computational zero gradient is explicitly not an insertion potential.
    np.testing.assert_array_equal(MU(t, p, jnp.zeros(13)), np.zeros(13))
    for n in (*np.eye(13), N.at[3].set(0), N.at[0].set(0), N.at[-2].set(0)):
        assert np.all(np.isfinite(PROPERTIES(t, p, n)))
        assert np.all(SPECIES(t, p, n) >= 0)
    for i in (3, 12):
        assert SPECIES(t, p, N.at[i].set(0))[-1] == 0
        for amount in (1e-6, 1e-10, 1e-14):
            assert np.all(np.isfinite(PROPERTIES(t, p, N.at[i].set(amount))))
    # Trace-C potential follows the same analytic dilute limit after relaxation.
    trace1, trace2 = N.at[-1].set(1e-9), N.at[-1].set(1e-11)
    np.testing.assert_allclose(MU(t, p, trace2)[-1]-MU(t, p, trace1)[-1],
                               magma.R*t*np.log(.01), rtol=0, atol=2e-4)
    y = SPECIES(t, p, trace1)/trace1.sum()
    affinity = (model._reaction_gibbs(p)+jnp.asarray(model._REACTION)@model._interaction_matrix()@y
                + magma.R*t*(jnp.log(y[0])+jnp.log(y[13])-jnp.log(y[3])-jnp.log(y[12])))
    assert abs(affinity) < 2e-9
    q = jnp.eye(13)[0]
    left, right = PROPERTIES(1480.-1e-5, p, q), PROPERTIES(1480.+1e-5, p, q)
    assert abs(left[0]-right[0]) < .01 and abs(left[1]-right[1]) < 1e-5
    assert abs(left[2]-right[2]) > .1
    # Equimolar Ca and C lies on the numerical min() switch, not a physical branch.
    n = N.at[-1].set(N[3])
    h = jax.jit(jax.hessian(model.liquid_gibbs, 2))(t, p, n)
    dx = jnp.eye(13)[-1]*1e-5
    np.testing.assert_allclose(h[:, -1], (MU(t, p, n+dx)-MU(t, p, n-dx))/2e-5, rtol=2e-6, atol=1e-4)
    with pytest.raises(ValueError, match="shape"):
        G(t, p, N[:12])
    with pytest.raises(ValueError, match="scalar"):
        G(jnp.array([t]), p, N)
