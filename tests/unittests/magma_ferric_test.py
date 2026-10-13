"""Independent ferric liquid standards, mixing, derivatives and zero limits."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos.magma import (
    CMFAS_COMPONENTS, FERRIC_COMPONENTS, element_matrix,
    liquid_gibbs, liquid_standard_gibbs,
)


REFERENCE = Path(__file__).resolve().parents[1] / "reference"
SOURCE = json.loads((REFERENCE / "ferric_source_v1.json").read_text())
MELTS = json.loads((REFERENCE / "ferric_melts_v1.json").read_text())
STANDARDS = jax.jit(lambda t, p: liquid_standard_gibbs(
    t, p, include_fe=True, include_ca_al=True, include_ferric=True))
MU = jax.jit(jax.grad(liquid_gibbs, 2))


def properties(fun, t, p, *args):
    return jnp.array([
        fun(t, p, *args), -jax.grad(fun, 0)(t, p, *args),
        -t*jax.grad(jax.grad(fun, 0), 0)(t, p, *args),
        jax.grad(fun, 1)(t, p, *args),
        jax.grad(jax.grad(fun, 1), 0)(t, p, *args),
        jax.grad(jax.grad(fun, 1), 1)(t, p, *args),
    ])


LIQUID_PROPERTIES = jax.jit(lambda t, p, n: properties(liquid_gibbs, t, p, n))
FERRIC_PROPERTIES = jax.jit(lambda t, p: properties(lambda t, p: STANDARDS(t, p)[5], t, p))


@pytest.mark.parametrize("row", SOURCE["states"])
def test_ferric_standard_matches_independent_source_quadrature(row):
    expected = [row[key] for key in ("g_J_mol", "s_J_mol_K", "cp_J_mol_K",
                                    "v_m3_mol", "dvdt_m3_mol_K", "dvdp_m3_mol_Pa")]
    np.testing.assert_allclose(FERRIC_PROPERTIES(row["T_K"], row["P_Pa"]), expected,
                               rtol=3e-12, atol=1e-20)


@pytest.mark.parametrize("row", MELTS["states"])
def test_liquid_matches_independent_melts_properties(row):
    T, P, n = row["T_K"], row["P_Pa"], jnp.array(row["n_mol"])
    actual = np.asarray(LIQUID_PROPERTIES(T, P, n))
    expected = np.array([row[key] for key in ("g_J", "s_J_K", "cp_J_K", "v_m3",
                                             "dvdt_m3_K", "dvdp_m3_Pa")])
    # Native oxide round trips differ at ~1e-11 relative precision; V also
    # retains the existing runtime/source offset. Keep absolute SI bounds.
    tolerance = np.array([5e-5, 1e-8, 1e-8, 1e-14, 2e-19, 2e-24])
    assert np.all(np.abs(actual-expected) <= tolerance), (actual-expected)
    present = np.array(n) > 0
    standard = np.array(row["standard_J_mol"], dtype=float)
    np.testing.assert_allclose(np.asarray(STANDARDS(T, P))[present], standard[present],
                               rtol=0, atol=3e-8)
    if np.all(present):
        np.testing.assert_allclose(MU(T, P, n), row["mu_J_mol"], rtol=0, atol=2e-6)
    np.testing.assert_allclose(row["returned_component_fractions"], n/n.sum(),
                               rtol=0, atol=1e-10)


@pytest.mark.parametrize("T,P", [(1200., 1e5), (1600., 5e7), (2200., 5e8)])
def test_zero_ferric_recovers_existing_liquid_and_thermal_derivatives(T, P):
    # Exact boundary: G and fixed-composition T/P derivatives are defined.
    n = jnp.array([.4, .275, .075, .25, .12])
    np.testing.assert_allclose(LIQUID_PROPERTIES(T, P, jnp.r_[n, 0.]),
                               LIQUID_PROPERTIES(T, P, n), rtol=3e-15, atol=1e-20)
    # JIT and eager arithmetic can differ by one ulp across JAX versions.
    np.testing.assert_allclose(STANDARDS(T, P)[:5], liquid_standard_gibbs(
        T, P, include_fe=True, include_ca_al=True), rtol=0, atol=1e-9)
    # Composition derivatives use the positive interior. The five present
    # potentials approach the old model; the absent ferric mu diverges.
    old_mu = np.asarray(MU(T, P, n))
    errors, ferric_mu = [], []
    for epsilon in (1e-4, 1e-7, 1e-10):
        mu = np.asarray(MU(T, P, jnp.r_[n, epsilon]))
        errors.append(max(abs(mu[:5]-old_mu)))
        ferric_mu.append(mu[5])
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-4
    assert ferric_mu[2] < ferric_mu[1] < ferric_mu[0]


def test_atom_ledger_tracks_both_iron_valences_and_oxygen():
    assert FERRIC_COMPONENTS == CMFAS_COMPONENTS + ("Fe2O3",)
    n = jnp.array([.4, .275, .075, .25, .12, .025])
    # Oxides: SiO2=1, MgO=.55, FeO=.15, CaO=.25, Al2O3=.12, Fe2O3=.025 mol.
    np.testing.assert_allclose(element_matrix(FERRIC_COMPONENTS) @ n,
                               [3.385, .55, 1., .2, .24, .25, 0., 0., 0., 0., 0., 0., 0.], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(element_matrix(("Fe2O3",))[:, 0], [3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def test_extensivity_hessian_jit_vmap_and_endpoints():
    T, P, n = 1700., 5e7, jnp.array([.4, .275, .075, .25, .12, .025])
    g, mu = liquid_gibbs(T, P, n), MU(T, P, n)
    np.testing.assert_allclose(n @ mu, g, rtol=0, atol=2e-9)
    np.testing.assert_allclose(liquid_gibbs(T, P, 7*n), 7*g, rtol=0, atol=1e-8)
    hessian = jax.jit(jax.hessian(liquid_gibbs, 2))(T, P, n)
    np.testing.assert_allclose(hessian, hessian.T, rtol=0, atol=3e-9)
    np.testing.assert_allclose(hessian @ n, 0., rtol=0, atol=3e-9)
    values = jax.jit(jax.vmap(liquid_gibbs, in_axes=(0, None, None)))(
        jnp.array([1600., T, 1800.]), P, n)
    assert np.all(np.isfinite(values))
    assert values[1] == pytest.approx(float(g), rel=0, abs=2e-9)
    np.testing.assert_array_equal(LIQUID_PROPERTIES(T, P, jnp.zeros(6)), np.zeros(6))
    for i in range(6):
        assert liquid_gibbs(T, P, jnp.eye(6)[i]) == pytest.approx(float(STANDARDS(T, P)[i]), rel=0, abs=1e-9)


@pytest.mark.parametrize("fe,ca_al", [(False, False), (True, False), (False, True)])
def test_ferric_standard_requires_unambiguous_six_component_basis(fe, ca_al):
    with pytest.raises(ValueError, match="include_ferric requires"):
        liquid_standard_gibbs(1700., 1e5, include_fe=fe, include_ca_al=ca_al, include_ferric=True)
