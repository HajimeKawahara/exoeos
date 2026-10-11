"""Ca--Mg--Fe pyroxenes: independent MELTS properties and relaxed derivatives."""

from functools import partial
import importlib.util
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos.magma import (
    PYROXENE_COMPONENTS, _pyroxene_coefficients, _pyroxene_order,
    clinopyroxene_gibbs, enstatite_gibbs, pyroxene_gibbs,
)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = json.loads((ROOT / "tests/reference/pyroxene_melts_v1.json").read_text())
ENERGIES = {phase: jax.jit(partial(pyroxene_gibbs, phase=phase))
            for phase in ("clinopyroxene", "orthopyroxene")}
DERIVATIVES = {phase: (jax.jit(jax.grad(fun, 0)), jax.jit(jax.grad(fun, 1)),
                       jax.jit(jax.grad(jax.grad(fun, 0), 0)), jax.jit(jax.grad(fun, 2)))
               for phase, fun in ENERGIES.items()}


@pytest.mark.parametrize("row", REFERENCE["properties"])
def test_properties_against_independent_melts(row):
    if "status" in row:
        assert row["status"] == "unavailable_nonfinite_native_endpoint"
        return
    phase, T, P, n = row["phase"], row["T_K"], row["P_Pa"], jnp.array(row["n_mol"])
    assert ENERGIES[phase](T, P, n) == pytest.approx(row["g_J"], abs=5e-9, rel=0)
    if "mu_J_mol" in row:
        dt, dp, dtt, dn = DERIVATIVES[phase]
        assert -dt(T, P, n) == pytest.approx(row["s_J_K"], abs=3e-11, rel=0)
        assert -T*dtt(T, P, n) == pytest.approx(row["cp_J_K"], abs=3e-10, rel=0)
        assert dp(T, P, n) == pytest.approx(row["v_m3"], abs=2e-18, rel=0)
        np.testing.assert_allclose(dn(T, P, n), row["mu_J_mol"], atol=5e-9, rtol=0)


@pytest.mark.parametrize("T,P", [(1200., 1e5), (1600., 5e7), (2200., 5e8)])
def test_exact_edges_recover_existing_binary_and_enstatite(T, P):
    assert PYROXENE_COMPONENTS == ("CaSiO3", "MgSiO3", "FeSiO3")
    for y in (0., .2, .8, 1.):
        assert pyroxene_gibbs(T, P, [1., 1-y, y]) == pytest.approx(
            float(clinopyroxene_gibbs(T, P, [1-y, y])), abs=3e-9, rel=0)
    fun = ENERGIES["orthopyroxene"]
    assert fun(T, P, jnp.array([0., 1., 0.])) == pytest.approx(float(enstatite_gibbs(T, P)), abs=2e-9, rel=0)
    for arg in (0, 1):
        assert jax.grad(fun, arg)(T, P, jnp.array([0., 1., 0.])) == pytest.approx(
            float(jax.grad(enstatite_gibbs, arg)(T, P)), rel=2e-13)
    for phase, fun in ENERGIES.items():
        assert fun(T, P, jnp.zeros(3)) == 0.
        for n in ([0., 0., 2.], [.2, 1.8, 0.], [0., 1., 1.]):
            assert np.isfinite(fun(T, P, jnp.array(n)))
            assert jax.grad(fun, 1)(T, P, jnp.array(n)) > 0


@pytest.mark.parametrize("phase", ENERGIES)
def test_implicit_order_hessian_and_finite_differences(phase):
    fun = ENERGIES[phase]
    T, P, n = 1600., 5e7, jnp.array([.3, 1.2, .5])
    mu = jax.grad(fun, 2)(T, P, n)
    np.testing.assert_allclose(n@mu, fun(T, P, n), atol=3e-9, rtol=0)
    np.testing.assert_allclose(fun(T, P, 7*n), 7*fun(T, P, n), atol=1e-8, rtol=0)
    hessian = jax.jit(jax.hessian(fun, 2))(T, P, n)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-8, rtol=0)
    np.testing.assert_allclose(hessian@n, 0., atol=1e-8, rtol=0)
    for i in range(3):
        step = jnp.eye(3)[i]*1e-5
        fd = (jax.grad(fun, 2)(T, P, n+step)-jax.grad(fun, 2)(T, P, n-step))/(2e-5)
        np.testing.assert_allclose(hessian[:, i], fd, atol=7e-5, rtol=2e-7)
    c, f = 2*n[0]/n.sum(), 2*n[2]/n.sum()
    coefficients = _pyroxene_coefficients(T, P, c, f, phase == "clinopyroxene")
    b = _pyroxene_order(T, c, f, coefficients)
    slope = coefficients[1]+2*coefficients[2]*b+8.3143*T*jnp.log((1-f+b)*b/((f-b)*(1-c-b)))
    assert abs(slope) < 1e-9
    assert 0 < b < min(f, 1-c)
    assert abs(b-f*(1-c)/(2-c)) > 1e-3  # Not a random-site approximation.
    states = jnp.array([[1200., 1e5], [T, P], [2200., 5e8]])
    batched = jax.jit(jax.vmap(lambda tp: fun(tp[0], tp[1], n)))(states)
    assert np.all(np.isfinite(batched))
    assert batched[1] == pytest.approx(float(fun(T, P, n)), abs=2e-9, rel=0)
    for arg, step in ((0, .01), (1, 100.)):
        plus, minus = [T, P, n], [T, P, n]
        plus[arg] += step
        minus[arg] -= step
        fd = (fun(*plus)-fun(*minus))/(2*step)
        assert jax.grad(fun, arg)(T, P, n) == pytest.approx(float(fd), rel=5e-7)


def test_invalid_static_inputs():
    with pytest.raises(ValueError, match="shape"):
        pyroxene_gibbs(1600., 1e5, [1., 1.])
    with pytest.raises(ValueError, match="phase"):
        pyroxene_gibbs(1600., 1e5, [1., 1., 1.], phase="unknown")
    with pytest.raises(ValueError, match="scalars"):
        pyroxene_gibbs([1600.], 1e5, [1., 1., 1.])


@pytest.mark.parametrize("phase", ENERGIES)
def test_reduced_polynomial_matches_independent_full_site_transcription(phase):
    model = json.loads((ROOT / "examples/m2_solid_mixing/parameters.json").read_text())["models"][phase]
    T, P = 1730., 2.5e8
    for c, f, b in ((.8, .2, .06), (.05, .45, .3), (.2, 1.5, .65)):
        coordinates = np.array([f-b, 0., 0., 0., b, 1-c-b, 0., 0.])
        def polynomial(terms):
            return sum(coefficient*np.prod(coordinates**np.array(powers)) for coefficient, powers in terms)
        # Native En pure reference is 1.08018328*T for BOTH structures.
        expected = (polynomial(model["H_polynomial"])-T*polynomial(model["S_polynomial"])
                    +(P/1e5-1)*polynomial(model["V_polynomial"])-1.08018328*T*(1-c))
        actual = _pyroxene_coefficients(T, P, c, f, phase == "clinopyroxene")@jnp.array([1., b, b*b])
        assert actual == pytest.approx(expected, abs=3e-9, rel=0)


@pytest.fixture(scope="module")
def cooling():
    pytest.importorskip("scipy")
    # The numerical example reuses the preceding example's oxide bookkeeping.
    sys.path.insert(0, str(ROOT / "examples"))
    try:
        spec = importlib.util.spec_from_file_location("magma_pyroxene", ROOT / "examples/magma_pyroxene.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_cooling_conservation_and_solid_tangent_stability(cooling):
    for T in (1800., 1600., 1475., 1465.):
        state = cooling.equilibrium(T)
        inventory = np.array(state["liquid_mol"])+cooling.SOLID_COMPONENTS@state["solid_mol"]
        np.testing.assert_allclose(inventory, cooling.component_amounts(cooling.DEFAULT_OXIDES), atol=1e-12, rtol=0)
        assert min(state["liquid_mol"]+state["solid_mol"]) >= 0
        gaps = np.array(state["solid_insertion_J_mol_Si"])
        assert min(gaps) >= -5e-4
        np.testing.assert_allclose(gaps[np.array(state["phase_mol_Si"]) > 0], 0., atol=5e-4, rtol=0)
    assert min(state["phase_mol_Si"][:4]) > 0  # Both variable-composition pyroxenes coexist.
    assert 0 < state["opx_Ca_M2"] < state["cpx_Ca_M2"] < 1
    assert state["opx_Fe_fraction"] > 0
    scaled = cooling.equilibrium(1465., oxides={k: 7*v for k, v in cooling.DEFAULT_OXIDES.items()})
    np.testing.assert_allclose(scaled["solid_mol"], 7*np.array(state["solid_mol"]), atol=2e-8, rtol=0)
    with pytest.raises(RuntimeError, match="liquid disappearance"):
        cooling.equilibrium(1450.)


@pytest.mark.parametrize("row", REFERENCE["restricted_equilibria"])
def test_cooling_against_independent_native_chemical_potential_roots(cooling, row):
    state = cooling.equilibrium(row["T_K"], row["P_Pa"], row["initial_oxide_moles"])
    for field in ("liquid_mol", "solid_mol"):
        np.testing.assert_allclose(state[field], row[field], atol=5e-8, rtol=0)
    assert state["g_J"] == pytest.approx(row["g_J"], abs=4e-5, rel=0)
