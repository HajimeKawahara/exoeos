"""Ca/Al standards, native MELTS comparisons and conserved dry equilibria."""

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos.magma import (
    CMAS_COMPONENTS, CMFAS_COMPONENTS, R, anorthite_gibbs,
    clinopyroxene_gibbs, clinopyroxene_standard_gibbs,
    liquid_gibbs, liquid_standard_gibbs,
)


ROOT = Path(__file__).resolve().parents[2]
SAVED = json.loads((ROOT / "tests/reference/dry_magma_melts_v1.json").read_text())


@pytest.fixture(scope="module")
def dry():
    pytest.importorskip("scipy")
    spec = importlib.util.spec_from_file_location("magma_dry", ROOT / "examples/magma_dry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("state", SAVED["properties"])
def test_added_standards_and_solutions_match_actual_melts(state):
    T, P = state["T_K"], state["P_Pa"]
    np.testing.assert_allclose(
        liquid_standard_gibbs(T, P, include_fe=True, include_ca_al=True),
        state["liquid_standard_J_mol"], atol=3e-8, rtol=0)
    assert anorthite_gibbs(T, P) == pytest.approx(state["anorthite_g_J_mol"], abs=3e-8, rel=0)
    n = jnp.array(state["liquid_mol"])
    assert liquid_gibbs(T, P, n) == pytest.approx(state["liquid_g_J"], abs=4e-5, rel=0)
    np.testing.assert_allclose(jax.grad(liquid_gibbs, 2)(T, P, n), state["liquid_mu_J_mol"],
                               atol=5e-5, rtol=0)
    for row in state["clinopyroxene"]:
        if "status" in row:
            # The native endpoint is explicitly unavailable, never a fabricated value.
            assert row["status"] == "unavailable_nonfinite_native_endpoint"
            continue
        n = jnp.array(row["n_mol"])
        np.testing.assert_allclose(clinopyroxene_standard_gibbs(T, P), row["standard_J_mol"],
                                   atol=3e-8, rtol=0)
        assert clinopyroxene_gibbs(T, P, n) == pytest.approx(row["g_J"], abs=3e-8, rel=0)
        if np.all(n > 0):
            np.testing.assert_allclose(jax.grad(clinopyroxene_gibbs, 2)(T, P, n),
                                       row["mu_J_mol"], atol=3e-7, rtol=0)


def test_basis_subsets_and_exact_zero_energy_limits():
    T, P = 1600., 5e7
    assert CMAS_COMPONENTS == ("SiO2", "Mg2SiO4", "CaSiO3", "Al2O3")
    assert CMFAS_COMPONENTS == ("SiO2", "Mg2SiO4", "Fe2SiO4", "CaSiO3", "Al2O3")
    binary, ternary = jnp.array([.4, .3]), jnp.array([.4, .3, .1])
    assert liquid_gibbs(T, P, jnp.r_[binary, 0., 0.]) == liquid_gibbs(T, P, binary)
    assert liquid_gibbs(T, P, jnp.r_[ternary, 0., 0.]) == liquid_gibbs(T, P, ternary)
    n = jnp.array([.4, .3, .2, .1])
    assert liquid_gibbs(T, P, n) == liquid_gibbs(T, P, jnp.insert(n, 2, 0.))
    g0 = liquid_standard_gibbs(T, P, include_ca_al=True)
    np.testing.assert_allclose(g0, liquid_standard_gibbs(
        T, P, include_fe=True, include_ca_al=True)[jnp.array([0, 1, 3, 4])], rtol=0, atol=1e-9)
    for fun, standards in [(liquid_gibbs, g0),
                            (clinopyroxene_gibbs, clinopyroxene_standard_gibbs(T, P))]:
        size = len(standards)
        assert fun(T, P, jnp.zeros(size)) == 0.
        for i in range(size):
            assert fun(T, P, jnp.eye(size)[i]) == pytest.approx(float(standards[i]), abs=1e-9, rel=0)


@pytest.mark.parametrize("fun,n", [(liquid_gibbs, [.4, .25, .075, .25, .12]),
                                   (liquid_gibbs, [.4, .25, .25, .12]),
                                   (clinopyroxene_gibbs, [.7, .3])])
def test_jax_derivatives_extensivity_and_hessian(fun, n):
    T, P, n = 1700., 5e7, jnp.array(n)
    energy = jax.jit(fun)(T, P, n)
    mu = jax.jit(jax.grad(fun, 2))(T, P, n)
    np.testing.assert_allclose(n @ mu, energy, atol=3e-9, rtol=0)
    np.testing.assert_allclose(fun(T, P, 7*n), 7*energy, atol=1e-8, rtol=0)
    hessian = jax.jit(jax.hessian(fun, 2))(T, P, n)
    np.testing.assert_allclose(hessian, hessian.T, atol=3e-9, rtol=0)
    np.testing.assert_allclose(hessian @ n, 0., atol=3e-9, rtol=0)
    values = jax.jit(jax.vmap(fun, in_axes=(0, None, None)))(jnp.array([1600., T, 1800.]), P, n)
    assert np.all(np.isfinite(values))
    assert values[1] == pytest.approx(float(energy), abs=3e-9, rel=0)
    for arg, step in [(0, .01), (1, 100.)]:
        plus, minus = [T, P, n], [T, P, n]
        plus[arg] += step
        minus[arg] -= step
        fd = (fun(*plus)-fun(*minus))/(2*step)
        np.testing.assert_allclose(jax.grad(fun, arg)(T, P, n), fd, rtol=3e-7, atol=2e-8)


def test_clinopyroxene_has_one_mixed_site_and_anorthite_has_physical_derivatives():
    T, P, n = 1700., 5e7, jnp.array([.7, .3])
    expected = clinopyroxene_standard_gibbs(T, P) + R*T*jnp.log(n) + 7029.12*n[::-1]**2
    np.testing.assert_allclose(jax.grad(clinopyroxene_gibbs, 2)(T, P, n), expected,
                               atol=2e-9, rtol=0)
    assert -jax.grad(anorthite_gibbs, 0)(T, P) > 0
    assert jax.grad(anorthite_gibbs, 1)(T, P) > 0
    cp = -T*jax.grad(jax.grad(anorthite_gibbs, 0), 0)(T, 1e5)
    expected_cp = 439.37-3734.1/jnp.sqrt(T)-3.1702e8/T**3
    assert cp == pytest.approx(float(expected_cp), rel=2e-12)


@pytest.mark.parametrize("row", SAVED["restricted_equilibria"])
def test_competing_phases_match_independent_melts_roots_and_conserve_elements(dry, row):
    state = dry.equilibrium(row["T_K"], row["P_Pa"], row["initial_oxide_moles"])
    for field in ("liquid_mol", "solid_mol", "phase_mol"):
        np.testing.assert_allclose(state[field], row[field], atol=5e-8, rtol=0)
        assert min(state[field]) >= 0
    assert state["g_J"] == pytest.approx(row["g_J"], abs=5e-5, rel=0)
    # Atom counts in liquid coordinates, including oxygen and 2 Al per Al2O3.
    atoms = np.array([[0, 0, 0, 1, 0], [0, 2, 0, 0, 0], [0, 0, 2, 0, 0],
                      [0, 0, 0, 0, 2], [1, 1, 1, 1, 0], [2, 4, 4, 3, 3]])
    expected = atoms @ dry.component_amounts(row["initial_oxide_moles"])
    inventory = np.array(state["liquid_mol"]) + dry.SOLID_COMPONENTS @ state["solid_mol"]
    np.testing.assert_allclose(atoms @ inventory, expected, atol=1e-12, rtol=0)
    masses = np.array([40.078, 24.305, 55.845, 26.981538, 28.0855, 15.9994])
    assert masses @ atoms @ inventory == pytest.approx(float(masses @ expected), abs=1e-11, rel=0)
    gaps = np.array(state["solid_insertion_J_mol"])
    assert min(gaps) >= -2e-4
    present = np.array(state["phase_mol"]) > 0
    np.testing.assert_allclose(gaps[present], 0., atol=2e-4, rtol=0)


def test_cooling_sequence_scaling_and_unresolved_disappearance(dry):
    assert dry.equilibrium(1800.)["phase_mol"] == [0.]*5
    # Near onset, an absent solution's composition must be searched explicitly.
    assert dry.equilibrium(1670.)["phase_mol"][0] > 0
    low = dry.equilibrium(1425.)
    assert min(low["phase_mol"][:3]) > 0
    scaled = dry.equilibrium(1425., oxides={k: 7*v for k, v in dry.DEFAULT_OXIDES.items()})
    np.testing.assert_allclose(scaled["solid_mol"], 7*np.array(low["solid_mol"]), atol=1e-9, rtol=0)
    with pytest.raises(RuntimeError, match="liquid disappearance"):
        dry.equilibrium(1300.)


def test_invalid_shapes_and_oxide_basis(dry):
    for fun, n in [(liquid_gibbs, [1.]*6), (clinopyroxene_gibbs, [1.]*3)]:
        with pytest.raises(ValueError, match="shape"):
            fun(1600., 1e5, n)
    with pytest.raises(ValueError, match="scalars"):
        anorthite_gibbs([1600.], 1e5)
    with pytest.raises(ValueError, match="positive liquid basis"):
        dry.component_amounts({**dry.DEFAULT_OXIDES, "SiO2": .1})
