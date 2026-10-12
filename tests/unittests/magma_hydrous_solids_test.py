"""Hydrous mineral references, relaxed derivatives and physical site faces."""

from functools import partial
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos.magma import (HYDROUS_COMPONENTS, HYDROUS_OXIDES, element_matrix,
                         liquid_gibbs, liquid_moles_from_oxides)
from exoeos.magma_solids import amphibole, biotite, hornblende

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = json.loads((ROOT/'tests/reference/hydrous_solids_melts_v1.json').read_text())
MODELS = {'biotite': biotite, 'hornblende': hornblende,
          'clinoamphibole': amphibole, 'orthoamphibole': amphibole}


def properties(fun, t, p, n):
    return jnp.array([fun(t, p, n), -jax.grad(fun)(t, p, n),
                      -t*jax.grad(jax.grad(fun))(t, p, n), jax.grad(fun, 1)(t, p, n)])


@pytest.fixture(params=tuple(MODELS), scope='module')
def phase(request):
    name = request.param
    module = MODELS[name]
    kwargs = {'phase': name} if module is amphibole else {}
    fun = partial(module.gibbs, **kwargs)
    yield name, module, kwargs, jax.jit(fun), jax.jit(partial(properties, fun)), jax.jit(jax.grad(fun, 2))
    jax.clear_caches()


def test_independent_native_properties(phase):
    name, module, _, _, prop, mu = phase
    standard = jax.jit(module.standard_gibbs)
    rows = [r for r in REFERENCE['properties'] if r['phase'] == name]
    assert len(rows) == 45
    for row in rows:
        t, p, n = row['T_K'], row['P_Pa'], np.array(row['n_mol'])
        actual = np.array(prop(t, p, n))
        expected = np.array([row[k] for k in ('g_J', 's_J_K', 'cp_J_K', 'v_m3')])
        assert np.all(abs(actual-expected) < [1e-8, 2e-9, 2e-8, 1e-17]), actual-expected
        np.testing.assert_allclose(mu(t, p, n), row['mu_J_mol'], atol=1e-8, rtol=0)
        np.testing.assert_allclose(standard(t, p), row['standard_J_mol'], atol=1e-8, rtol=0)


def test_joint_hessian_homogeneity_and_finite_differences(phase):
    name, _, _, fun, prop, mu = phase
    row = next(r for r in REFERENCE['properties'] if r['phase'] == name)
    t, p, n = 1100., 5e7, jnp.array(row['n_mol'])
    np.testing.assert_allclose(n@mu(t, p, n), fun(t, p, n), atol=6e-9, rtol=0)
    np.testing.assert_allclose(prop(t, p, 7*n), 7*prop(t, p, n), atol=3e-8, rtol=2e-13)
    np.testing.assert_allclose(mu(t, p, 7*n), mu(t, p, n), atol=6e-9, rtol=0)
    # Scale pressure so small mixed blocks cannot pass a loose J tolerance.
    state = jnp.r_[t/1000, p/1e8, n]
    joint = lambda z: fun(1000*z[0], 1e8*z[1], z[2:])
    gradient, hessian = jax.jit(jax.grad(joint)), jax.jit(jax.hessian(joint))
    h = hessian(state)
    np.testing.assert_allclose(h, h.T, atol=1e-7, rtol=2e-10)
    np.testing.assert_allclose(h[2:, 2:]@n, 0., atol=1e-8, rtol=0)
    for i in range(state.size):
        step = jnp.eye(state.size)[i]*1e-5
        finite = (gradient(state+step)-gradient(state-step))/2e-5
        np.testing.assert_allclose(h[:, i], finite, atol=.003, rtol=4e-6)
    batched = jax.jit(jax.vmap(fun, in_axes=(0, None, None)))(jnp.array([800., t, 2200.]), p, n)
    np.testing.assert_allclose(batched[1], fun(t, p, n), atol=4e-9, rtol=0)


def test_zero_phase_and_pure_fixed_face_limits(phase):
    name, module, kwargs, fun, prop, _ = phase
    np.testing.assert_array_equal(prop(1200., 5e7, np.zeros(len(module.COMPONENTS))), np.zeros(4))
    # No zero-phase composition gradient is interpreted as a chemical potential.
    for t in (800., 1200., 2200.):
        for n in np.eye(len(module.COMPONENTS)):
            value = np.array(prop(t, 5e7, n))
            assert np.all(np.isfinite(value)) and value[2] > 0 and value[3] > 0
            if name != 'orthoamphibole':
                np.testing.assert_allclose(value[0], n@module.standard_gibbs(t, 5e7), atol=4e-9, rtol=0)
            dt = (fun(t+.02, 5e7, n)-fun(t-.02, 5e7, n))/.04
            cp = t*(prop(t+.02, 5e7, n)[1]-prop(t-.02, 5e7, n)[1])/.04
            np.testing.assert_allclose(-dt, value[1], atol=1e-6, rtol=2e-8)
            np.testing.assert_allclose(cp, value[2], atol=1e-6, rtol=2e-8)
            mixing = module.mixing_gibbs(t, 5e7, n, **kwargs)
            np.testing.assert_allclose(value[0], n@module.standard_gibbs(t, 5e7)+mixing, atol=4e-9, rtol=0)
    for row in REFERENCE['endpoints']:
        if row['phase'] != name:
            continue
        values = prop(row['T_K'], row['P_Pa'], np.array(row['n_mol']))
        for value, key, tol in zip(values, ('g_J', 's_J_K', 'cp_J_K', 'v_m3'), (1e-8, 2e-9, 2e-8, 1e-17)):
            if row[key] is not None:
                np.testing.assert_allclose(value, row[key], atol=tol, rtol=0)


def test_atom_ledger_and_signed_physical_domains(phase):
    name, module, _, fun, prop, mu = phase
    data = json.loads((ROOT/'examples/m2_solid_mixing/parameters.json').read_text())
    matrix = np.array(data['models'][name]['endmember_oxide_moles']).T
    indices = [data['oxide_order'].index(x.lower()) for x in HYDROUS_OXIDES]
    expected = element_matrix(HYDROUS_OXIDES)@matrix[indices]
    np.testing.assert_array_equal(element_matrix(module.COMPONENTS), expected)
    np.testing.assert_array_equal(expected[-1], np.full(len(module.COMPONENTS), 2))
    if module is hornblende:
        n = jnp.array([-.2, .6, .6])
    elif module is amphibole:
        n = jnp.array([-.2, .5, .7])
    else:
        return
    assert np.all(expected@n >= 0)
    assert np.all(np.isfinite(prop(1100., 5e7, n)))
    np.testing.assert_allclose(n@mu(1100., 5e7, n), fun(1100., 5e7, n), atol=5e-9, rtol=0)


def test_dilute_approaches_and_api_shapes(phase):
    _, module, _, fun, prop, mu = phase
    for pure in np.eye(len(module.COMPONENTS)):
        for eps in (1e-5, 1e-9, 1e-12):
            n = jnp.asarray(pure+eps*(1-pure))
            assert np.all(np.isfinite(prop(1200., 5e7, n)))
            potential = mu(1200., 5e7, n)
            assert np.all(np.isfinite(potential))
            np.testing.assert_allclose(n@potential, fun(1200., 5e7, n), atol=8e-9, rtol=0)
    with pytest.raises(ValueError, match='shape'):
        fun(1200., 5e7, jnp.ones(5))
    with pytest.raises(ValueError, match='scalar'):
        fun(jnp.array([1200.]), 5e7, jnp.ones(len(module.COMPONENTS)))


def test_hydrous_mineral_transfer_from_liquid_conserves_atoms(phase):
    name, module, _, fun, _, _ = phase
    data = json.loads((ROOT/'examples/m2_solid_mixing/parameters.json').read_text())
    matrix = np.array(data['models'][name]['endmember_oxide_moles']).T
    indices = [data['oxide_order'].index(x.lower()) for x in HYDROUS_OXIDES]
    y = jnp.ones(len(module.COMPONENTS))/len(module.COMPONENTS)
    consumed = liquid_moles_from_oxides(matrix[indices]@y)
    bulk = consumed+jnp.ones(12)
    liquid = bulk-.1*consumed
    np.testing.assert_allclose(element_matrix(HYDROUS_COMPONENTS)@consumed,
                               element_matrix(module.COMPONENTS)@y, atol=4e-15, rtol=0)
    assert consumed[-1] == 1.  # One formula mol binds one mol H2O.
    total = lambda xi: liquid_gibbs(1200., 5e7, bulk-xi*consumed)+fun(1200., 5e7, xi*y)
    slope = jax.jit(jax.grad(total))(.1)
    mu_liquid = jax.grad(liquid_gibbs, 2)(1200., 5e7, liquid)
    np.testing.assert_allclose(slope, fun(1200., 5e7, y)-consumed@mu_liquid, atol=2e-8, rtol=0)
    np.testing.assert_allclose(element_matrix(HYDROUS_COMPONENTS)@liquid
                               +element_matrix(module.COMPONENTS)@(.1*y),
                               element_matrix(HYDROUS_COMPONENTS)@bulk, atol=1e-14, rtol=0)


@pytest.mark.parametrize('structure', ['clinoamphibole', 'orthoamphibole'])
def test_amphibole_stationarity_and_order_faces(structure):
    order = jax.jit(partial(amphibole.order, phase=structure))
    energy, sites, _ = amphibole._model(structure)
    for n in (jnp.array([.4, .3, .3]), jnp.array([-.2, .5, .7]),
              jnp.array([.6, .4, 0.]), jnp.array([.5, 0., .5]),
              jnp.array([-.3, .3, 1.]), jnp.array([-5/7, 5/7, 1.])):
        q = order(1200., 5e7, n)
        assert np.all(np.isfinite(q)) and np.all(sites(q) >= -1e-14)
        np.testing.assert_allclose(2*q[0]+3*q[1]+2*q[2], 7*n[1]/n.sum(), atol=2e-15)
        if np.all(sites(q) > 0):
            # Two independent atom-preserving Fe/Mg exchanges.
            directions = jnp.array([[1., 0., -1., 0.], [0., 1., -1.5, 0.]]).T
            slope = jax.grad(energy, 2)(1200., 5e7, q)
            np.testing.assert_allclose(directions.T@slope, 0., atol=2e-6)
            curvature = directions.T@jax.hessian(energy, 2)(1200., 5e7, q)@directions
            assert np.linalg.eigvalsh(curvature)[0] > 0


def test_transcription_and_native_projection_diagnostic():
    path = ROOT/'tests/reference/extract_hydrous_solid_coefficients.py'
    spec = importlib.util.spec_from_file_location('extract_hydrous_solids', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.render() == module.DESTINATION.read_text()
    # The native oxide projection must not silently replace the requested state.
    rows = [r for r in REFERENCE['properties'] if r['phase'] == 'hornblende']
    assert all(not np.allclose(r['oxide_projection_fractions'], r['n_mol']) for r in rows)
    with pytest.raises(ValueError, match='phase'):
        amphibole.gibbs(1200., 1e5, jnp.ones(3), phase='amphibole')


def test_restricted_water_effect_against_independent_native_roots():
    path = ROOT/'examples/magma_hydrous_solids.py'
    spec = importlib.util.spec_from_file_location('hydrous_example', path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    for row in REFERENCE['cooling']:
        t, p = row['T_K'], row['P_Pa']
        bulk = jnp.array(row['bulk_n_mol'])
        if row['water_mol'] == 0:
            bulk = bulk[:-1]
        xi = example.crystal_amount(t, p, bulk)
        np.testing.assert_allclose(xi, row['forsterite_mol'], atol=2e-9, rtol=0)
        np.testing.assert_allclose(example.total_gibbs(t, p, xi, bulk), row['g_J'], atol=3e-5, rtol=0)
        slope = jax.jit(jax.grad(example.total_gibbs, 2))(t, p, xi, bulk)
        assert float(slope) >= -1e-7
        if xi > 0:
            assert abs(float(slope)) < 1e-7
    # These particular paths have positive curvature throughout the interval,
    # justifying the increasing-slope bisection without a global solver claim.
    for water in (0., .02, .1, .3):
        bulk = example.DRY_HOST if water == 0 else jnp.r_[example.DRY_HOST, water]
        curvature = jax.jit(jax.vmap(jax.grad(jax.grad(example.total_gibbs, 2), 2),
                                    in_axes=(None, None, 0, None)))
        assert np.all(curvature(1600., 5e7, jnp.linspace(0., .749, 101), bulk) > 0)
