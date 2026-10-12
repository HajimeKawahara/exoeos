"""Dry mineral standards, native properties, site faces and relaxed AD."""

from functools import partial
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import magma
from exoeos.magma_solids import feldspar, oxides, pyroxene, spinel

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = json.loads((ROOT/'tests/reference/dry_solids_melts_v1.json').read_text())
MODELS = {'plagioclase': feldspar, 'alkali-feldspar': feldspar, 'spinel': spinel,
          'rhm-oxide': oxides, 'clinopyroxene': pyroxene, 'orthopyroxene': pyroxene}


@pytest.fixture(params=tuple(MODELS), scope='module')
def phase(request):
    name = request.param
    module = MODELS[name]
    kwargs = {'phase': name} if module in (feldspar, pyroxene) else {}
    fun = partial(module.gibbs, **kwargs)
    standard = partial(module.standard_gibbs, **({'phase': name} if module is feldspar else {}))
    result = (name, module, jax.jit(fun), jax.jit(standard),
              jax.jit(jax.grad(fun, (0, 1, 2))),
              jax.jit(jax.grad(jax.grad(fun, 0), 0)))
    yield result
    jax.clear_caches()


def test_independent_native_properties(phase):
    name, _, fun, standard, grad, dtt = phase
    rows = [r for r in REFERENCE['properties'] if r['phase'] == name]
    assert len(rows) == (39 if name == 'rhm-oxide' else 33)
    for row in rows:
        T, P, n = row['T_K'], row['P_Pa'], jnp.array(row['n_mol'])
        np.testing.assert_allclose(standard(T, P), row['standard_J_mol'], rtol=0, atol=4e-9)
        assert fun(T, P, n) == pytest.approx(row['g_J'], abs=5e-9, rel=0)
        dt, dp, mu = grad(T, P, n)
        assert -dt == pytest.approx(row['s_J_K'], abs=1e-9, rel=0)
        assert dp == pytest.approx(row['v_m3'], abs=1e-17, rel=0)
        assert -T*dtt(T, P, n) == pytest.approx(row['cp_J_K'], abs=2e-8, rel=0)
        np.testing.assert_allclose(mu, row['mu_J_mol'], rtol=0, atol=1e-8)


def test_relaxed_hessian_and_extensivity(phase):
    name, module, fun, _, grad, dtt = phase
    row = next(r for r in REFERENCE['properties'] if r['phase'] == name)
    T, P, n = 1600., 5e7, jnp.array(row['n_mol'])
    mu = grad(T, P, n)[2]
    np.testing.assert_allclose(n@mu, fun(T, P, n), atol=4e-9, rtol=0)
    np.testing.assert_allclose(fun(T, P, 3.7*n), 3.7*fun(T, P, n), atol=1e-8, rtol=0)
    # Scale external coordinates so small pressure blocks cannot pass on
    # a large absolute tolerance. Include T/n, P/n and T/P Maxwell pairs.
    state = jnp.r_[T/1000., P/1e8, n]
    joint = lambda z: fun(1000.*z[0], 1e8*z[1], z[2:])
    joint_gradient = jax.jit(jax.grad(joint))
    full = jax.jit(jax.hessian(joint))(state)
    np.testing.assert_allclose(full, full.T, atol=1e-7, rtol=2e-10)
    for index in (0, 1):
        delta = jnp.eye(state.size)[index]*1e-5
        finite = (joint_gradient(state+delta)-joint_gradient(state-delta))/2e-5
        np.testing.assert_allclose(full[:, index], finite, atol=.003, rtol=3e-6)
    hessian = full[2:, 2:]
    np.testing.assert_allclose(hessian, hessian.T, atol=2e-8, rtol=0)
    np.testing.assert_allclose(hessian@n, 0., atol=2e-8, rtol=0)
    direction = jnp.linspace(-.7, .9, n.size)
    eps = 2e-6
    fd = (grad(T, P, n+eps*direction)[2]-grad(T, P, n-eps*direction)[2])/(2*eps)
    np.testing.assert_allclose(hessian@direction, fd, atol=.003, rtol=2e-7)
    for arg, step in ((0, .02), (1, 500.)):
        plus, minus = [T, P, n], [T, P, n]
        plus[arg] += step
        minus[arg] -= step
        fd = (fun(*plus)-fun(*minus))/(2*step)
        np.testing.assert_allclose(grad(T, P, n)[arg], fd, atol=1e-8, rtol=3e-7)
    fd_tt = (grad(T+.02, P, n)[0]-grad(T-.02, P, n)[0])/.04
    np.testing.assert_allclose(dtt(T, P, n), fd_tt, atol=2e-9, rtol=1e-7)
    batch = jax.jit(jax.vmap(fun, in_axes=(0, None, None)))(jnp.array([1400., T, 1800.]), P, n)
    assert np.all(np.isfinite(batch))
    assert batch[1] == pytest.approx(float(fun(T, P, n)), abs=3e-9, rel=0)
    assert fun(T, P, jnp.zeros(len(module.COMPONENTS))) == 0.
    # The zero-branch AD sentinel is not an absent phase's chemical potential.
    for derivative in grad(T, P, jnp.zeros(len(module.COMPONENTS))):
        np.testing.assert_array_equal(derivative, np.zeros_like(derivative))


def test_positive_interior_limits_against_native(phase):
    name, module, fun, _, grad, dtt = phase
    rows = [r for r in REFERENCE['interior_limits'] if r['phase'] == name]
    for row in rows:
        T, P, n = row['T_K'], row['P_Pa'], jnp.array(row['n_mol'])
        dt, dp, mu = grad(T, P, n)
        actual = np.array([fun(T, P, n), -dt, -T*dtt(T, P, n), dp])
        expected = np.array([row[k] for k in ('g_J', 's_J_K', 'cp_J_K', 'v_m3')])
        assert np.all(abs(actual-expected) <= [5e-9, 1e-9, 2e-8, 1e-17]), actual-expected
        assert np.all(np.isfinite(mu))
        np.testing.assert_allclose(n@mu, actual[0], atol=4e-9, rtol=0)
        # At an interior root every modeled site is positive, including the
        # small exchange populations that must not land on a zero-site face.
        kwargs = {'phase': name} if module is pyroxene else {}
        order = module.order(T, P, n, **kwargs)
        sites = module._model(name)[1] if module is pyroxene else module._SITES
        assert np.all(sites(order) > 0)


def test_pure_endmembers_and_fixed_face_derivatives(phase):
    name, module, fun, standard, grad, dtt = phase
    for T in (1200., 1600., 2200.):
        P = 5e7
        for n in jnp.eye(len(module.COMPONENTS)):
            value = fun(T, P, n)
            assert np.isfinite(value)
            if name != 'orthopyroxene':
                np.testing.assert_allclose(value, n@standard(T, P), atol=5e-9, rtol=0)
            # The mixture chemical potential is singular at a face; only T/P
            # derivatives are asserted here.
            dt, dp, _ = grad(T, P, n)
            assert np.isfinite(dt) and np.isfinite(dp) and dp > 0
            fd = (fun(T+.02, P, n)-fun(T-.02, P, n))/.04
            np.testing.assert_allclose(dt, fd, atol=1e-6, rtol=1e-7)
            fd_tt = (grad(T+.02, P, n)[0]-grad(T-.02, P, n)[0])/.04
            np.testing.assert_allclose(dtt(T, P, n), fd_tt, atol=2e-9, rtol=1e-7)
    with pytest.raises(ValueError, match='shape'):
        fun(1600., 5e7, jnp.ones(len(module.COMPONENTS)+1))
    with pytest.raises(ValueError, match='scalar'):
        fun(jnp.array([1600.]), 5e7, jnp.ones(len(module.COMPONENTS)))


def test_available_native_endpoint_properties(phase):
    name, _, fun, _, grad, dtt = phase
    for row in REFERENCE['endpoints']:
        if row['phase'] != name:
            continue
        T, P, n = row['T_K'], row['P_Pa'], jnp.array(row['n_mol'])
        dt, dp, _ = grad(T, P, n)
        values = (fun(T, P, n), -dt, -T*dtt(T, P, n), dp)
        for value, key, tolerance in zip(values, ('g_J', 's_J_K', 'cp_J_K', 'v_m3'),
                                          (5e-9, 1e-9, 2e-8, 1e-17)):
            if row[key] is not None and (key == 'g_J' or row['derivative_reference']):
                np.testing.assert_allclose(value, row[key], atol=tolerance, rtol=0)


@pytest.mark.parametrize('module', [feldspar, spinel, oxides, pyroxene])
def test_element_ledger_against_independent_oxide_stoichiometry(module):
    data = json.loads((ROOT/'examples/m2_solid_mixing/parameters.json').read_text())
    name = {feldspar: 'plagioclase', spinel: 'spinel', oxides: 'rhm-oxide', pyroxene: 'clinopyroxene'}[module]
    matrix = np.array(data['models'][name]['endmember_oxide_moles']).T
    indices = [data['oxide_order'].index(x.lower()) for x in magma.DRY11_OXIDES]
    expected = np.asarray(magma.element_matrix(magma.DRY11_OXIDES))@matrix[indices]
    np.testing.assert_array_equal(magma.element_matrix(module.COMPONENTS), expected)


@pytest.mark.parametrize('phase', ['clinopyroxene', 'orthopyroxene'])
def test_pyroxene_restrictions_and_signed_physical_faces(phase):
    fun = jax.jit(partial(pyroxene.gibbs, phase=phase))
    old = jax.jit(partial(magma.pyroxene_gibbs, phase=phase))
    for n in ([.3, 1.2, .5], [1., .8, .2], [0., 2., 0.], [0., 0., 2.]):
        ca, mg, fe = np.array(n)/(sum(n)/2)
        full = jnp.array([ca-fe, 1-ca, fe, 0., 0., 0., 0.])*(sum(n)/2)
        np.testing.assert_allclose(fun(1600., 5e7, full), old(1600., 5e7, jnp.array(n)), atol=4e-9, rtol=0)
    # Ca-Tschermak and aegirine have signed formal endmember coefficients.
    for n in ([0., 0., 0., 1., -1., 1., 0.], [0., 0., 0., -1., 1., 0., 1.]):
        assert np.isfinite(fun(1600., 5e7, jnp.array(n)))
    jax.clear_caches()


def test_spinel_signed_fe2_free_faces():
    fun = jax.jit(spinel.gibbs)
    for n in ([1., -1., 0., 1., 0.], [0., -1., 1., 1., 0.], [0., 0., .5, .5, 0.]):
        assert np.isfinite(fun(1600., 5e7, jnp.array(n)))
    jax.clear_caches()


@pytest.mark.parametrize('name', ['rutile', 'corundum'])
def test_pure_oxides_against_native(name):
    fun = getattr(oxides, name+'_gibbs')
    evaluate = jax.jit(lambda t, p: jnp.array([fun(t, p), -jax.grad(fun)(t, p),
                                             -t*jax.grad(jax.grad(fun))(t, p), jax.grad(fun, 1)(t, p)]))
    for row in REFERENCE['properties']:
        if row['phase'] == name:
            expected = [row[k] for k in ('g_J', 's_J_K', 'cp_J_K', 'v_m3')]
            np.testing.assert_allclose(evaluate(row['T_K'], row['P_Pa']), expected, atol=2e-9, rtol=3e-13)
    np.testing.assert_allclose(oxides.standard_gibbs(1600., 5e7)[3]-oxides.corundum_gibbs(1600., 5e7), 20000.)


def test_coefficient_extraction_is_reproducible():
    path = ROOT/'tests/reference/extract_dry_solid_coefficients.py'
    spec = importlib.util.spec_from_file_location('extract_dry_solids', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.render() == module.DESTINATION.read_text()


def test_invalid_phase_names():
    for module in (feldspar, pyroxene):
        with pytest.raises(ValueError, match='phase must'):
            module.gibbs(1600., 5e7, jnp.ones(len(module.COMPONENTS)), phase='unknown')


def test_rhm_pure_reference_selects_global_order_branch():
    # Independent one-dimensional source formula and dense bracketed roots.
    # At 1730 K/500 bar, s=0 is locally stable but is NOT the minimum.
    from scipy.optimize import brentq
    from scipy.special import xlogy

    reference = jax.jit(oxides._REFERENCE)
    pure = jax.jit(oxides.gibbs)
    derivative = jax.jit(jax.grad(pure, (0, 1)))
    dtt = jax.jit(jax.grad(jax.grad(pure, 0), 0))
    standard_derivative = jax.jit(jax.grad(lambda t, p: oxides.standard_gibbs(t, p)[0], (0, 1)))
    standard_dtt = jax.jit(jax.grad(jax.grad(lambda t, p: oxides.standard_gibbs(t, p)[0], 0), 0))
    for P in (1e5, 5e7, 5e8):
        a, b = 17477.+.010758*(P/1e5-1), 3189.+.035089*(P/1e5-1)
        for T in (1700., 1720., 1730., 1740., 1760., 1800.):
            rt = 8.3143*T

            def slope(s):
                return rt*np.log((1+s)/(1-s))-2*a*s+b*(2*s-4*s**3)

            grid = np.linspace(1e-10, 1-1e-12, 2001)
            signs = slope(grid)
            roots = [0.]+[brentq(slope, lo, hi, xtol=1e-14) for lo, hi, left, right in
                           zip(grid[:-1], grid[1:], signs[:-1], signs[1:]) if left*right < 0]
            energies = [a*(1-s*s)+b*s*s*(1-s*s)+rt*(xlogy(1+s, 1+s)+xlogy(1-s, 1-s)-2*np.log(2)) for s in roots]
            np.testing.assert_allclose(reference(T, P)[0], min(energies), atol=1e-9, rtol=0)
            n = jnp.array([1., 0., 0., 0.])
            np.testing.assert_allclose(pure(T, P, n), oxides.standard_gibbs(T, P)[0], atol=5e-9, rtol=0)
            np.testing.assert_allclose(derivative(T, P, n), standard_derivative(T, P), atol=1e-10, rtol=0)
            np.testing.assert_allclose(dtt(T, P, n), standard_dtt(T, P), atol=1e-11, rtol=0)
    jax.clear_caches()
