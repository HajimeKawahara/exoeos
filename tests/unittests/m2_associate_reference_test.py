"""Finite species scalar, common gauges and matched interval certificates."""
import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import total_solution_gibbs_RT, total_solution_state
from exoeos.ma_interval import _Interval

PATH = Path(__file__).resolve().parents[2] / "examples/m2_material/associate_reference.py"
SPEC = importlib.util.spec_from_file_location("m2_associate_test_provider", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
T = 2173.15
UPPER = np.array([1., .02, .01, .04, .02, .002, .00002, .0001, .12, .00001,
                  .003, .00002, .0001, .005, .00001, .000001, .002, .000001])
LOWER = np.r_[.75, np.zeros(17)]


def test_common_gauge_preserves_every_atom_of_each_associate():
    gas = {name: -10. + i for i, name in enumerate(("O2", "Mg1", "Ca1", "Al1", "Cr1", "Ti1"))}
    host = np.arange(5.) - 12.
    mu0, receipt = M.associated_standards_rt(T, host, gas)
    elements = M.COMPONENTS[:10]
    gauge = dict(zip(elements, np.arange(10.) * .17))
    shifted_gas = {name: value + (2*gauge["O"] if name == "O2" else gauge[name[:-1]])
                   for name, value in gas.items()}
    shifted, _ = M.associated_standards_rt(T, host + [gauge[e] for e in elements[:5]], shifted_gas)
    expected = [sum(count*gauge[element] for element, count in formula.items()) for formula in M.FORMULAS]
    np.testing.assert_allclose(shifted-mu0, expected, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(mu0[:5], host)
    assert receipt["pure_to_gas_conversions"]["Cr"]["pure_phase"] == "Cr(cr)"
    assert sum(M.FORMULAS[M.COMPONENTS.index("Cr2O")].values()) == 3
    with pytest.raises(ValueError, match="2100"):
        M.associated_standards_rt(1900., host, gas)


def test_host_limit_and_independent_scalar_extensivity():
    model, _ = M.make_associated_model(T)
    x = np.r_[[.95, .001, .003, .04, .006], np.zeros(13)]
    assert float(model.gex_RT(T, 2.7e7, x)) == pytest.approx(float(model.host.gex_RT(T, 2.7e7, x[:5])), abs=1e-14)
    n = UPPER*.1
    n[0] = .95
    standards = np.arange(18.)*.23-12.
    energy = lambda values: total_solution_gibbs_RT(model, T, 2.7e7, values, standards)
    state = total_solution_state(model, T, 2.7e7, n, standards)
    derivative = jax.grad(energy)(n)
    np.testing.assert_allclose(state.mu_RT, derivative, atol=1e-12)
    assert float(energy(n*3.7)) == pytest.approx(float(energy(n))*3.7, abs=1e-12)
    assert np.dot(n, derivative) == pytest.approx(float(energy(n)), abs=1e-12)
    hessian = np.asarray(jax.hessian(energy)(n))
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-9)
    np.testing.assert_allclose(hessian@n, 0., atol=1e-9)
    jax.jit(energy)(n).block_until_ready()
    # Mg + O -> MgO is a finite atom-conserving reaction, not a new reservoir.
    direction = np.zeros(18)
    direction[[2, 5, 10]] = [-1., -1., 1.]
    formula = np.array([[row.get(e, 0.) for row in M.FORMULAS] for e in M.COMPONENTS[:10]])
    np.testing.assert_array_equal(formula@direction, 0.)
    step = 1e-8
    fd = (energy(n+step*direction)-energy(n-step*direction))/(2*step)
    assert float(fd) == pytest.approx(float(direction@derivative), abs=5e-6)


def test_pure_arithmetic_hessian_bound_matches_the_exact_selected_scalar():
    model, _ = M.make_associated_model(T)
    lower = M.associated_curvature_lower_bound(model, T, LOWER, UPPER)
    assert 1.45 < lower < 1.46
    x = UPPER*.1
    x[0] = 1-x[1:].sum()
    interval = M.associated_excess(T, np.asarray(model.host.host.dry_model.interaction_K),
                                   np.asarray(model.host.epsilon), model.additional_matrix,
                                   [_Interval(value, value) for value in x[1:]])
    value = float(model.gex_RT(T, 2.7e7, x))
    assert interval.lo <= value <= interval.hi
    energy = lambda solutes: total_solution_gibbs_RT(model, T, 2.7e7,
                jnp.concatenate((jnp.array([1-jnp.sum(solutes)]), solutes)), jnp.zeros(18))
    actual = np.linalg.eigvalsh(np.asarray(jax.hessian(energy)(x[1:])))[0]
    assert actual >= lower


def test_reference_cross_coefficients_and_invalid_input_are_explicit():
    model, receipt = M.make_associated_model(T)
    matrix = np.asarray(model.additional_matrix)
    assert matrix[4, 8] == -3.8
    assert -1.14 < matrix[3, 8] < -1.12
    assert matrix[6, 7] == -13072./T
    assert "not measured to be zero" in receipt["zero_terms"]
    for x in (np.ones(18), np.r_[1., np.zeros(16)], np.r_[np.nan, np.zeros(17)]):
        with pytest.raises(ValueError):
            model.validate_state(T, 2.7e7, x)
