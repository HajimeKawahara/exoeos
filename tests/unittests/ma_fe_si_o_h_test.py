"""Acceptance checks for the scalar alloy extension with ideal H dilution."""

import json
from pathlib import Path
import runpy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import MaFeSiOLiquid, MaFeSiOHLiquid, solution_state, total_gex_RT


ROOT = Path(__file__).resolve().parents[2]
DRY_REFERENCE = json.loads((ROOT / "tests/reference/fe_si_o_ma2001.json").read_text())


@pytest.mark.parametrize("hydrogen_fraction", [0.0, 0.2, 0.8, 1 - 1e-8])
def test_dry_and_h_dilution_match_independent_dry_reference(hydrogen_fraction):
    model = MaFeSiOHLiquid()
    for reference in DRY_REFERENCE["states"]:
        dry_fraction = 1 - hydrogen_fraction
        x = np.append(dry_fraction * np.asarray(reference["x"]), hydrogen_fraction)
        model.validate_state(reference["T_K"], reference["P_Pa"], x)
        state = solution_state(model, reference["T_K"], reference["P_Pa"], x)
        np.testing.assert_allclose(state.gex_RT, dry_fraction * reference["gex_formal_RT"], rtol=5e-12, atol=5e-12)
        np.testing.assert_allclose(state.lngamma, reference["ln_gamma_formal"] + [0.0], rtol=5e-12, atol=5e-12)


@pytest.mark.parametrize("amounts", [[1.7, 0.2, 0.1, 0.3], [0.9, 0.06, 0.04, 0.6]])
def test_extensive_energy_derivatives_euler_gibbs_duhem_and_finite_differences(amounts):
    model = MaFeSiOHLiquid()
    n = jnp.asarray(amounts)
    temperature, pressure = 2350.0, 1e5
    excess = lambda values: total_gex_RT(model, temperature, pressure, values)
    # Standards are external; this explicit ideal term exercises all four x_i.
    full_mixing = lambda values: excess(values) + jnp.sum(values * jnp.log(values / values.sum()))
    for energy in (excess, full_mixing):
        chemical_potentials = jax.grad(energy)(n)
        hessian = jax.hessian(energy)(n)
        np.testing.assert_allclose(energy(7 * n), 7 * energy(n), rtol=5e-12, atol=5e-12)
        np.testing.assert_allclose(n @ chemical_potentials, energy(n), rtol=5e-12, atol=5e-12)
        np.testing.assert_allclose(hessian, hessian.T, rtol=0, atol=5e-12)
        np.testing.assert_allclose(n @ hessian, 0, rtol=0, atol=5e-12)
        step = 1e-4
        finite_difference = []
        for direction in step * np.eye(4):
            values = [energy(n + k * direction) for k in (-2, -1, 1, 2)]
            finite_difference.append((values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step))
        np.testing.assert_allclose(chemical_potentials, finite_difference, rtol=2e-9, atol=2e-9)
    x = n / n.sum()
    state = solution_state(model, temperature, pressure, x)
    np.testing.assert_allclose(jax.grad(full_mixing)(n), jnp.log(x) + state.lngamma, rtol=5e-12, atol=5e-12)
    # Independent dry extensive construction contains no n_H dependence.
    np.testing.assert_allclose(excess(n), total_gex_RT(model.dry_model, temperature, pressure, n[:3]), rtol=5e-12, atol=5e-12)
    np.testing.assert_allclose(jax.hessian(excess)(n)[3, :], 0, rtol=0, atol=5e-12)


def test_jit_vmap_parameter_derivatives_and_ideal_limit():
    model = MaFeSiOHLiquid()
    compositions = jnp.asarray([[0.68, 0.08, 0.04, 0.2], [0.51, 0.06, 0.03, 0.4]])
    evaluate = jax.jit(jax.vmap(solution_state, in_axes=(None, None, None, 0)))
    actual = evaluate(model, 2350.0, 1e5, compositions)
    for i, x in enumerate(compositions):
        expected = solution_state(model.dry_model, 2350.0, 1e5, x[:3] / x[:3].sum())
        np.testing.assert_allclose(actual.lngamma[i], jnp.append(expected.lngamma, 0.0), rtol=5e-12, atol=5e-12)
    ideal = evaluate(MaFeSiOHLiquid(MaFeSiOLiquid(jnp.zeros(3))), 2350.0, 1e5, compositions)
    np.testing.assert_array_equal(ideal.gex_RT, np.zeros(2))
    np.testing.assert_array_equal(ideal.lngamma, np.zeros((2, 4)))
    derivative = jax.jacfwd(lambda m: solution_state(m, 2350.0, 1e5, compositions[0]).lngamma)(model)
    assert derivative.dry_model.interaction_K.shape == (4, 3)
    np.testing.assert_allclose(derivative.dry_model.interaction_K[3], 0, rtol=0, atol=1e-16)
    pressure_derivative = jax.jacfwd(lambda p: solution_state(model, 2350.0, p, compositions[0]).lngamma)(1e5)
    np.testing.assert_array_equal(pressure_derivative, np.zeros(4))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_dtype_and_standard_conversion_preserve_h_potential(dtype):
    dry = MaFeSiOLiquid(jnp.asarray([12.41 * 1873, -16500, -5 * 1873], dtype=dtype))
    model = MaFeSiOHLiquid(dry)
    temperature, pressure = jnp.asarray(2350, dtype=dtype), jnp.asarray(1e5, dtype=dtype)
    x = jnp.asarray([0.68, 0.08, 0.04, 0.2], dtype=dtype)
    state = jax.jit(solution_state)(model, temperature, pressure, x)
    shift = jax.jit(model.standard_state_shift_RT)(temperature)
    assert state.lngamma.dtype == dtype
    assert state.gex_RT.dtype == dtype
    assert shift.dtype == dtype
    np.testing.assert_array_equal(shift, jnp.append(dry.standard_state_shift_RT(temperature), dtype(0)))
    # Arbitrary nonzero H standard remains unchanged; zero shift is not mu0_H=0.
    source_mu0 = jnp.asarray([-2, 3, 4, 1.7], dtype=dtype)
    source_gamma = jnp.append(solution_state(dry, temperature, pressure, x[:3] / x[:3].sum()).lngamma + dry.standard_state_shift_RT(temperature), dtype(0))
    source_mu = source_mu0 + jnp.log(x) + source_gamma
    formal_mu = source_mu0 + shift + jnp.log(x) + state.lngamma
    np.testing.assert_allclose(formal_mu, source_mu, rtol=3e-6, atol=3e-6)
    assert shift[3] == 0
    assert model.components == ("Fe", "Si", "O", "H")
    assert model.activity_basis == "mole_fraction"
    assert model.standard_state_convention == "symmetric"
    assert model.reference_model_id != dry.reference_model_id


@pytest.mark.parametrize("hydrogen_budget", [0.04, 0.4, 1.2])
def test_example_closes_finite_h_and_inert_he_with_synthetic_exchange(hydrogen_budget):
    run_reference = runpy.run_path(str(ROOT / "examples/fe_si_o_h_control.py"))["run_reference"]
    record = run_reference(hydrogen_budget)
    alloy = np.asarray(record["state"]["alloy_amounts_mol"])
    gas = np.asarray(record["state"]["gas_amounts_mol"])
    assert 0 < alloy[3] < hydrogen_budget
    assert 0 < gas[0] < hydrogen_budget / 2
    np.testing.assert_allclose(alloy[3] + 2 * gas[0], hydrogen_budget, rtol=0, atol=1e-15)
    assert gas[1] == 0.3
    assert abs(record["acceptance"]["exchange_residual_RT"]) < 1e-12
    assert record["acceptance"]["max_element_residual_mol"] < 1e-12


@pytest.mark.parametrize("x", [
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.4, 0.1, 0.5],
    [1e-20, 0.5, 0.0, 0.5],
    [0.8, 0.1, 0.05, -0.05],
    [0.8, 0.1, 0.05, 0.1],
    [0.8, 0.1, 0.05, np.nan],
    [0.8, 0.1, 0.05, np.inf],
])
def test_validation_rejects_unsupported_compositions(x):
    with pytest.raises(ValueError):
        MaFeSiOHLiquid().validate_state(2350.0, 1e5, x)


def test_shape_temperature_pressure_and_parameters_use_dry_validation():
    model = MaFeSiOHLiquid()
    x = [0.68, 0.08, 0.04, 0.2]
    for temperature, pressure in [(0, 1e5), (np.nan, 1e5), (2350, 0), (2350, np.inf)]:
        with pytest.raises(ValueError):
            model.validate_state(temperature, pressure, x)
    for temperature, pressure, composition in [([2350], 1e5, x), (2350, [1e5], x), (2350, 1e5, x[:3])]:
        with pytest.raises(ValueError):
            model.gex_RT(temperature, pressure, composition)
    with pytest.raises(ValueError):
        MaFeSiOHLiquid(MaFeSiOLiquid([np.nan, 0, 0])).validate_state(2350, 1e5, x)
    assert not np.isfinite(model.gex_RT(2350, 1e5, [0, 0, 0, 1]))
