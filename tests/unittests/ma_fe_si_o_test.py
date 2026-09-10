"""Acceptance checks for the completed, formally rebased Fe-Si-O model."""

from decimal import Decimal, localcontext
import json
from pathlib import Path
import runpy

import jax
from jax.experimental import enable_x64
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import MaFeSiOLiquid, solution_state, total_gex_RT


REFERENCE_DIRECTORY = Path(__file__).resolve().parents[1] / "reference"
REFERENCE = json.loads((REFERENCE_DIRECTORY / "fe_si_o_ma2001.json").read_text())
STATES = REFERENCE["states"]
INTERIORS = [state for state in STATES if state["id"].startswith("ternary_")]
REFERENCE_ACTIVITIES = runpy.run_path(
    str(REFERENCE_DIRECTORY / "generate_fe_si_o_ma2001.py")
)["activities"]
DEFAULT_INTERACTION_K = (12.41 * 1873, -16500.0, -5 * 1873)


@pytest.fixture(autouse=True)
def _use_float64():
    with enable_x64():
        yield


def _five_point_derivative(function, value, step):
    value = np.asarray(value)

    def difference(direction):
        return (
            function(value - 2 * direction) - 8 * function(value - direction)
            + 8 * function(value + direction) - function(value + 2 * direction)
        ) / (12 * step)

    if value.ndim == 0:
        return np.asarray(difference(step))
    return np.stack([difference(d) for d in step * np.eye(value.size)], axis=-1)


@pytest.mark.parametrize("reference", STATES, ids=lambda state: state["id"])
def test_native_state_matches_independent_reference(reference):
    model = MaFeSiOLiquid()
    temperature, pressure, x = reference["T_K"], reference["P_Pa"], reference["x"]
    model.validate_state(temperature, pressure, x)
    state = solution_state(model, temperature, pressure, x)
    shift = model.standard_state_shift_RT(temperature)
    np.testing.assert_allclose(state.gex_RT, reference["gex_formal_RT"], rtol=5e-12, atol=5e-12)
    np.testing.assert_allclose(state.lngamma, reference["ln_gamma_formal"], rtol=5e-12, atol=5e-12)
    np.testing.assert_allclose(shift, reference["mu0_formal_minus_source_over_RT"], rtol=5e-12, atol=5e-12)
    np.testing.assert_allclose(state.lngamma + shift, reference["ln_gamma_source"], rtol=5e-12, atol=5e-12)


@pytest.mark.parametrize("reference", INTERIORS, ids=lambda state: state["id"])
def test_extensivity_euler_gibbs_duhem_and_amount_hessian(reference):
    model = MaFeSiOLiquid()
    temperature, pressure = reference["T_K"], reference["P_Pa"]
    x = jnp.asarray(reference["x"])
    amounts = 2.3 * x
    total = lambda n: total_gex_RT(model, temperature, pressure, n)
    gradient = jax.grad(total)(amounts)
    hessian = jax.hessian(total)(amounts)
    np.testing.assert_allclose(total(7 * amounts), 7 * total(amounts), rtol=5e-12, atol=5e-12)
    np.testing.assert_allclose(amounts @ gradient, total(amounts), rtol=5e-12, atol=5e-12)
    np.testing.assert_allclose(gradient, reference["ln_gamma_formal"], rtol=5e-12, atol=5e-12)
    np.testing.assert_allclose(hessian, hessian.T, rtol=0, atol=5e-12)
    np.testing.assert_allclose(hessian @ amounts, 0, rtol=0, atol=5e-12)

    def tangent_activities(solutes):
        fractions = jnp.concatenate((jnp.atleast_1d(1 - solutes.sum()), solutes))
        return solution_state(model, temperature, pressure, fractions).lngamma

    jacobian = jax.jacfwd(tangent_activities)(x[1:])
    np.testing.assert_allclose(x @ jacobian, 0, rtol=0, atol=5e-12)


def test_full_activity_and_standard_shift_derivatives_match_finite_differences():
    temperature = jnp.asarray(2350.0)
    parameters = jnp.asarray(DEFAULT_INTERACTION_K)
    solutes = jnp.asarray([0.1, 0.05])

    def evaluate(temp, interaction, fractions):
        model = MaFeSiOLiquid(interaction)
        x = jnp.concatenate((jnp.atleast_1d(1 - fractions.sum()), fractions))
        state = solution_state(model, temp, 1e5, x)
        return jnp.concatenate((state.lngamma, model.standard_state_shift_RT(temp)))

    cases = [
        (lambda v: evaluate(v, parameters, solutes), temperature, 0.1, 1e-12),
        (lambda v: evaluate(temperature, v, solutes), parameters, 0.1, 1e-12),
        (lambda v: evaluate(temperature, parameters, v), solutes, 1e-4, 2e-9),
    ]
    for function, value, step, tolerance in cases:
        analytic = jax.jacfwd(function)(value)
        numerical = _five_point_derivative(function, value, step)
        np.testing.assert_allclose(analytic, numerical, rtol=2e-8, atol=tolerance)


def test_model_pytree_and_ignored_pressure_derivatives():
    model = MaFeSiOLiquid()
    x = jnp.asarray([0.85, 0.1, 0.05])
    evaluate = lambda value: solution_state(value, 2350.0, 1e5, x).lngamma
    model_derivative = jax.jacfwd(evaluate)(model)
    parameter_derivative = jax.jacfwd(lambda values: evaluate(MaFeSiOLiquid(values)))(model.interaction_K)
    assert model_derivative.interaction_K.shape == (3, 3)
    np.testing.assert_allclose(model_derivative.interaction_K, parameter_derivative, rtol=5e-12, atol=5e-12)
    pressure_derivative = jax.jacfwd(lambda pressure: solution_state(model, 2350.0, pressure, x).lngamma)(1e5)
    np.testing.assert_array_equal(pressure_derivative, np.zeros(3))


def test_dynamic_model_jit_and_vmap_preserve_reference_states():
    model = MaFeSiOLiquid()
    temperatures = jnp.asarray([state["T_K"] for state in STATES])
    pressures = 1e5 * jnp.arange(1, len(STATES) + 1)
    compositions = jnp.asarray([state["x"] for state in STATES])
    evaluate = jax.jit(jax.vmap(solution_state, in_axes=(None, 0, 0, 0)))
    actual = evaluate(model, temperatures, pressures, compositions)
    expected = [state["ln_gamma_formal"] for state in STATES]
    np.testing.assert_allclose(actual.lngamma, expected, rtol=5e-12, atol=5e-12)
    np.testing.assert_allclose(
        actual.gex_RT, [state["gex_formal_RT"] for state in STATES], rtol=5e-12, atol=5e-12,
    )
    zero = evaluate(MaFeSiOLiquid(jnp.zeros(3)), temperatures, pressures, compositions)
    np.testing.assert_array_equal(zero.lngamma, np.zeros((len(STATES), 3)))
    np.testing.assert_array_equal(zero.gex_RT, np.zeros(len(STATES)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_preserves_explicit_floating_dtype(dtype):
    model = MaFeSiOLiquid(jnp.asarray(DEFAULT_INTERACTION_K, dtype=dtype))
    temperature = jnp.asarray(2350, dtype=dtype)
    pressure = jnp.asarray(1e5, dtype=dtype)
    x = jnp.asarray([0.85, 0.1, 0.05], dtype=dtype)
    state = jax.jit(solution_state)(model, temperature, pressure, x)
    assert state.gex_RT.dtype == dtype
    assert state.lngamma.dtype == dtype
    assert model.standard_state_shift_RT(temperature).dtype == dtype
    np.testing.assert_allclose(state.lngamma, INTERIORS[1]["ln_gamma_formal"], rtol=2e-6, atol=2e-6)


def test_parameter_dtype_promotes_the_state():
    model = MaFeSiOLiquid(jnp.asarray(DEFAULT_INTERACTION_K, dtype=jnp.float64))
    x = jnp.asarray([0.85, 0.1, 0.05], dtype=jnp.float32)
    model.validate_state(jnp.float32(2350), jnp.float32(1e5), x)
    state = jax.jit(solution_state)(
        model, jnp.float32(2350), jnp.float32(1e5), x,
    )
    assert state.gex_RT.dtype == jnp.float64
    assert state.lngamma.dtype == jnp.float64
    assert model.standard_state_shift_RT(jnp.float32(2350)).dtype == jnp.float64


def test_zero_interactions_and_model_conventions():
    model = MaFeSiOLiquid(jnp.zeros(3))
    state = solution_state(model, 1873.0, 1e5, [0.9, 0.07, 0.03])
    np.testing.assert_array_equal(state.lngamma, np.zeros(3))
    assert state.gex_RT == 0
    np.testing.assert_allclose(model.standard_state_shift_RT(1873), [0, -6.65, 4.29 - 16500 / 1873])
    assert model.components == ("Fe", "Si", "O")
    assert model.activity_basis == "mole_fraction"
    assert model.standard_state_convention == "symmetric"
    assert model.reference_model_id == REFERENCE["model_id"]
    assert model.reference_pressure_Pa == REFERENCE["reference_pressure_Pa"]


@pytest.mark.parametrize("interaction", [DEFAULT_INTERACTION_K, (21000.0, -15000.0, -8000.0)])
def test_standard_conversion_preserves_chemical_potentials_and_reactions(interaction):
    temperature = 2350.0
    x = [0.85, 0.1, 0.05]
    model = MaFeSiOLiquid(interaction)
    with localcontext() as context:
        context.prec = 60
        parameters = dict(REFERENCE["parameters"])
        parameters["epsilon_SiSi_at_reference_temperature"] = str(Decimal(str(interaction[0])) / 1873)
        parameters["epsilon_OO_times_temperature_K"] = str(interaction[1])
        parameters["epsilon_SiO_at_reference_temperature"] = str(Decimal(str(interaction[2])) / 1873)
        reference = REFERENCE_ACTIVITIES(parameters, Decimal(str(temperature)), [Decimal(str(v)) for v in x])
    source = np.asarray([float(v) for v in reference["ln_gamma_source"]] + [0.0, 0.0])
    state = solution_state(model, temperature, 1e5, x)
    formal = np.concatenate((state.lngamma, [0.0, 0.0]))
    shift = np.concatenate((model.standard_state_shift_RT(temperature), [0.0, 0.0]))
    # Species: Fe, Si, O in metal, followed by ideal FeO and SiO2 controls.
    mu0 = np.asarray([-2.0, 3.0, 4.0, 5.0, -7.0])
    log_x = np.log(x + [0.2, 0.8])
    source_mu = mu0 + log_x + source
    formal_mu = mu0 + shift + log_x + formal
    np.testing.assert_allclose(formal_mu, source_mu, rtol=5e-12, atol=5e-12)
    # Balanced reactions: 2 FeO + Si -> SiO2 + 2 Fe; FeO -> Fe + O.
    reactions = np.asarray([[2, -1, 0, -2, 1], [1, 0, 1, -1, 0]])
    np.testing.assert_allclose(reactions @ formal_mu, reactions @ source_mu, rtol=5e-12, atol=5e-12)
    assert np.all(np.abs(reactions @ shift) > 1)


@pytest.mark.parametrize("x", [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
def test_formal_pure_solute_scalar_has_no_supported_activity_vector(x):
    model = MaFeSiOLiquid()
    assert model.gex_RT(3000.0, 1e5, x) == 0
    assert np.any(~np.isfinite(solution_state(model, 3000.0, 1e5, x).lngamma))
    with pytest.raises(ValueError):
        model.validate_state(3000.0, 1e5, x)


@pytest.mark.parametrize("resident", [1, 2])
@pytest.mark.parametrize("iron_fraction", [1e-4, 1e-7])
def test_binary_approach_to_formal_pure_solute_limits(resident, iron_fraction):
    model = MaFeSiOLiquid()
    x = np.zeros(3)
    x[0], x[resident] = iron_fraction, 1 - iron_fraction
    state = solution_state(model, 3000.0, 1e5, x)
    interaction = DEFAULT_INTERACTION_K[resident - 1] / 3000
    np.testing.assert_allclose(state.gex_RT, interaction * iron_fraction * np.log(iron_fraction), rtol=2e-8, atol=2e-12)
    np.testing.assert_allclose(state.lngamma[resident], -interaction * iron_fraction, rtol=2e-8, atol=2e-12)


@pytest.mark.parametrize("x", [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.95, 0.0, 0.05]])
def test_pure_fe_and_zero_solute_edges_have_finite_amount_hessians(x):
    model = MaFeSiOLiquid()
    hessian = jax.hessian(lambda amounts: total_gex_RT(model, 3000.0, 1e5, amounts))(jnp.asarray(x))
    assert np.all(np.isfinite(hessian))
    np.testing.assert_allclose(hessian, hessian.T, rtol=0, atol=5e-12)


@pytest.mark.parametrize("temperature,pressure,x", [
    (0.0, 1e5, [0.8, 0.15, 0.05]),
    (np.nan, 1e5, [0.8, 0.15, 0.05]),
    (3000.0, 0.0, [0.8, 0.15, 0.05]),
    (3000.0, np.inf, [0.8, 0.15, 0.05]),
    (3000.0, 1e5, [np.nan, 0.15, 0.05]),
    (3000.0, 1e5, [0.9, 0.15, -0.05]),
    (3000.0, 1e5, [0.8, 0.1, 0.05]),
    (3000.0, 1e5, [0.0, 0.5, 0.5]),
    (3000.0, 1e5, [1e-20, 1.0, 0.0]),
    (3000.0, 1e5, [1e-20, 0.0, 1.0 - np.finfo(float).eps]),
])
def test_eager_validation_rejects_unsupported_states(temperature, pressure, x):
    with pytest.raises(ValueError):
        MaFeSiOLiquid().validate_state(temperature, pressure, x)


def test_static_shapes_and_nonfinite_parameters_are_rejected():
    with pytest.raises(ValueError):
        MaFeSiOLiquid([1.0, 2.0])
    model = MaFeSiOLiquid()
    for temperature, pressure, x in [
        ([3000.0], 1e5, [0.8, 0.15, 0.05]),
        (3000.0, [1e5], [0.8, 0.15, 0.05]),
        (3000.0, 1e5, [0.8, 0.2]),
        (3000.0, 1e5, [[0.8, 0.15, 0.05]]),
    ]:
        with pytest.raises(ValueError):
            model.gex_RT(temperature, pressure, x)
    with pytest.raises(ValueError):
        model.standard_state_shift_RT([3000.0])
    with pytest.raises(ValueError):
        MaFeSiOLiquid([np.nan, -16500.0, -9365.0]).validate_state(3000.0, 1e5, [0.8, 0.15, 0.05])
