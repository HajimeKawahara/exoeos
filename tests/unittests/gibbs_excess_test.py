"""Excess Gibbs-energy kernel contracts."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from exoeos import IdealSolution, SolutionState, solution_state, total_gex_RT


class _MargulesModel(NamedTuple):
    """Binary symmetric Margules model used as an analytic test double."""

    interaction: jax.Array

    def gex_RT(self, T, P, x):
        """Return ``A x_1 x_2``."""

        del T, P
        return self.interaction * x[0] * x[1]


class _StateDependentModel:
    """Temperature- and pressure-dependent analytic test double."""

    def gex_RT(self, T, P, x):
        """Return a state-scaled binary interaction."""

        return (T / 1000.0 + P / 1.0e5) * x[0] * x[1]


def test_ideal_solution_has_zero_excess_state() -> None:
    model = IdealSolution()
    temperature = 1200.0
    pressure = 2.0e5
    composition = jnp.asarray([0.25, 0.75])

    state = solution_state(model, temperature, pressure, composition)

    assert isinstance(state, SolutionState)
    assert model.activity_basis == "mole_fraction"
    assert model.standard_state_convention == "symmetric"
    assert jnp.allclose(model.gex_RT(temperature, pressure, composition), 0.0)
    assert jnp.allclose(
        total_gex_RT(model, temperature, pressure, composition),
        0.0,
    )
    assert jnp.allclose(state.reduced_excess_gibbs, 0.0)
    assert jnp.allclose(state.log_activity_coefficients, jnp.zeros(2))
    assert jnp.allclose(state.gex_RT, state.reduced_excess_gibbs)
    assert jnp.allclose(state.lngamma, state.log_activity_coefficients)
    assert jax.tree_util.tree_leaves(model) == []


def test_margules_state_matches_analytic_activity_coefficients() -> None:
    interaction = jnp.asarray(2.4)
    model = _MargulesModel(interaction)
    composition = jnp.asarray([0.3, 0.7])

    state = solution_state(model, 1000.0, 1.0e5, composition)
    expected_gex_RT = interaction * composition[0] * composition[1]
    expected_lngamma = interaction * jnp.asarray(
        [composition[1] ** 2, composition[0] ** 2]
    )

    assert jnp.allclose(state.gex_RT, expected_gex_RT)
    assert jnp.allclose(state.lngamma, expected_lngamma)
    assert jnp.allclose(state.gex_RT, jnp.dot(composition, state.lngamma))


def test_total_gex_is_extensive_in_component_amounts() -> None:
    model = _MargulesModel(jnp.asarray(2.4))
    amounts = jnp.asarray([0.6, 1.4])
    scaled_amounts = 3.0 * amounts

    total = total_gex_RT(model, 1000.0, 1.0e5, amounts)
    scaled_total = total_gex_RT(model, 1000.0, 1.0e5, scaled_amounts)

    assert jnp.allclose(scaled_total, 3.0 * total)


def test_solution_state_supports_jit_vmap_and_amount_hessian() -> None:
    model = _MargulesModel(jnp.asarray(2.4))
    composition = jnp.asarray([0.3, 0.7])
    evaluate = lambda T, P, x: solution_state(model, T, P, x)

    expected = evaluate(1000.0, 1.0e5, composition)
    compiled = jax.jit(evaluate)(1000.0, 1.0e5, composition)
    batched = jax.vmap(evaluate)(
        jnp.asarray([900.0, 1000.0]),
        jnp.asarray([1.0e5, 2.0e5]),
        jnp.asarray([[0.2, 0.8], [0.3, 0.7]]),
    )
    hessian = jax.hessian(
        lambda n: total_gex_RT(model, 1000.0, 1.0e5, n)
    )(composition)

    assert all(
        jnp.allclose(actual, reference)
        for actual, reference in zip(
            jax.tree_util.tree_leaves(compiled),
            jax.tree_util.tree_leaves(expected),
        )
    )
    assert batched.reduced_excess_gibbs.shape == (2,)
    assert batched.log_activity_coefficients.shape == (2, 2)
    assert jnp.allclose(hessian, hessian.T)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_solution_state_preserves_floating_dtype(dtype) -> None:
    state = solution_state(
        IdealSolution(),
        jnp.asarray(1000.0, dtype=dtype),
        jnp.asarray(1.0e5, dtype=dtype),
        jnp.asarray([0.3, 0.7], dtype=dtype),
    )

    assert all(leaf.dtype == dtype for leaf in jax.tree_util.tree_leaves(state))


def test_solution_state_promotes_to_model_parameter_dtype() -> None:
    model = _MargulesModel(jnp.asarray(2.4, dtype=jnp.float64))
    state = jax.jit(
        lambda value: solution_state(
            value,
            jnp.asarray(1000.0, dtype=jnp.float32),
            jnp.asarray(1.0e5, dtype=jnp.float32),
            jnp.asarray([0.3, 0.7], dtype=jnp.float32),
        )
    )(model)

    assert all(
        leaf.dtype == jnp.float64 for leaf in jax.tree_util.tree_leaves(state)
    )


def test_solution_state_differentiates_model_parameters() -> None:
    composition = jnp.asarray([0.3, 0.7])
    energy_gradient = jax.grad(
        lambda interaction: solution_state(
            _MargulesModel(interaction),
            1000.0,
            1.0e5,
            composition,
        ).gex_RT
    )(jnp.asarray(2.4))
    activity_gradient = jax.jacrev(
        lambda interaction: solution_state(
            _MargulesModel(interaction),
            1000.0,
            1.0e5,
            composition,
        ).lngamma
    )(jnp.asarray(2.4))

    assert jnp.allclose(energy_gradient, composition[0] * composition[1])
    assert jnp.allclose(
        activity_gradient,
        jnp.asarray([composition[1] ** 2, composition[0] ** 2]),
    )


def test_solution_state_forwards_temperature_and_pressure_derivatives() -> None:
    model = _StateDependentModel()
    composition = jnp.asarray([0.3, 0.7])
    temperature = 1200.0
    pressure = 2.0e5
    scale = temperature / 1000.0 + pressure / 1.0e5

    state = solution_state(model, temperature, pressure, composition)
    temperature_gradient = jax.grad(
        lambda value: solution_state(model, value, pressure, composition).gex_RT
    )(temperature)
    pressure_gradient = jax.grad(
        lambda value: solution_state(model, temperature, value, composition).gex_RT
    )(pressure)

    assert jnp.allclose(
        state.gex_RT,
        scale * composition[0] * composition[1],
    )
    assert jnp.allclose(
        state.lngamma,
        scale * jnp.asarray([composition[1] ** 2, composition[0] ** 2]),
    )
    assert jnp.allclose(
        temperature_gradient,
        composition[0] * composition[1] / 1000.0,
    )
    assert jnp.allclose(
        pressure_gradient,
        composition[0] * composition[1] / 1.0e5,
    )


@pytest.mark.parametrize(
    "temperature,pressure,composition",
    [
        (jnp.ones(2), 1.0e5, jnp.asarray([0.3, 0.7])),
        (1000.0, jnp.ones(2), jnp.asarray([0.3, 0.7])),
        (1000.0, 1.0e5, jnp.ones((1, 2))),
        (1000.0, 1.0e5, jnp.asarray(1.0)),
    ],
)
def test_solution_state_rejects_batched_input_shapes(
    temperature,
    pressure,
    composition,
) -> None:
    with pytest.raises(ValueError):
        solution_state(
            IdealSolution(),
            temperature,
            pressure,
            composition,
        )


def test_total_gex_rejects_vector_model_output() -> None:
    class _InvalidModel:
        def gex_RT(self, T, P, x):
            del T, P
            return x

    with pytest.raises(ValueError, match="must return a scalar"):
        total_gex_RT(_InvalidModel(), 1000.0, 1.0e5, jnp.asarray([0.3, 0.7]))


@pytest.mark.parametrize(
    "temperature,pressure,amounts",
    [
        (jnp.ones(2), 1.0e5, jnp.asarray([0.3, 0.7])),
        (1000.0, jnp.ones(2), jnp.asarray([0.3, 0.7])),
        (1000.0, 1.0e5, jnp.ones((1, 2))),
        (1000.0, 1.0e5, jnp.asarray(1.0)),
    ],
)
def test_total_gex_rejects_batched_input_shapes(
    temperature,
    pressure,
    amounts,
) -> None:
    with pytest.raises(ValueError):
        total_gex_RT(
            IdealSolution(),
            temperature,
            pressure,
            amounts,
        )
