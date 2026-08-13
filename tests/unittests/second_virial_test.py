"""Analytic and transformation contracts for the second-virial EOS."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from exoeos import SecondVirialEOS, TRhoState, state_tp, state_trho


GAS_CONSTANT = 8.31446261815324


@pytest.fixture
def second_virial_eos() -> SecondVirialEOS:
    return SecondVirialEOS(
        jnp.asarray([[1.0e-4, 2.0e-5], [2.0e-5, 8.0e-5]]),
    )


def test_second_virial_matches_analytic_helmholtz_state(
    second_virial_eos: SecondVirialEOS,
) -> None:
    temperature = 700.0
    molar_density = 50.0
    composition = jnp.asarray([0.4, 0.6])
    coefficients = second_virial_eos.coefficients
    mixture_coefficient = composition @ coefficients @ composition

    state = state_trho(
        second_virial_eos,
        temperature,
        molar_density,
        composition,
    )
    expected_alphar = molar_density * mixture_coefficient
    expected_mu_res_RT = 2.0 * molar_density * coefficients @ composition
    expected_Z = 1.0 + expected_alphar
    expected_lnphi = expected_mu_res_RT - jnp.log(expected_Z)

    assert jnp.allclose(
        second_virial_eos.alphar(temperature, molar_density, composition),
        expected_alphar,
    )
    assert jnp.allclose(state.alphar, expected_alphar)
    assert jnp.allclose(state.mu_res_RT, expected_mu_res_RT)
    assert jnp.allclose(state.Z, expected_Z)
    assert jnp.allclose(
        state.P,
        molar_density * GAS_CONSTANT * temperature * expected_Z,
    )
    assert jnp.allclose(state.lnphi, expected_lnphi)
    assert jnp.allclose(state.gres_RT, composition @ expected_lnphi)


def test_state_tp_round_trips_a_density_state(
    second_virial_eos: SecondVirialEOS,
) -> None:
    temperature = 700.0
    molar_density = 50.0
    composition = jnp.asarray([0.4, 0.6])
    reference = state_trho(
        second_virial_eos,
        temperature,
        molar_density,
        composition,
    )

    recovered = state_tp(
        second_virial_eos,
        temperature,
        reference.P,
        composition,
    )
    explicit_vapor = state_tp(
        second_virial_eos,
        temperature,
        reference.P,
        composition,
        phase="vapor",
    )

    assert isinstance(recovered, TRhoState)
    assert all(
        jnp.allclose(actual, expected)
        for actual, expected in zip(
            jax.tree_util.tree_leaves(recovered),
            jax.tree_util.tree_leaves(reference),
        )
    )
    assert all(
        jnp.allclose(actual, expected)
        for actual, expected in zip(
            jax.tree_util.tree_leaves(explicit_vapor),
            jax.tree_util.tree_leaves(recovered),
        )
    )


def test_zero_virial_coefficient_recovers_the_ideal_gas_limit() -> None:
    eos = SecondVirialEOS(jnp.zeros((2, 2)))
    temperature = 500.0
    pressure = 1.2e5
    composition = jnp.asarray([0.25, 0.75])

    state = state_tp(eos, temperature, pressure, composition)

    assert jnp.allclose(state.rho, pressure / (GAS_CONSTANT * temperature))
    assert jnp.allclose(state.P, pressure)
    assert jnp.allclose(state.Z, 1.0)
    assert jnp.allclose(state.alphar, 0.0)
    assert jnp.allclose(state.lnphi, jnp.zeros(2))
    assert jnp.allclose(state.gres_RT, 0.0)


def test_state_tp_selects_the_stable_vapor_root_for_negative_B() -> None:
    coefficient = -1.0e-4
    eos = SecondVirialEOS(jnp.asarray([[coefficient]]))
    temperature = 500.0
    ideal_density = 1.0e3
    pressure = GAS_CONSTANT * temperature * ideal_density

    state = state_tp(eos, temperature, pressure, jnp.asarray([1.0]))
    discriminant = 1.0 + 4.0 * coefficient * ideal_density
    expected_density = 2.0 * ideal_density / (1.0 + jnp.sqrt(discriminant))

    assert jnp.allclose(state.rho, expected_density)
    assert jnp.allclose(state.P, pressure)
    assert 1.0 + 2.0 * coefficient * state.rho > 0.0


@pytest.mark.parametrize(
    "temperature,pressure,composition",
    [
        (jnp.ones(2), 1.0e5, jnp.asarray([0.4, 0.6])),
        (700.0, jnp.ones(2), jnp.asarray([0.4, 0.6])),
        (700.0, 1.0e5, jnp.ones((1, 2))),
    ],
)
def test_state_tp_rejects_batched_input_shapes(
    second_virial_eos: SecondVirialEOS,
    temperature,
    pressure,
    composition,
) -> None:
    with pytest.raises(ValueError):
        state_tp(second_virial_eos, temperature, pressure, composition)


def test_second_virial_rejects_invalid_shapes_and_phase(
    second_virial_eos: SecondVirialEOS,
) -> None:
    with pytest.raises(ValueError):
        SecondVirialEOS(jnp.ones((2, 3)))
    with pytest.raises(ValueError):
        second_virial_eos.alphar(700.0, 50.0, jnp.asarray([1.0]))
    with pytest.raises(ValueError, match="phase"):
        state_tp(
            second_virial_eos,
            700.0,
            1.0e5,
            jnp.asarray([0.4, 0.6]),
            phase="solid",
        )


def test_state_tp_supports_jit_vmap_and_grad(
    second_virial_eos: SecondVirialEOS,
) -> None:
    composition = jnp.asarray([0.4, 0.6])
    compiled = jax.jit(state_tp, static_argnames=("phase",))(
        second_virial_eos,
        700.0,
        2.0e5,
        composition,
        phase="vapor",
    )
    batched = jax.vmap(
        lambda T, P, x: state_tp(second_virial_eos, T, P, x)
    )(
        jnp.asarray([600.0, 700.0]),
        jnp.asarray([1.0e5, 2.0e5]),
        jnp.asarray([[0.3, 0.7], [0.4, 0.6]]),
    )
    density_pressure_gradient = jax.grad(
        lambda P: state_tp(second_virial_eos, 700.0, P, composition).rho
    )(2.0e5)
    mixture_coefficient = (
        composition @ second_virial_eos.coefficients @ composition
    )
    expected_gradient = 1.0 / (
        GAS_CONSTANT
        * 700.0
        * (1.0 + 2.0 * mixture_coefficient * compiled.rho)
    )

    assert isinstance(compiled, TRhoState)
    assert all(
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree_util.tree_leaves(compiled)
    )
    assert batched.rho.shape == (2,)
    assert batched.lnphi.shape == (2, 2)
    assert jnp.allclose(density_pressure_gradient, expected_gradient)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_second_virial_is_a_pytree_and_preserves_dtype(dtype) -> None:
    coefficients = jnp.asarray([[1.0e-4]], dtype=dtype)
    eos = SecondVirialEOS(coefficients)
    state = state_tp(
        eos,
        jnp.asarray(500.0, dtype=dtype),
        jnp.asarray(1.0e5, dtype=dtype),
        jnp.asarray([1.0], dtype=dtype),
    )

    leaves = jax.tree_util.tree_leaves(eos)
    assert len(leaves) == 1
    assert jnp.allclose(leaves[0], coefficients)
    assert all(leaf.dtype == dtype for leaf in jax.tree_util.tree_leaves(state))


def test_state_tp_differentiates_virial_coefficients() -> None:
    composition = jnp.asarray([0.4, 0.6])
    coefficients = jnp.asarray([[1.0e-4, 2.0e-5], [2.0e-5, 8.0e-5]])

    gradient = jax.jit(jax.grad(
        lambda values: state_tp(
            SecondVirialEOS(values),
            700.0,
            2.0e5,
            composition,
        ).gres_RT
    ))(coefficients)

    assert gradient.shape == coefficients.shape
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradient, gradient.T)
