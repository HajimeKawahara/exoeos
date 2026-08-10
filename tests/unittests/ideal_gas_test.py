"""Analytic and differentiability contracts for the ideal-gas backend."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from exoeos import IdealGas, ThermodynamicState


GAS_CONSTANT = 8.31446261815324
BOLTZMANN_CONSTANT = 1.380649e-23


@pytest.fixture
def ideal_gas() -> IdealGas:
    return IdealGas(
        molar_masses=jnp.asarray([2.0e-3, 28.0e-3]),
        molar_heat_capacities=jnp.asarray([28.0, 32.0]),
        reference_enthalpies=jnp.asarray([100.0, 400.0]),
        reference_entropies=jnp.asarray([10.0, 25.0]),
        reference_temperature=300.0,
        reference_pressure=1.0e5,
    )


def test_scalar_state_matches_analytic_ideal_mixture(
    ideal_gas: IdealGas,
) -> None:
    temperature = 600.0
    pressure = 2.0e5
    mole_fractions = jnp.asarray([0.25, 0.75])

    state = ideal_gas.state(temperature, pressure, mole_fractions)

    mean_molar_mass = jnp.sum(mole_fractions * jnp.asarray([2.0e-3, 28.0e-3]))
    heat_capacity_cp = jnp.sum(mole_fractions * jnp.asarray([28.0, 32.0]))
    expected_enthalpy = jnp.sum(
        mole_fractions
        * (
            jnp.asarray([100.0, 400.0])
            + jnp.asarray([28.0, 32.0]) * (temperature - 300.0)
        )
    )
    expected_entropy = (
        jnp.sum(
            mole_fractions
            * (
                jnp.asarray([10.0, 25.0])
                + jnp.asarray([28.0, 32.0]) * jnp.log(temperature / 300.0)
            )
        )
        - GAS_CONSTANT * jnp.log(pressure / 1.0e5)
        - GAS_CONSTANT * jnp.sum(mole_fractions * jnp.log(mole_fractions))
    )

    assert isinstance(state, ThermodynamicState)
    assert jnp.allclose(state.compressibility_factor, 1.0)
    assert jnp.allclose(
        state.mass_density,
        pressure * mean_molar_mass / (GAS_CONSTANT * temperature),
    )
    assert jnp.allclose(
        state.number_density,
        pressure / (BOLTZMANN_CONSTANT * temperature),
    )
    assert jnp.allclose(state.molar_enthalpy, expected_enthalpy)
    assert jnp.allclose(state.molar_entropy, expected_entropy)
    assert jnp.allclose(state.molar_heat_capacity_cp, heat_capacity_cp)
    assert jnp.allclose(
        state.molar_heat_capacity_cv,
        heat_capacity_cp - GAS_CONSTANT,
    )
    assert jnp.allclose(
        state.adiabatic_gradient,
        GAS_CONSTANT / heat_capacity_cp,
    )
    assert jnp.allclose(state.thermal_expansion, 1.0 / temperature)
    assert jnp.allclose(state.log_fugacity_coefficients, jnp.zeros(2))
    assert jnp.allclose(state.residual_gibbs, 0.0)
    assert jnp.allclose(state.residual_enthalpy, 0.0)

    assert jnp.allclose(state.Z, state.compressibility_factor)
    assert jnp.allclose(state.h, state.molar_enthalpy)
    assert jnp.allclose(state.s, state.molar_entropy)
    assert jnp.allclose(state.cp, state.molar_heat_capacity_cp)
    assert jnp.allclose(state.cv, state.molar_heat_capacity_cv)


def test_reference_state_includes_mixing_entropy_and_zero_limit(
    ideal_gas: IdealGas,
) -> None:
    mole_fractions = jnp.asarray([0.25, 0.75])
    mixed = ideal_gas.state(300.0, 1.0e5, mole_fractions)
    expected_mixing_entropy = -GAS_CONSTANT * jnp.sum(
        mole_fractions * jnp.log(mole_fractions)
    )

    assert jnp.allclose(
        mixed.molar_enthalpy,
        jnp.sum(mole_fractions * jnp.asarray([100.0, 400.0])),
    )
    assert jnp.allclose(
        mixed.molar_entropy,
        jnp.sum(mole_fractions * jnp.asarray([10.0, 25.0])) + expected_mixing_entropy,
    )

    pure = ideal_gas.state(300.0, 1.0e5, jnp.asarray([1.0, 0.0]))
    assert jnp.isfinite(pure.molar_entropy)
    assert jnp.allclose(pure.molar_enthalpy, 100.0)
    assert jnp.allclose(pure.molar_entropy, 10.0)


def test_state_broadcasts_thermodynamic_batches_and_species_axis(
    ideal_gas: IdealGas,
) -> None:
    temperatures = jnp.asarray([[300.0], [600.0]])
    pressures = jnp.asarray([1.0e5, 2.0e5, 3.0e5])
    composition = jnp.asarray([0.5, 0.5])

    state = ideal_gas.state(temperatures, pressures, composition)

    scalar_fields = (
        state.compressibility_factor,
        state.mass_density,
        state.number_density,
        state.molar_enthalpy,
        state.molar_entropy,
        state.molar_heat_capacity_cp,
        state.molar_heat_capacity_cv,
        state.adiabatic_gradient,
        state.residual_gibbs,
        state.residual_enthalpy,
        state.thermal_expansion,
    )
    assert all(value.shape == (2, 3) for value in scalar_fields)
    assert state.log_fugacity_coefficients.shape == (2, 3, 2)
    assert jnp.allclose(state.molar_heat_capacity_cp, 30.0)


def test_state_is_a_pytree_and_supports_jit_vmap_and_grad(
    ideal_gas: IdealGas,
) -> None:
    compiled_state = jax.jit(
        lambda temperature, pressure, composition: ideal_gas.state(
            temperature,
            pressure,
            composition,
        )
    )(600.0, 2.0e5, jnp.asarray([0.25, 0.75]))
    leaves, tree_definition = jax.tree_util.tree_flatten(compiled_state)
    reconstructed = jax.tree_util.tree_unflatten(tree_definition, leaves)

    assert isinstance(reconstructed, ThermodynamicState)
    assert leaves
    assert jnp.allclose(reconstructed.mass_density, compiled_state.mass_density)

    compiled_model_argument = jax.jit(
        lambda model: model.state(600.0, 2.0e5, jnp.asarray([0.25, 0.75])).cp
    )(ideal_gas)
    assert jnp.allclose(compiled_model_argument, 31.0)

    temperatures = jnp.asarray([400.0, 600.0])
    pressures = jnp.asarray([1.0e5, 2.0e5])
    compositions = jnp.asarray([[0.25, 0.75], [2.0 / 3.0, 1.0 / 3.0]])
    batched = jax.vmap(ideal_gas.state)(
        temperatures,
        pressures,
        compositions,
    )
    assert isinstance(batched, ThermodynamicState)
    assert batched.mass_density.shape == (2,)
    assert batched.log_fugacity_coefficients.shape == (2, 2)

    temperature = 600.0
    pressure = 2.0e5
    composition = jnp.asarray([0.25, 0.75])
    density = ideal_gas.state(temperature, pressure, composition).mass_density
    density_temperature_gradient = jax.grad(
        lambda value: ideal_gas.state(value, pressure, composition).mass_density
    )(temperature)
    density_pressure_gradient = jax.grad(
        lambda value: ideal_gas.state(temperature, value, composition).mass_density
    )(pressure)
    entropy_composition_gradient = jax.grad(
        lambda value: ideal_gas.state(temperature, pressure, value).molar_entropy
    )(composition)

    assert jnp.allclose(density_temperature_gradient, -density / temperature)
    assert jnp.allclose(density_pressure_gradient, density / pressure)
    assert jnp.all(jnp.isfinite(entropy_composition_gradient))


def test_state_preserves_explicit_float32_model_dtype() -> None:
    dtype = jnp.float32
    model = IdealGas(
        jnp.asarray([2.0e-3, 28.0e-3], dtype=dtype),
        jnp.asarray([28.0, 32.0], dtype=dtype),
    )
    state = model.state(
        jnp.asarray(600.0, dtype=dtype),
        jnp.asarray(2.0e5, dtype=dtype),
        jnp.asarray([0.25, 0.75], dtype=dtype),
    )

    assert all(leaf.dtype == dtype for leaf in jax.tree_util.tree_leaves(state))


def test_state_promotes_float16_to_keep_si_constants_representable() -> None:
    dtype = jnp.float16
    model = IdealGas(
        jnp.asarray([2.0e-3], dtype=dtype),
        jnp.asarray([28.0], dtype=dtype),
    )
    state = model.state(
        jnp.asarray(300.0, dtype=dtype),
        jnp.asarray(1.0e4, dtype=dtype),
        jnp.asarray([1.0], dtype=dtype),
    )

    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree_util.tree_leaves(state))
    assert jnp.isfinite(state.number_density)


def test_model_heat_capacities_are_differentiable_parameters() -> None:
    molar_masses = jnp.asarray([2.0e-3, 28.0e-3])
    composition = jnp.asarray([0.25, 0.75])

    def enthalpy(molar_heat_capacities):
        model = IdealGas(
            molar_masses,
            molar_heat_capacities,
            reference_enthalpies=jnp.zeros(2),
            reference_entropies=jnp.zeros(2),
            reference_temperature=300.0,
        )
        return model.state(500.0, 1.0e5, composition).molar_enthalpy

    gradient = jax.grad(enthalpy)(jnp.asarray([28.0, 32.0]))
    assert jnp.allclose(gradient, jnp.asarray([50.0, 150.0]))


@pytest.mark.parametrize(
    "molar_masses,molar_heat_capacities,reference_enthalpies,reference_entropies",
    [
        (jnp.ones(2), jnp.ones(3), None, None),
        (jnp.ones((1, 2)), jnp.ones(2), None, None),
        (jnp.ones(2), jnp.ones(2), jnp.ones(3), None),
        (jnp.ones(2), jnp.ones(2), None, jnp.ones(3)),
    ],
)
def test_constructor_rejects_incompatible_component_shapes(
    molar_masses,
    molar_heat_capacities,
    reference_enthalpies,
    reference_entropies,
) -> None:
    with pytest.raises(ValueError):
        IdealGas(
            molar_masses,
            molar_heat_capacities,
            reference_enthalpies=reference_enthalpies,
            reference_entropies=reference_entropies,
        )


def test_state_rejects_wrong_species_dimension(ideal_gas: IdealGas) -> None:
    with pytest.raises(ValueError):
        ideal_gas.state(300.0, 1.0e5, jnp.ones(3))
