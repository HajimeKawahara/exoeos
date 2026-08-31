"""Synthetic-table tests for the Marcum silicate-hydrogen EOS."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import (
    MarcumSilicateHydrogenEOS,
    MassDensityProvider,
    SilicateHydrogenState,
)


ENDMEMBER_FRACTIONS = np.linspace(0.0001, 4.0, 11) / 4.0
TEMPERATURES = np.arange(3000.0, 10000.0 + 50.0, 50.0)
PRESSURES_GPA = np.concatenate(
    (np.arange(1.0, 5.0, 1.0), np.arange(5.0, 800.0 + 5.0, 5.0))
)
FIELD_COUNT = 9


def _raw_fields(
    composition_index: float,
    temperature_index: float,
    pressure_index: float,
) -> np.ndarray:
    """Return values affine in all three native table coordinates."""

    return (
        np.arange(1.0, FIELD_COUNT + 1.0)
        + 0.1 * composition_index
        + 0.001 * temperature_index
        + 0.0001 * pressure_index
    )


@pytest.fixture(scope="module")
def native_fields() -> jax.Array:
    composition_indices = jnp.arange(11.0)[:, None, None, None]
    temperature_indices = jnp.arange(141.0)[None, :, None, None]
    pressure_indices = jnp.arange(164.0)[None, None, :, None]
    field_indices = jnp.arange(1.0, FIELD_COUNT + 1.0)[None, None, None, :]
    fields = (
        field_indices
        + 0.1 * composition_indices
        + 0.001 * temperature_indices
        + 0.0001 * pressure_indices
    )
    missing = (
        jnp.asarray(TEMPERATURES)[None, :, None] > 6000.0
    ) & (jnp.asarray(PRESSURES_GPA)[None, None, :] < 5.0)
    return jnp.where(missing[..., None], jnp.nan, fields)


@pytest.fixture(scope="module")
def eos(native_fields: jax.Array) -> MarcumSilicateHydrogenEOS:
    return MarcumSilicateHydrogenEOS(native_fields)


def _composition(index: int) -> jax.Array:
    hydrogen_endmember_fraction = ENDMEMBER_FRACTIONS[index]
    return jnp.asarray(
        [1.0 - hydrogen_endmember_fraction, hydrogen_endmember_fraction]
    )


def _expected_state(pressure: float, raw_fields: np.ndarray) -> np.ndarray:
    conversions = np.asarray(
        [1.0e3, 1.0e6, 1.0e3, 1.0e-5, 1.0e3, 1.0e9, 1.0e3, 1.0, 1.0]
    )
    return np.concatenate(([pressure], raw_fields * conversions))


def _assert_all_nan(state: SilicateHydrogenState) -> None:
    assert all(bool(jnp.isnan(leaf)) for leaf in jax.tree_util.tree_leaves(state))


def test_state_tp_converts_an_exact_grid_point_to_si(
    eos: MarcumSilicateHydrogenEOS,
) -> None:
    state = eos.state_tp(4000.0, 100.0e9, _composition(5))
    expected = _expected_state(100.0e9, _raw_fields(5.0, 20.0, 23.0))

    assert isinstance(state, SilicateHydrogenState)
    assert jnp.allclose(jnp.asarray(state), jnp.asarray(expected), rtol=1.0e-12)


def test_state_tp_interpolates_trilinearly_on_native_coordinates(
    eos: MarcumSilicateHydrogenEOS,
) -> None:
    hydrogen_endmember_fraction = np.mean(ENDMEMBER_FRACTIONS[5:7])
    composition = jnp.asarray(
        [1.0 - hydrogen_endmember_fraction, hydrogen_endmember_fraction]
    )

    state = eos.state_tp(4025.0, 102.5e9, composition)
    expected = _expected_state(
        102.5e9,
        _raw_fields(5.5, 20.5, 23.5),
    )

    assert jnp.allclose(jnp.asarray(state), jnp.asarray(expected), rtol=1.0e-12)


def test_state_aliases_and_derived_properties(
    eos: MarcumSilicateHydrogenEOS,
) -> None:
    state = eos.state_tp(4000.0, 100.0e9, _composition(5))

    assert state.P == state.pressure
    assert state.rho == state.mass_density
    assert state.h == state.specific_enthalpy
    assert state.s == state.specific_entropy
    assert state.alpha == state.thermal_expansion
    assert state.cp == state.specific_heat_capacity_cp
    assert state.Ks == state.adiabatic_bulk_modulus
    assert state.rho0 == state.reference_mass_density
    assert state.eta == state.compression_ratio
    assert state.gamma == state.gruneisen_parameter
    assert state.u == pytest.approx(state.h - state.P / state.rho)
    assert state.specific_internal_energy == pytest.approx(state.u)
    assert state.nabla_ad == pytest.approx(
        state.alpha * state.P / (state.rho * state.cp)
    )
    assert state.adiabatic_gradient == pytest.approx(state.nabla_ad)


def test_eos_satisfies_the_mass_density_provider_contract(
    eos: MarcumSilicateHydrogenEOS,
) -> None:
    provider: MassDensityProvider = eos
    temperature = 4000.0
    pressure = 100.0e9
    composition = _composition(5)

    assert provider.endmembers == ("MgSiO3", "MgSiO3H4")
    assert jnp.allclose(provider.molar_masses, jnp.asarray([0.10039, 0.104422]))
    assert jnp.allclose(eos.endmember_fraction_grid, ENDMEMBER_FRACTIONS)
    assert provider.mass_density_tp(
        temperature, pressure, composition
    ) == pytest.approx(
        eos.state_tp(temperature, pressure, composition).rho
    )


def test_exact_native_domain_bounds_are_in_range(
    eos: MarcumSilicateHydrogenEOS,
) -> None:
    lower = eos.state_tp(3000.0, 1.0e9, _composition(0))
    upper = eos.state_tp(10000.0, 800.0e9, _composition(10))

    assert jnp.all(jnp.isfinite(jnp.asarray(lower)))
    assert jnp.all(jnp.isfinite(jnp.asarray(upper)))


def test_ragged_low_pressure_boundary_does_not_poison_exact_endpoints(
    eos: MarcumSilicateHydrogenEOS,
) -> None:
    available_edge = eos.state_tp(6000.0, 4.0e9, _composition(5))
    first_high_pressure_cell = eos.state_tp(6050.0, 5.0e9, _composition(5))
    missing_cell = eos.state_tp(6050.0, 4.0e9, _composition(5))
    interpolation_requiring_missing_corner = eos.state_tp(
        6025.0,
        4.5e9,
        _composition(5),
    )
    available_edge_gradient = jax.grad(
        lambda temperature: eos.state_tp(
            temperature,
            4.0e9,
            _composition(5),
        ).rho
    )(6000.0)

    assert jnp.all(jnp.isfinite(jnp.asarray(available_edge)))
    assert jnp.isfinite(available_edge_gradient)
    assert jnp.all(jnp.isfinite(jnp.asarray(first_high_pressure_cell)))
    _assert_all_nan(missing_cell)
    _assert_all_nan(interpolation_requiring_missing_corner)


@pytest.mark.parametrize(
    ("temperature", "pressure", "composition"),
    (
        (2999.0, 100.0e9, _composition(5)),
        (10001.0, 100.0e9, _composition(5)),
        (4000.0, 0.99e9, _composition(5)),
        (4000.0, 801.0e9, _composition(5)),
        (4000.0, 100.0e9, jnp.asarray([1.0, 0.0])),
        (4000.0, 100.0e9, jnp.asarray([-0.01, 1.01])),
        (jnp.nan, 100.0e9, _composition(5)),
        (4000.0, jnp.inf, _composition(5)),
        (4000.0, 100.0e9, jnp.asarray([jnp.nan, jnp.nan])),
    ),
)
def test_out_of_bounds_and_nonfinite_queries_return_nan(
    eos: MarcumSilicateHydrogenEOS,
    temperature,
    pressure,
    composition,
) -> None:
    _assert_all_nan(eos.state_tp(temperature, pressure, composition))


def test_composition_is_not_implicitly_normalized(
    eos: MarcumSilicateHydrogenEOS,
) -> None:
    _assert_all_nan(eos.state_tp(4000.0, 100.0e9, jnp.asarray([0.25, 0.25])))


@pytest.mark.parametrize(
    ("temperature", "pressure", "composition"),
    (
        (jnp.asarray([4000.0]), 100.0e9, _composition(5)),
        (4000.0, jnp.asarray([100.0e9]), _composition(5)),
        (4000.0, 100.0e9, jnp.asarray([1.0])),
        (4000.0, 100.0e9, jnp.asarray([[0.5, 0.5]])),
    ),
)
def test_state_tp_rejects_non_scalar_states_and_wrong_composition_shapes(
    eos: MarcumSilicateHydrogenEOS,
    temperature,
    pressure,
    composition,
) -> None:
    with pytest.raises(ValueError):
        eos.state_tp(temperature, pressure, composition)


def test_constructor_rejects_a_wrong_table_shape(native_fields: jax.Array) -> None:
    with pytest.raises(ValueError, match="shape"):
        MarcumSilicateHydrogenEOS(native_fields[:-1])


def test_state_tp_supports_jit_vmap_grad_and_pytree_round_trip(
    eos: MarcumSilicateHydrogenEOS,
) -> None:
    composition = _composition(5)
    compiled = jax.jit(
        lambda model, temperature, pressure, fractions: model.state_tp(
            temperature,
            pressure,
            fractions,
        )
    )
    scalar_state = compiled(eos, 4025.0, 102.5e9, composition)
    batched_state = jax.jit(
        jax.vmap(eos.state_tp, in_axes=(0, 0, 0))
    )(
        jnp.asarray([4000.0, 4025.0, 4050.0]),
        jnp.asarray([100.0e9, 102.5e9, 105.0e9]),
        jnp.stack((_composition(4), _composition(5), _composition(6))),
    )
    leaves, tree_definition = jax.tree_util.tree_flatten(scalar_state)
    reconstructed_state = jax.tree_util.tree_unflatten(tree_definition, leaves)
    model_leaves, model_definition = jax.tree_util.tree_flatten(eos)
    reconstructed_model = jax.tree_util.tree_unflatten(
        model_definition,
        model_leaves,
    )
    temperature_gradient = jax.grad(
        lambda temperature: eos.mass_density_tp(
            temperature,
            102.5e9,
            composition,
        )
    )(4025.0)
    pressure_gradient = jax.grad(
        lambda pressure: eos.mass_density_tp(4025.0, pressure, composition)
    )(102.5e9)
    midpoint_fraction = jnp.asarray(np.mean(ENDMEMBER_FRACTIONS[5:7]))
    composition_gradient = jax.grad(
        lambda fraction: eos.mass_density_tp(
            4025.0,
            102.5e9,
            jnp.asarray([1.0 - fraction, fraction]),
        )
    )(midpoint_fraction)

    assert isinstance(scalar_state, SilicateHydrogenState)
    assert isinstance(batched_state, SilicateHydrogenState)
    assert isinstance(reconstructed_state, SilicateHydrogenState)
    assert isinstance(reconstructed_model, MarcumSilicateHydrogenEOS)
    assert len(leaves) == len(SilicateHydrogenState._fields)
    assert batched_state.mass_density.shape == (3,)
    assert jnp.all(jnp.isfinite(jnp.asarray(batched_state)))
    assert all(
        jnp.array_equal(actual, expected)
        for actual, expected in zip(reconstructed_state, scalar_state)
    )
    assert jnp.isfinite(temperature_gradient)
    assert jnp.isfinite(pressure_gradient)
    assert jnp.isfinite(composition_gradient)
    assert temperature_gradient != 0.0
    assert pressure_gradient != 0.0
    assert composition_gradient != 0.0


def test_float32_table_and_inputs_produce_float32_state(
    native_fields: jax.Array,
) -> None:
    float32_eos = MarcumSilicateHydrogenEOS(native_fields.astype(jnp.float32))
    state = float32_eos.state_tp(
        jnp.asarray(4025.0, dtype=jnp.float32),
        jnp.asarray(102.5e9, dtype=jnp.float32),
        _composition(5).astype(jnp.float32),
    )

    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree_util.tree_leaves(state))
