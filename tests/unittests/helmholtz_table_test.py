"""Thermodynamic and differentiability contracts for potential EOS tables."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos.helmholtz_table import HelmholtzTable


GAS_CONSTANT = 2300.0
HEAT_CAPACITY_CV = 6100.0
REFERENCE_TEMPERATURE = 300.0
REFERENCE_ENTROPY = 50000.0
TEMPERATURES = np.asarray([120.0, 210.0, 490.0, 900.0, 1800.0])
MASS_DENSITIES = np.asarray([0.08, 0.3, 1.7, 6.0, 22.0, 90.0])


def _ideal_data():
    temperature, density = np.meshgrid(TEMPERATURES, MASS_DENSITIES, indexing="ij")
    pressure = density * GAS_CONSTANT * temperature
    energy = HEAT_CAPACITY_CV * temperature
    entropy = (
        HEAT_CAPACITY_CV * np.log(temperature / REFERENCE_TEMPERATURE)
        - GAS_CONSTANT * np.log(density)
        + REFERENCE_ENTROPY
    )
    return pressure, energy, entropy


@pytest.fixture(scope="module", params=["helmholtz", "thermodynamic", "responses"])
def ideal_table(request) -> HelmholtzTable:
    pressure, energy, entropy = _ideal_data()
    if request.param == "helmholtz":
        return HelmholtzTable.from_helmholtz(
            TEMPERATURES,
            MASS_DENSITIES,
            energy - TEMPERATURES[:, None] * entropy,
        )
    responses = {}
    if request.param == "responses":
        responses = {
            "dlnrho_dlnT_P": -np.ones_like(pressure),
            "dlnrho_dlnP_T": np.ones_like(pressure),
            "dlns_dlnT_P": (HEAT_CAPACITY_CV + GAS_CONSTANT) / entropy,
        }
    return HelmholtzTable.from_thermodynamic_data(
        TEMPERATURES, MASS_DENSITIES, pressure, energy, entropy, **responses
    )


@pytest.fixture(scope="module")
def nonlinear_table() -> HelmholtzTable:
    _, energy, entropy = _ideal_data()
    x = np.log(TEMPERATURES[:, None] / REFERENCE_TEMPERATURE)
    y = np.log(MASS_DENSITIES[None, :])
    reduced_perturbation = 300.0 * np.sin(1.3 * x) * np.cos(0.7 * y)
    return HelmholtzTable.from_helmholtz(
        TEMPERATURES,
        MASS_DENSITIES,
        energy - TEMPERATURES[:, None] * (entropy - reduced_perturbation),
    )


def test_ideal_gas_values_and_responses_on_nonuniform_grid(
    ideal_table: HelmholtzTable,
) -> None:
    temperatures = jnp.asarray([120.0, 175.0, 380.0, 750.0, 1800.0])
    densities = jnp.asarray([0.08, 0.2, 1.2, 15.0, 90.0])
    state = jax.jit(jax.vmap(ideal_table.state_trho))(temperatures, densities)
    entropy = (
        HEAT_CAPACITY_CV * jnp.log(temperatures / REFERENCE_TEMPERATURE)
        - GAS_CONSTANT * jnp.log(densities)
        + REFERENCE_ENTROPY
    )
    cp = HEAT_CAPACITY_CV + GAS_CONSTANT

    np.testing.assert_allclose(state.temperature, temperatures, rtol=1.0e-11)
    np.testing.assert_allclose(state.rho, densities, rtol=1.0e-11)
    np.testing.assert_allclose(
        state.P, densities * GAS_CONSTANT * temperatures, rtol=1.0e-10
    )
    np.testing.assert_allclose(state.u, HEAT_CAPACITY_CV * temperatures, rtol=1.0e-10)
    np.testing.assert_allclose(state.s, entropy, rtol=1.0e-10)
    np.testing.assert_allclose(
        state.a, temperatures * (HEAT_CAPACITY_CV - entropy), rtol=1.0e-10
    )
    np.testing.assert_allclose(state.cv, HEAT_CAPACITY_CV, rtol=1.0e-9)
    np.testing.assert_allclose(state.cp, cp, rtol=1.0e-9)
    np.testing.assert_allclose(state.dlnrho_dlnT_P, -1.0, rtol=1.0e-9)
    np.testing.assert_allclose(state.dlnrho_dlnP_T, 1.0, rtol=1.0e-9)
    np.testing.assert_allclose(state.dlns_dlnT_P, cp / entropy, rtol=1.0e-9)
    np.testing.assert_allclose(state.dlns_dlnP_T, -GAS_CONSTANT / entropy, rtol=1.0e-9)
    np.testing.assert_allclose(state.nabla_ad, GAS_CONSTANT / cp, rtol=1.0e-9)


def test_endpoint_derivatives_are_not_halved_by_clipping(
    ideal_table: HelmholtzTable,
) -> None:
    points = jnp.asarray(
        [
            [temperature, density]
            for temperature in (TEMPERATURES[0], TEMPERATURES[-1])
            for density in (MASS_DENSITIES[0], MASS_DENSITIES[-1])
        ]
    )
    derivative = jax.jit(
        jax.vmap(
            jax.grad(lambda point: ideal_table.specific_helmholtz(point[0], point[1]))
        )
    )(points)
    expected_entropy = (
        HEAT_CAPACITY_CV * jnp.log(points[:, 0] / REFERENCE_TEMPERATURE)
        - GAS_CONSTANT * jnp.log(points[:, 1])
        + REFERENCE_ENTROPY
    )
    np.testing.assert_allclose(derivative[:, 0], -expected_entropy, rtol=1.0e-10)
    np.testing.assert_allclose(
        derivative[:, 1], GAS_CONSTANT * points[:, 0] / points[:, 1], rtol=1.0e-10
    )


def test_nonideal_nodal_data_and_response_constraints_are_preserved() -> None:
    temperature, density = np.meshgrid(TEMPERATURES, MASS_DENSITIES, indexing="ij")
    pressure, energy, entropy = _ideal_data()
    # Add a = B*rho + D*T**2 + E*rho*T**2 to the ideal potential.
    density_coefficient, temperature_coefficient, cross_coefficient = 50.0, -0.01, 0.002
    pressure += density**2 * (density_coefficient + cross_coefficient * temperature**2)
    energy += density_coefficient * density - temperature**2 * (
        temperature_coefficient + cross_coefficient * density
    )
    entropy -= (
        2.0 * temperature * (temperature_coefficient + cross_coefficient * density)
    )
    cv = HEAT_CAPACITY_CV - 2.0 * temperature * (
        temperature_coefficient + cross_coefficient * density
    )
    pressure_temperature = (
        GAS_CONSTANT * density + 2.0 * cross_coefficient * density**2 * temperature
    )
    pressure_density = GAS_CONSTANT * temperature + 2.0 * density * (
        density_coefficient + cross_coefficient * temperature**2
    )
    cp = cv + temperature * pressure_temperature**2 / (density**2 * pressure_density)
    table = HelmholtzTable.from_thermodynamic_data(
        TEMPERATURES,
        MASS_DENSITIES,
        pressure,
        energy,
        entropy,
        dlnrho_dlnT_P=-temperature
        * pressure_temperature
        / (density * pressure_density),
        dlnrho_dlnP_T=pressure / (density * pressure_density),
        dlns_dlnT_P=cp / entropy,
    )
    state = jax.jit(jax.vmap(table.state_trho))(
        jnp.asarray(temperature.ravel()), jnp.asarray(density.ravel())
    )

    for actual, expected in (
        (state.P, pressure),
        (state.u, energy),
        (state.s, entropy),
        (state.a, energy - temperature * entropy),
        (state.cv, cv),
        (state.cp, cp),
        (state.pressure_temperature_derivative, pressure_temperature),
        (state.pressure_density_derivative, pressure_density),
    ):
        np.testing.assert_allclose(actual, expected.ravel(), rtol=1.0e-10)


@pytest.mark.parametrize("constrain_responses", [False, True])
def test_large_entropy_offset_preserves_finite_nodal_pressure_and_energy(
    constrain_responses: bool,
) -> None:
    temperature, density = np.meshgrid(TEMPERATURES, MASS_DENSITIES, indexing="ij")
    pressure, energy, _ = _ideal_data()
    entropy = np.full_like(pressure, 1.0e60)
    responses = {}
    if constrain_responses:
        responses = {
            "dlnrho_dlnT_P": -np.ones_like(pressure),
            "dlnrho_dlnP_T": np.ones_like(pressure),
            "dlns_dlnT_P": (HEAT_CAPACITY_CV + GAS_CONSTANT) / entropy,
        }
    table = HelmholtzTable.from_thermodynamic_data(
        TEMPERATURES, MASS_DENSITIES, pressure, energy, entropy, **responses
    )
    states = jax.jit(jax.vmap(table.state_trho))(
        jnp.asarray(temperature.ravel()), jnp.asarray(density.ravel())
    )

    # The small u/T contribution is lost in q=u/T-s, but its separately
    # constrained derivatives must still recover the finite P and u values.
    assert all(np.all(np.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(states))
    np.testing.assert_allclose(states.P, pressure.ravel(), rtol=1.0e-10)
    np.testing.assert_allclose(states.u, energy.ravel(), rtol=1.0e-10)
    np.testing.assert_allclose(states.s, entropy.ravel(), rtol=1.0e-14)


def test_potential_is_c2_at_cell_boundaries_and_anchor_changes(
    nonlinear_table: HelmholtzTable,
) -> None:
    def potential(log_coordinates):
        return nonlinear_table.specific_helmholtz(*jnp.exp(log_coordinates))

    def value_gradient_hessian(log_coordinates):
        value, gradient = jax.value_and_grad(potential)(log_coordinates)
        hessian = jax.hessian(potential)(log_coordinates)
        return jnp.concatenate((value[None], gradient, hessian.ravel()))

    evaluate = jax.jit(jax.vmap(value_gradient_hessian))
    for axis, grid, other_coordinate in (
        (0, TEMPERATURES, np.log(2.4)),
        (1, MASS_DENSITIES, np.log(620.0)),
    ):
        # The numerical potential anchor switches at each log-cell midpoint.
        crossings = np.concatenate((grid[1:-1], np.sqrt(grid[:-1] * grid[1:])))
        for knot in crossings:
            point = np.asarray([other_coordinate, other_coordinate])
            point[axis] = np.log(knot)
            offset = np.zeros(2)
            offset[axis] = 1.0e-8
            left, center, right = evaluate(
                jnp.asarray([point - offset, point, point + offset])
            )
            np.testing.assert_allclose(left, center, rtol=1.0e-6, atol=2.0e-3)
            np.testing.assert_allclose(right, center, rtol=1.0e-6, atol=2.0e-3)


def test_state_matches_direct_potential_derivatives_and_first_law(
    nonlinear_table: HelmholtzTable,
) -> None:
    temperature, density = 640.0, 2.4
    point = jnp.asarray([temperature, density])

    def potential(values):
        return nonlinear_table.specific_helmholtz(*values)

    gradient = jax.grad(potential)(point)
    hessian = jax.hessian(potential)(point)
    state = nonlinear_table.state_trho(*point)
    derivatives = jax.jacrev(
        lambda values: jnp.asarray(
            (
                nonlinear_table.state_trho(*values).s,
                nonlinear_table.state_trho(*values).u,
                nonlinear_table.state_trho(*values).P,
            )
        )
    )(point)
    entropy_derivatives, energy_derivatives, pressure_derivatives = derivatives

    np.testing.assert_allclose(state.s, -gradient[0], rtol=1.0e-12)
    np.testing.assert_allclose(state.P, density**2 * gradient[1], rtol=1.0e-12)
    np.testing.assert_allclose(state.cv, -temperature * hessian[0, 0], rtol=1.0e-12)
    np.testing.assert_allclose(state.u, state.a + temperature * state.s, rtol=1.0e-12)
    np.testing.assert_allclose(
        state.pressure_temperature_derivative, pressure_derivatives[0], rtol=1.0e-12
    )
    np.testing.assert_allclose(
        state.pressure_density_derivative, pressure_derivatives[1], rtol=1.0e-12
    )
    np.testing.assert_allclose(
        entropy_derivatives[1], -pressure_derivatives[0] / density**2, rtol=1.0e-12
    )
    np.testing.assert_allclose(
        energy_derivatives,
        temperature * entropy_derivatives + jnp.asarray([0.0, state.P / density**2]),
        rtol=1.0e-10,
    )

    isobaric_density_derivative = -pressure_derivatives[0] / pressure_derivatives[1]
    isobaric_entropy_derivative = entropy_derivatives @ jnp.asarray(
        [1.0, isobaric_density_derivative]
    )
    np.testing.assert_allclose(
        state.cp, temperature * isobaric_entropy_derivative, rtol=1.0e-11
    )
    np.testing.assert_allclose(
        state.dlnrho_dlnT_P,
        temperature / density * isobaric_density_derivative,
        rtol=1.0e-11,
    )
    np.testing.assert_allclose(
        state.dlnrho_dlnP_T, state.P / (density * pressure_derivatives[1]), rtol=1.0e-11
    )
    np.testing.assert_allclose(
        state.dlns_dlnT_P,
        temperature * isobaric_entropy_derivative / state.s,
        rtol=1.0e-11,
    )
    np.testing.assert_allclose(
        state.dlns_dlnP_T,
        state.P * entropy_derivatives[1] / (state.s * pressure_derivatives[1]),
        rtol=1.0e-11,
    )
    adiabatic_temperature_pressure_derivative = -entropy_derivatives[1] / (
        entropy_derivatives[0] * pressure_derivatives[1]
        - entropy_derivatives[1] * pressure_derivatives[0]
    )
    np.testing.assert_allclose(
        state.nabla_ad,
        state.P / temperature * adiabatic_temperature_pressure_derivative,
        rtol=1.0e-11,
    )


def test_model_and_state_pytree_round_trip_with_jit_and_vmap(
    nonlinear_table: HelmholtzTable,
) -> None:
    leaves, structure = jax.tree_util.tree_flatten(nonlinear_table)
    reconstructed = jax.tree_util.tree_unflatten(structure, leaves)
    evaluate = jax.jit(
        lambda table, temperature, density: table.state_trho(temperature, density)
    )
    state = evaluate(reconstructed, 620.0, 2.4)
    expected = nonlinear_table.state_trho(620.0, 2.4)
    state_leaves, state_structure = jax.tree_util.tree_flatten(state)
    restored_state = jax.tree_util.tree_unflatten(state_structure, state_leaves)
    batched = jax.jit(jax.vmap(reconstructed.state_trho))(
        jnp.asarray([300.0, 620.0]), jnp.asarray([0.6, 2.4])
    )

    assert isinstance(reconstructed, HelmholtzTable)
    assert leaves and all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert type(restored_state) is type(state)
    assert batched.P.shape == (2,)
    np.testing.assert_allclose(batched.P[1], state.P, rtol=1.0e-12)
    for actual, reference in zip(state_leaves, jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(actual, reference, rtol=1.0e-12)


def test_tp_inversion_matches_trho_and_implicit_derivatives_at_endpoints(
    nonlinear_table: HelmholtzTable,
) -> None:
    temperatures = jnp.asarray([TEMPERATURES[0], 620.0, TEMPERATURES[-1]])
    densities = jnp.asarray([MASS_DENSITIES[0], 2.4, MASS_DENSITIES[-1]])
    expected = jax.vmap(nonlinear_table.state_trho)(temperatures, densities)
    bounds = (MASS_DENSITIES[0], MASS_DENSITIES[-1])

    def solve(table, temperature, pressure):
        return table.state_tp(temperature, pressure, density_bounds=bounds)

    actual = jax.jit(jax.vmap(solve, in_axes=(None, 0, 0)))(
        nonlinear_table, temperatures, expected.P
    )

    def density_from_tp(point):
        return solve(nonlinear_table, point[0], point[1]).rho

    derivatives = jax.jit(jax.vmap(jax.grad(density_from_tp)))(
        jnp.stack((temperatures, expected.P), axis=1)
    )
    for result, reference in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)
    ):
        np.testing.assert_allclose(result, reference, rtol=1.0e-10)
    np.testing.assert_allclose(
        derivatives[:, 0],
        expected.dlnrho_dlnT_P * densities / temperatures,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        derivatives[:, 1],
        expected.dlnrho_dlnP_T * densities / expected.P,
        rtol=1.0e-10,
    )
    assert np.all(np.abs(derivatives) > 0.0)


def test_tp_invalid_queries_and_brackets_return_nan(
    nonlinear_table: HelmholtzTable,
) -> None:
    temperature, density = 620.0, 2.4
    pressure = float(nonlinear_table.state_trho(temperature, density).P)
    lower, upper = MASS_DENSITIES[0], MASS_DENSITIES[-1]
    queries = [
        (119.0, pressure, lower, upper),
        (1801.0, pressure, lower, upper),
        (np.nan, pressure, lower, upper),
        (temperature, -1.0, lower, upper),
        (temperature, 0.0, lower, upper),
        (temperature, np.nan, lower, upper),
        (temperature, np.inf, lower, upper),
        (temperature, 1.0e-12, lower, upper),
        (temperature, 1.0e15, lower, upper),
        (temperature, pressure, upper, lower),
        (temperature, pressure, density, density),
        (temperature, pressure, lower / 2.0, upper),
        (temperature, pressure, lower, upper * 2.0),
        (temperature, pressure, -1.0, upper),
        (temperature, pressure, np.nan, upper),
        (temperature, pressure, 4.0, 10.0),
        (temperature, pressure, 0.2, 1.0),
    ]

    def solve(query):
        return nonlinear_table.state_tp(query[0], query[1], density_bounds=query[2:])

    states = jax.jit(jax.vmap(solve))(jnp.asarray(queries))
    assert all(np.all(np.isnan(leaf)) for leaf in jax.tree_util.tree_leaves(states))


def test_tp_implicit_derivative_includes_table_coefficients(
    nonlinear_table: HelmholtzTable,
) -> None:
    temperature, density = 620.0, 2.4
    reference = nonlinear_table.state_trho(temperature, density)

    def scaled_table_density(scale):
        table = replace(
            nonlinear_table, derivatives=nonlinear_table.derivatives * scale
        )
        return table.state_tp(
            temperature,
            reference.P,
            density_bounds=(MASS_DENSITIES[0], MASS_DENSITIES[-1]),
        ).rho

    derivative = jax.jit(jax.grad(scaled_table_density))(1.0)
    # Scaling a by k scales pressure by k, hence drho/dk = -P/P_rho.
    np.testing.assert_allclose(
        derivative,
        -reference.P / reference.pressure_density_derivative,
        rtol=1.0e-10,
    )


def test_float32_ideal_states_and_closed_endpoint_derivatives() -> None:
    pressure, energy, entropy = _ideal_data()
    table = HelmholtzTable.from_thermodynamic_data(
        TEMPERATURES, MASS_DENSITIES, pressure, energy, entropy
    )
    table = jax.tree_util.tree_map(lambda value: value.astype(jnp.float32), table)
    points = jnp.asarray(
        [[TEMPERATURES[0], MASS_DENSITIES[0]], [TEMPERATURES[-1], MASS_DENSITIES[-1]]],
        dtype=jnp.float32,
    )
    states = jax.jit(jax.vmap(lambda point: table.state_trho(*point)))(points)
    gradients = jax.jit(
        jax.vmap(jax.grad(lambda point: table.specific_helmholtz(*point)))
    )(points)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree_util.tree_leaves(states))
    np.testing.assert_allclose(
        states.P, points[:, 0] * points[:, 1] * GAS_CONSTANT, rtol=2.0e-5
    )
    np.testing.assert_allclose(states.cv, HEAT_CAPACITY_CV, rtol=2.0e-5)
    np.testing.assert_allclose(gradients[:, 0], -states.s, rtol=2.0e-5)
    np.testing.assert_allclose(
        gradients[:, 1], GAS_CONSTANT * points[:, 0] / points[:, 1], rtol=2.0e-5
    )


@pytest.mark.parametrize(
    "temperatures",
    [
        np.asarray([1.0, 1.0 + 1.0e-10, 1.0 + 2.0e-10]),
        np.asarray([1.0e20, 1.0000001e20, 1.0000002e20]),
    ],
)
def test_constructor_rejects_grid_collapse_in_active_dtype(temperatures) -> None:
    previous = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="strictly increasing"):
            HelmholtzTable.from_helmholtz(
                temperatures, MASS_DENSITIES, np.ones((3, len(MASS_DENSITIES)))
            )
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize(
    "temperature,pressure,bounds",
    [
        (jnp.asarray([300.0]), 1.0e6, (0.1, 10.0)),
        (300.0, jnp.asarray([1.0e6]), (0.1, 10.0)),
        (300.0, 1.0e6, 1.0),
        (300.0, 1.0e6, (0.1,)),
        (300.0, 1.0e6, (0.1, 1.0, 10.0)),
        (300.0, 1.0e6, ((0.1, 10.0),)),
    ],
)
def test_tp_rejects_invalid_input_shapes(
    nonlinear_table: HelmholtzTable, temperature, pressure, bounds
) -> None:
    with pytest.raises(ValueError):
        nonlinear_table.state_tp(temperature, pressure, density_bounds=bounds)


@pytest.mark.parametrize(
    "temperature,density",
    [
        (119.0, 1.0),
        (1801.0, 1.0),
        (300.0, 0.079),
        (300.0, 91.0),
        (0.0, 1.0),
        (-100.0, 1.0),
        (300.0, 0.0),
        (300.0, -1.0),
        (np.nan, 1.0),
        (np.inf, 1.0),
        (300.0, np.nan),
        (300.0, np.inf),
    ],
)
def test_invalid_queries_return_nan(
    nonlinear_table: HelmholtzTable,
    temperature: float,
    density: float,
) -> None:
    state = nonlinear_table.state_trho(temperature, density)
    assert all(np.isnan(leaf) for leaf in jax.tree_util.tree_leaves(state))
    assert np.isnan(nonlinear_table.specific_helmholtz(temperature, density))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "invalid",
    [
        np.asarray([1.0, 2.0]),
        np.asarray([1.0, 3.0, 2.0]),
        np.asarray([1.0, 1.0, 2.0]),
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([-1.0, 1.0, 2.0]),
        np.asarray([1.0, 2.0, np.inf]),
        np.asarray([1.0, np.nan, 3.0]),
        np.asarray([[1.0, 2.0, 3.0]]),
    ],
)
def test_constructor_rejects_invalid_axes(axis: int, invalid: np.ndarray) -> None:
    axes = [TEMPERATURES, MASS_DENSITIES]
    axes[axis] = invalid
    values = np.ones((len(axes[0]), len(axes[1])))
    with pytest.raises(ValueError):
        HelmholtzTable.from_helmholtz(*axes, values)


@pytest.mark.parametrize(
    "invalid",
    [
        np.ones((len(TEMPERATURES), len(MASS_DENSITIES) - 1)),
        np.ones((len(TEMPERATURES), len(MASS_DENSITIES), 1)),
        np.full((len(TEMPERATURES), len(MASS_DENSITIES)), np.nan),
        np.full((len(TEMPERATURES), len(MASS_DENSITIES)), np.inf),
    ],
)
def test_constructor_rejects_invalid_potential_values(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError):
        HelmholtzTable.from_helmholtz(TEMPERATURES, MASS_DENSITIES, invalid)


def test_response_constraints_must_be_supplied_together() -> None:
    pressure, energy, entropy = _ideal_data()
    with pytest.raises(ValueError):
        HelmholtzTable.from_thermodynamic_data(
            TEMPERATURES,
            MASS_DENSITIES,
            pressure,
            energy,
            entropy,
            dlnrho_dlnT_P=-np.ones_like(pressure),
        )


def test_response_constraints_reject_zero_compressibility() -> None:
    pressure, energy, entropy = _ideal_data()
    with pytest.raises(ValueError, match="nonzero"):
        HelmholtzTable.from_thermodynamic_data(
            TEMPERATURES,
            MASS_DENSITIES,
            pressure,
            energy,
            entropy,
            dlnrho_dlnT_P=-np.ones_like(pressure),
            dlnrho_dlnP_T=np.zeros_like(pressure),
            dlns_dlnT_P=(HEAT_CAPACITY_CV + GAS_CONSTANT) / entropy,
        )


@pytest.mark.parametrize(
    "temperature,density",
    [
        (jnp.asarray([300.0]), 1.0),
        (300.0, jnp.asarray([1.0])),
    ],
)
def test_queries_reject_non_scalar_inputs(
    nonlinear_table: HelmholtzTable,
    temperature,
    density,
) -> None:
    with pytest.raises(ValueError):
        nonlinear_table.specific_helmholtz(temperature, density)
    with pytest.raises(ValueError):
        nonlinear_table.state_trho(temperature, density)
