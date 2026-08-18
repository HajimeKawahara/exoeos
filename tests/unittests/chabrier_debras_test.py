"""Synthetic-table tests for the Chabrier-Debras 2021 EOS."""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from exoeos import ChabrierDebrasEOS, MassThermodynamicState


VARIANTS = ("Y0275", "Y0292", "Y0297")
LOG_TEMPERATURES = tuple(2.0 + 0.05 * index for index in range(121))
LOG_PRESSURES = tuple(-9.0 + 0.05 * index for index in range(441))
LOG_DENSITIES = tuple(-6.0 + 0.05 * index for index in range(241))
HEADER = (
    "#  log T [K]      log P [GPa]   log rho [g/cc] "
    "log U [MJ/kg]  log S [MJ/kg/K] dlrho/dlT_P    "
    "dlrho/dlP_T     dlS/dlT_P      dlS/dlP_T       grad_ad\n"
)


def _tp_row(log_temperature: float, log_pressure: float) -> tuple[float, ...]:
    temperature_offset = log_temperature - 2.0
    pressure_offset = log_pressure + 9.0
    cross_term = temperature_offset * pressure_offset
    return (
        log_temperature,
        log_pressure,
        -4.0 + 0.20 * temperature_offset + 0.05 * pressure_offset + 0.010 * cross_term,
        -1.0 + 0.10 * temperature_offset + 0.02 * pressure_offset + 0.005 * cross_term,
        -3.0 + 0.03 * temperature_offset + 0.04 * pressure_offset - 0.002 * cross_term,
        -1.0 + 0.02 * temperature_offset - 0.03 * pressure_offset + 0.001 * cross_term,
        1.0 + 0.01 * temperature_offset + 0.02 * pressure_offset - 0.0005 * cross_term,
        0.2 + 0.04 * temperature_offset + 0.01 * pressure_offset + 0.0007 * cross_term,
        -0.3 + 0.03 * temperature_offset - 0.01 * pressure_offset + 0.0003 * cross_term,
        0.25
        + 0.005 * temperature_offset
        + 0.002 * pressure_offset
        + 0.0001 * cross_term,
    )


def _trho_row(log_temperature: float, log_density: float) -> tuple[float, ...]:
    temperature_offset = log_temperature - 2.0
    density_offset = log_density + 6.0
    cross_term = temperature_offset * density_offset
    return (
        log_temperature,
        -6.0 + 0.40 * temperature_offset + 0.60 * density_offset + 0.010 * cross_term,
        log_density,
        -2.0 + 0.08 * temperature_offset + 0.03 * density_offset + 0.004 * cross_term,
        -4.0 + 0.02 * temperature_offset + 0.05 * density_offset - 0.001 * cross_term,
        -0.8 + 0.03 * temperature_offset - 0.02 * density_offset + 0.0008 * cross_term,
        0.9 + 0.02 * temperature_offset + 0.01 * density_offset - 0.0004 * cross_term,
        0.3 + 0.01 * temperature_offset + 0.03 * density_offset + 0.0006 * cross_term,
        -0.2 + 0.04 * temperature_offset - 0.01 * density_offset + 0.0002 * cross_term,
        0.28
        + 0.004 * temperature_offset
        + 0.003 * density_offset
        + 0.0001 * cross_term,
    )


def _write_table(
    path: Path,
    log_temperatures: tuple[float, ...],
    secondary_axis: tuple[float, ...],
    row_function,
) -> None:
    with path.open("w", encoding="ascii") as stream:
        stream.write(HEADER)
        for index, log_temperature in enumerate(log_temperatures, start=1):
            stream.write(f"#iT={index:4d} log T={log_temperature:6.3f}\n")
            for secondary_coordinate in secondary_axis:
                values = row_function(log_temperature, secondary_coordinate)
                stream.write("  ".join(f"{value: .12E}" for value in values))
                stream.write("\n")


def _write_eos_pair(directory: Path, variant: str = "Y0275") -> None:
    _write_table(
        directory / f"TABLEEOS_2021_TP_{variant}_v1",
        LOG_TEMPERATURES,
        LOG_PRESSURES,
        _tp_row,
    )
    _write_table(
        directory / f"TABLEEOS_2021_Trho_{variant}_v1",
        LOG_TEMPERATURES,
        LOG_DENSITIES,
        _trho_row,
    )


@pytest.fixture(scope="module")
def table_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("chabrier_debras_tables")
    _write_eos_pair(directory)
    for variant in VARIANTS[1:]:
        os.link(
            directory / "TABLEEOS_2021_TP_Y0275_v1",
            directory / f"TABLEEOS_2021_TP_{variant}_v1",
        )
        os.link(
            directory / "TABLEEOS_2021_Trho_Y0275_v1",
            directory / f"TABLEEOS_2021_Trho_{variant}_v1",
        )
    return directory


@pytest.fixture(scope="module")
def eos(table_directory: Path) -> ChabrierDebrasEOS:
    return ChabrierDebrasEOS.from_directory(table_directory)


def _bilinear_midpoint(rows: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    return tuple(sum(row[index] for row in rows) / 4.0 for index in range(10))


def _assert_state_matches_row(
    state: MassThermodynamicState,
    row: tuple[float, ...],
) -> None:
    expected = (
        10.0 ** row[1] * 1.0e9,
        10.0 ** row[2] * 1.0e3,
        10.0 ** row[3] * 1.0e6,
        10.0 ** row[4] * 1.0e6,
        *row[5:],
    )
    assert isinstance(state, MassThermodynamicState)
    assert jnp.allclose(jnp.asarray(state), jnp.asarray(expected), rtol=2.0e-11)


def test_state_tp_uses_si_units_at_an_exact_grid_point(
    eos: ChabrierDebrasEOS,
) -> None:
    state = eos.state_tp(100.0, 1.0)

    assert state.pressure == pytest.approx(1.0)
    assert state.mass_density == pytest.approx(0.1)
    assert state.specific_internal_energy == pytest.approx(1.0e5)
    assert state.specific_entropy == pytest.approx(1.0e3)
    assert state.dlnrho_dlnT_P == pytest.approx(-1.0)
    assert state.dlnrho_dlnP_T == pytest.approx(1.0)
    assert state.dlns_dlnT_P == pytest.approx(0.2)
    assert state.dlns_dlnP_T == pytest.approx(-0.3)
    assert state.adiabatic_gradient == pytest.approx(0.25)


def test_state_trho_uses_si_units_at_an_exact_grid_point(
    eos: ChabrierDebrasEOS,
) -> None:
    state = eos.state_trho(100.0, 1.0e-3)

    assert state.pressure == pytest.approx(1.0e3)
    assert state.mass_density == pytest.approx(1.0e-3)
    assert state.specific_internal_energy == pytest.approx(1.0e4)
    assert state.specific_entropy == pytest.approx(1.0e2)
    assert state.dlnrho_dlnT_P == pytest.approx(-0.8)
    assert state.dlnrho_dlnP_T == pytest.approx(0.9)
    assert state.dlns_dlnT_P == pytest.approx(0.3)
    assert state.dlns_dlnP_T == pytest.approx(-0.2)
    assert state.adiabatic_gradient == pytest.approx(0.28)


def test_state_tp_interpolates_bilinearly_in_log_coordinates(
    eos: ChabrierDebrasEOS,
) -> None:
    log_temperature_bounds = (3.0, 3.05)
    log_pressure_bounds = (-1.0, -0.95)
    rows = tuple(
        _tp_row(log_temperature, log_pressure)
        for log_temperature in log_temperature_bounds
        for log_pressure in log_pressure_bounds
    )
    expected = _bilinear_midpoint(rows)

    state = eos.state_tp(10.0**3.025, 10.0 ** (-0.975) * 1.0e9)

    _assert_state_matches_row(state, expected)


def test_state_trho_interpolates_bilinearly_in_log_coordinates(
    eos: ChabrierDebrasEOS,
) -> None:
    log_temperature_bounds = (3.0, 3.05)
    log_density_bounds = (-1.0, -0.95)
    rows = tuple(
        _trho_row(log_temperature, log_density)
        for log_temperature in log_temperature_bounds
        for log_density in log_density_bounds
    )
    expected = _bilinear_midpoint(rows)

    state = eos.state_trho(10.0**3.025, 10.0 ** (-0.975) * 1.0e3)

    _assert_state_matches_row(state, expected)


def test_exact_upper_grid_bound_is_in_range(eos: ChabrierDebrasEOS) -> None:
    tp_state = eos.state_tp(1.0e8, 1.0e22)
    trho_state = eos.state_trho(1.0e8, 1.0e9)

    _assert_state_matches_row(tp_state, _tp_row(8.0, 13.0))
    _assert_state_matches_row(trho_state, _trho_row(8.0, 6.0))


@pytest.mark.parametrize(
    ("variant", "helium_mass_fraction"),
    (("Y0275", 0.275), ("Y0292", 0.292), ("Y0297", 0.297)),
)
def test_supported_variants_load_their_named_pair(
    table_directory: Path,
    variant: str,
    helium_mass_fraction: float,
) -> None:
    selected = ChabrierDebrasEOS.from_directory(table_directory, variant=variant)

    assert selected.helium_mass_fraction == helium_mass_fraction
    assert jnp.isfinite(selected.state_tp(100.0, 1.0).mass_density)
    assert jnp.isfinite(selected.state_trho(100.0, 1.0e-3).pressure)


def test_invalid_variant_is_rejected(table_directory: Path) -> None:
    with pytest.raises(ValueError):
        ChabrierDebrasEOS.from_directory(table_directory, variant="Y0280")


def test_missing_tables_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ChabrierDebrasEOS.from_directory(tmp_path)


def test_incomplete_rectangular_grid_is_rejected(tmp_path: Path) -> None:
    _write_table(
        tmp_path / "TABLEEOS_2021_TP_Y0275_v1",
        (2.0, 2.05),
        (-9.0, -8.95),
        _tp_row,
    )
    _write_table(
        tmp_path / "TABLEEOS_2021_Trho_Y0275_v1",
        (2.0, 2.05),
        (-6.0, -5.95),
        _trho_row,
    )

    with pytest.raises(ValueError):
        ChabrierDebrasEOS.from_directory(tmp_path)


def test_out_of_bounds_queries_return_nan(eos: ChabrierDebrasEOS) -> None:
    states = (
        eos.state_tp(10.0, 1.0),
        eos.state_tp(100.0, 0.1),
        eos.state_trho(1.0e9, 1.0),
        eos.state_trho(100.0, 1.0e-4),
    )

    for state in states:
        assert all(bool(jnp.isnan(leaf)) for leaf in jax.tree_util.tree_leaves(state))


def test_state_methods_support_jit_vmap_and_pytree_round_trip(
    eos: ChabrierDebrasEOS,
) -> None:
    compiled_state_tp = jax.jit(
        lambda model, temperature, pressure: model.state_tp(temperature, pressure)
    )
    scalar_state = compiled_state_tp(
        eos,
        jnp.asarray(1.0e3),
        jnp.asarray(1.0e9),
    )
    temperatures = jnp.asarray([100.0, 1.0e3, 1.0e8])
    pressures = jnp.asarray([1.0, 1.0e9, 1.0e22])
    batched_state = jax.jit(jax.vmap(eos.state_tp))(temperatures, pressures)
    leaves, tree_definition = jax.tree_util.tree_flatten(scalar_state)
    reconstructed = jax.tree_util.tree_unflatten(tree_definition, leaves)
    model_leaves, model_definition = jax.tree_util.tree_flatten(eos)
    reconstructed_model = jax.tree_util.tree_unflatten(
        model_definition,
        model_leaves,
    )
    density_gradient = jax.grad(
        lambda temperature: eos.state_tp(temperature, 1.0e9).rho
    )(1.0e3)

    assert isinstance(scalar_state, MassThermodynamicState)
    assert isinstance(batched_state, MassThermodynamicState)
    assert isinstance(reconstructed, MassThermodynamicState)
    assert isinstance(reconstructed_model, ChabrierDebrasEOS)
    assert reconstructed_model.variant == eos.variant
    assert jnp.isfinite(density_gradient)
    assert len(leaves) == len(MassThermodynamicState._fields)
    assert batched_state.mass_density.shape == (3,)
    assert jnp.all(jnp.isfinite(jnp.asarray(batched_state)))
    assert all(
        jnp.array_equal(actual, expected)
        for actual, expected in zip(reconstructed, scalar_state)
    )
