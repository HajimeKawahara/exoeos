"""Chabrier-Debras 2021 tabulated hydrogen-helium equation of state."""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from jax.typing import ArrayLike

from exoeos.state import MassThermodynamicState


Array = jax.Array

_VARIANT_HELIUM_MASS_FRACTIONS = {
    "Y0275": 0.275,
    "Y0292": 0.292,
    "Y0297": 0.297,
}

_LOG_T_MIN = 2.0
_LOG_P_GPA_MIN = -9.0
_LOG_RHO_GCC_MIN = -6.0
_LOG_GRID_STEP = 0.05

_TEMPERATURE_MIN = 1.0e2
_TEMPERATURE_MAX = 1.0e8
_PRESSURE_MIN = 1.0
_PRESSURE_MAX = 1.0e22
_MASS_DENSITY_MIN = 1.0e-3
_MASS_DENSITY_MAX = 1.0e9

_TEMPERATURE_COUNT = 121
_PRESSURE_COUNT = 441
_DENSITY_COUNT = 241
_COLUMN_COUNT = 10
_FIELD_COUNT = 8


def _scalar_array(value: ArrayLike, name: str) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar; use jax.vmap for batches.")
    return array


def _expected_axis(start: float, count: int) -> np.ndarray:
    return start + _LOG_GRID_STEP * np.arange(count)


def _read_table(
    path: Path,
    coordinate_column: int,
    coordinate_name: str,
    coordinate_start: float,
    coordinate_count: int,
) -> np.ndarray:
    try:
        rows = np.loadtxt(path, comments="#")
    except ValueError as exc:
        raise ValueError(f"Invalid Chabrier-Debras table {path}: {exc}") from exc

    expected_shape = (
        _TEMPERATURE_COUNT * coordinate_count,
        _COLUMN_COUNT,
    )
    if rows.shape != expected_shape:
        raise ValueError(
            f"{path} must contain {expected_shape[0]} rows and "
            f"{_COLUMN_COUNT} columns; received shape {rows.shape}."
        )
    if not np.all(np.isfinite(rows)):
        raise ValueError(f"{path} contains non-finite values.")

    temperatures = _expected_axis(_LOG_T_MIN, _TEMPERATURE_COUNT)
    coordinates = _expected_axis(coordinate_start, coordinate_count)
    expected_temperatures = np.repeat(temperatures, coordinate_count)
    expected_coordinates = np.tile(coordinates, _TEMPERATURE_COUNT)
    tolerance = 5.0e-7
    if not np.allclose(
        rows[:, 0],
        expected_temperatures,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            f"{path} does not contain the expected temperature-major "
            "log10(T / K) grid."
        )
    if not np.allclose(
        rows[:, coordinate_column],
        expected_coordinates,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            f"{path} does not contain the expected {coordinate_name} grid."
        )

    return rows.reshape(_TEMPERATURE_COUNT, coordinate_count, _COLUMN_COUNT)


def _read_tp_table(path: Path) -> Array:
    rows = _read_table(
        path,
        coordinate_column=1,
        coordinate_name="log10(P / GPa)",
        coordinate_start=_LOG_P_GPA_MIN,
        coordinate_count=_PRESSURE_COUNT,
    )
    return jnp.asarray(rows[..., 2:])


def _read_trho_table(path: Path) -> Array:
    rows = _read_table(
        path,
        coordinate_column=2,
        coordinate_name="log10(rho / (g cm^-3))",
        coordinate_start=_LOG_RHO_GCC_MIN,
        coordinate_count=_DENSITY_COUNT,
    )
    fields = np.concatenate((rows[..., 1:2], rows[..., 3:]), axis=-1)
    return jnp.asarray(fields)


def _table_array(
    value: ArrayLike,
    expected_shape: tuple[int, int, int],
    name: str,
) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}; received {array.shape}."
        )
    return array


def _bilinear_interpolate(
    values: Array,
    log_temperature: Array,
    log_coordinate: Array,
    coordinate_minimum: float,
    in_bounds: Array,
) -> Array:
    raw_temperature_position = (log_temperature - _LOG_T_MIN) / _LOG_GRID_STEP
    raw_coordinate_position = (log_coordinate - coordinate_minimum) / _LOG_GRID_STEP
    temperature_position = jnp.clip(
        raw_temperature_position,
        0.0,
        values.shape[0] - 1,
    )
    coordinate_position = jnp.clip(
        raw_coordinate_position,
        0.0,
        values.shape[1] - 1,
    )

    temperature_index = jnp.floor(temperature_position).astype(jnp.int32)
    coordinate_index = jnp.floor(coordinate_position).astype(jnp.int32)
    temperature_index = jnp.clip(temperature_index, 0, values.shape[0] - 2)
    coordinate_index = jnp.clip(coordinate_index, 0, values.shape[1] - 2)

    temperature_fraction = temperature_position - temperature_index.astype(
        temperature_position.dtype
    )
    coordinate_fraction = coordinate_position - coordinate_index.astype(
        coordinate_position.dtype
    )

    lower_left = values[temperature_index, coordinate_index]
    upper_left = values[temperature_index + 1, coordinate_index]
    lower_right = values[temperature_index, coordinate_index + 1]
    upper_right = values[temperature_index + 1, coordinate_index + 1]
    lower = lower_left + temperature_fraction * (upper_left - lower_left)
    upper = lower_right + temperature_fraction * (upper_right - lower_right)
    interpolated = lower + coordinate_fraction * (upper - lower)

    return jnp.where(in_bounds, interpolated, jnp.nan)


@tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class ChabrierDebrasEOS:
    """Fixed-composition Chabrier-Debras 2021 H/He table pair.

    The public interface uses SI units. Table loading and validation happen on
    the host in :meth:`from_directory`; state evaluation is pure JAX. The
    supported variants are ``"Y0275"``, ``"Y0292"``, and ``"Y0297"``. They
    are separate published datasets, not samples of a continuous composition
    coordinate. The latter two represent the effective abundances defined by
    the authors rather than additional independently simulated compositions.

    Values outside the nominal rectangular grids return ``nan`` rather than
    being clipped or extrapolated. The published tables do not include a mask
    for physically invalid states within those rectangles.
    """

    tp_fields: Array
    trho_fields: Array
    variant: str

    def __init__(
        self,
        tp_fields: ArrayLike,
        trho_fields: ArrayLike,
        *,
        variant: str,
    ) -> None:
        if variant not in _VARIANT_HELIUM_MASS_FRACTIONS:
            available = ", ".join(_VARIANT_HELIUM_MASS_FRACTIONS)
            raise ValueError(
                f"Unknown variant {variant!r}; available variants: {available}."
            )
        object.__setattr__(
            self,
            "tp_fields",
            _table_array(
                tp_fields,
                (_TEMPERATURE_COUNT, _PRESSURE_COUNT, _FIELD_COUNT),
                "tp_fields",
            ),
        )
        object.__setattr__(
            self,
            "trho_fields",
            _table_array(
                trho_fields,
                (_TEMPERATURE_COUNT, _DENSITY_COUNT, _FIELD_COUNT),
                "trho_fields",
            ),
        )
        object.__setattr__(self, "variant", variant)

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, PathLike[str]],
        *,
        variant: str = "Y0275",
    ) -> "ChabrierDebrasEOS":
        """Load one published fixed-composition TP/T-rho table pair."""

        if variant not in _VARIANT_HELIUM_MASS_FRACTIONS:
            available = ", ".join(_VARIANT_HELIUM_MASS_FRACTIONS)
            raise ValueError(
                f"Unknown variant {variant!r}; available variants: {available}."
            )
        data_directory = Path(directory)
        tp_path = data_directory / f"TABLEEOS_2021_TP_{variant}_v1"
        trho_path = data_directory / f"TABLEEOS_2021_Trho_{variant}_v1"
        return cls(
            _read_tp_table(tp_path),
            _read_trho_table(trho_path),
            variant=variant,
        )

    @property
    def helium_mass_fraction(self) -> float:
        """Published helium or effective-helium mass fraction of the dataset."""

        return _VARIANT_HELIUM_MASS_FRACTIONS[self.variant]

    def state_tp(
        self,
        T: ArrayLike,
        P: ArrayLike,
    ) -> MassThermodynamicState:
        """Evaluate a state from temperature [K] and pressure [Pa]."""

        temperature = _scalar_array(T, "T")
        pressure = _scalar_array(P, "P")
        dtype = jnp.result_type(
            temperature,
            pressure,
            self.tp_fields,
            jnp.float32,
        )
        temperature = temperature.astype(dtype)
        pressure = pressure.astype(dtype)
        in_bounds = (
            jnp.isfinite(temperature)
            & jnp.isfinite(pressure)
            & (temperature >= _TEMPERATURE_MIN)
            & (temperature <= _TEMPERATURE_MAX)
            & (pressure >= _PRESSURE_MIN)
            & (pressure <= _PRESSURE_MAX)
        )
        interpolated = _bilinear_interpolate(
            self.tp_fields,
            jnp.log10(temperature),
            jnp.log10(pressure) - 9.0,
            _LOG_P_GPA_MIN,
            in_bounds,
        )

        return MassThermodynamicState(
            pressure=jnp.where(in_bounds, pressure, jnp.nan),
            mass_density=1.0e3 * jnp.power(10.0, interpolated[0]),
            specific_internal_energy=1.0e6 * jnp.power(10.0, interpolated[1]),
            specific_entropy=1.0e6 * jnp.power(10.0, interpolated[2]),
            dlnrho_dlnT_P=interpolated[3],
            dlnrho_dlnP_T=interpolated[4],
            dlns_dlnT_P=interpolated[5],
            dlns_dlnP_T=interpolated[6],
            adiabatic_gradient=interpolated[7],
        )

    def state_trho(
        self,
        T: ArrayLike,
        mass_density: ArrayLike,
    ) -> MassThermodynamicState:
        """Evaluate a state from temperature [K] and mass density [kg m^-3]."""

        temperature = _scalar_array(T, "T")
        density = _scalar_array(mass_density, "mass_density")
        dtype = jnp.result_type(
            temperature,
            density,
            self.trho_fields,
            jnp.float32,
        )
        temperature = temperature.astype(dtype)
        density = density.astype(dtype)
        in_bounds = (
            jnp.isfinite(temperature)
            & jnp.isfinite(density)
            & (temperature >= _TEMPERATURE_MIN)
            & (temperature <= _TEMPERATURE_MAX)
            & (density >= _MASS_DENSITY_MIN)
            & (density <= _MASS_DENSITY_MAX)
        )
        interpolated = _bilinear_interpolate(
            self.trho_fields,
            jnp.log10(temperature),
            jnp.log10(density) - 3.0,
            _LOG_RHO_GCC_MIN,
            in_bounds,
        )

        return MassThermodynamicState(
            pressure=1.0e9 * jnp.power(10.0, interpolated[0]),
            mass_density=jnp.where(in_bounds, density, jnp.nan),
            specific_internal_energy=1.0e6 * jnp.power(10.0, interpolated[1]),
            specific_entropy=1.0e6 * jnp.power(10.0, interpolated[2]),
            dlnrho_dlnT_P=interpolated[3],
            dlnrho_dlnP_T=interpolated[4],
            dlns_dlnT_P=interpolated[5],
            dlns_dlnP_T=interpolated[6],
            adiabatic_gradient=interpolated[7],
        )

    def tree_flatten(self):
        return (self.tp_fields, self.trho_fields), self.variant

    @classmethod
    def tree_unflatten(cls, variant, children):
        tp_fields, trho_fields = children
        return cls(tp_fields, trho_fields, variant=variant)
