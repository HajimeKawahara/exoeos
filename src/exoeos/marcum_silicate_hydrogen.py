"""Marcum et al. tabulated silicate-hydrogen equation of state."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from jax.typing import ArrayLike

from exoeos.state import SilicateHydrogenState


Array = jax.Array

_COMMIT = "5179f25721bd0bcc0827c7f77b5e24c18c36aa6f"
_TABLE_FILENAME = "MgSiO3-H_lookup.csv"
_TABLE_URL = (
    "https://raw.githubusercontent.com/s-marcum/MgSiO3-H-EOS/"
    f"{_COMMIT}/{_TABLE_FILENAME}"
)
_TABLE_SHA256 = "153e6261008663bd2923cf54a85d7f3a11cc4303a2a6b8bbf38714a4d78fe69f"
_CITATION = (
    "Marcum, S. P., Stixrude, L., & Young, E. D. (2026), "
    "arXiv:2608.27401, https://doi.org/10.48550/arXiv.2608.27401"
)

_HEADER = (
    "H_wt_pct",
    "P_GPa",
    "T_K",
    "rho_gcc",
    "H_kJ_g",
    "S_J_g_K",
    "alpha_1e5",
    "cp_J_g_K",
    "KS_GPa",
    "rho0_gcc",
    "eta",
    "gamma",
)
_HYDROGEN_WEIGHT_PERCENT_AXIS = np.asarray(
    (
        0.000100,
        0.400088,
        0.796889,
        1.190540,
        1.581080,
        1.968545,
        2.352971,
        2.734393,
        3.112848,
        3.488369,
        3.860990,
    )
)
_ENDMEMBER_FRACTION_AXIS = np.linspace(0.0001, 4.0, 11) / 4.0
_PRESSURE_GPA_AXIS = np.concatenate(
    (np.arange(1.0, 5.0), np.arange(5.0, 805.0, 5.0))
)
_TEMPERATURE_K_AXIS = np.arange(3000.0, 10050.0, 50.0)

_COMPOSITION_COUNT = 11
_PRESSURE_COUNT = 164
_TEMPERATURE_COUNT = 141
_FIELD_COUNT = 9
_ROW_COUNT = 250_844
_COLUMN_COUNT = 12
_LOW_PRESSURE_COUNT = 4
_LOW_PRESSURE_TEMPERATURE_MAX = 6000.0

_MGSIO3_MOLAR_MASS = 0.10039
_HYDROGEN_ATOMIC_MOLAR_MASS = 0.001008
_MOLAR_MASSES = (
    _MGSIO3_MOLAR_MASS,
    _MGSIO3_MOLAR_MASS + 4.0 * _HYDROGEN_ATOMIC_MOLAR_MASS,
)
_ENDMEMBERS = ("MgSiO3", "MgSiO3H4")

_TABLE_DOMAIN = {
    "temperature_K": (3000.0, 10000.0),
    "pressure_Pa": (1.0e9, 8.0e11),
    "hydrogen_weight_percent": (0.000100, 3.860990),
    "hydrogen_endmember_mole_fraction": (2.5e-5, 1.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_cache_directory() -> Path:
    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        root = Path(cache_root).expanduser()
    else:
        root = Path.home() / ".cache"
    return root / "exoeos" / "MgSiO3-H-EOS"


@dataclass(frozen=True, init=False)
class MarcumSilicateHydrogenTableLoader:
    """Fetch and load the pinned Marcum et al. MgSiO3-H lookup table."""

    cache_directory: Path

    def __init__(
        self,
        cache_directory: Optional[Union[str, PathLike[str]]] = None,
    ) -> None:
        directory = (
            _default_cache_directory()
            if cache_directory is None
            else Path(cache_directory).expanduser()
        )
        object.__setattr__(self, "cache_directory", directory)

    @property
    def expected_filename(self) -> str:
        """Published CSV filename."""

        return _TABLE_FILENAME

    @property
    def checksum(self) -> str:
        """SHA-256 checksum of the pinned CSV file."""

        return _TABLE_SHA256

    @property
    def table_checksum(self) -> str:
        """SHA-256 checksum of the pinned CSV file."""

        return self.checksum

    @property
    def citation(self) -> str:
        """Citation for the table source."""

        return _CITATION

    @property
    def table_domain(self) -> Dict[str, Tuple[float, float]]:
        """Nominal table-coordinate domain in SI units where applicable."""

        return dict(_TABLE_DOMAIN)

    @property
    def table_url(self) -> str:
        """URL of the CSV pinned to the verified upstream commit."""

        return _TABLE_URL

    @property
    def commit(self) -> str:
        """Pinned upstream Git commit."""

        return _COMMIT

    def fetch(self) -> Path:
        """Return the verified cached CSV, downloading it if necessary."""

        table_path = self.cache_directory / self.expected_filename
        if table_path.is_file() and _sha256(table_path) == self.checksum:
            return table_path

        self.cache_directory.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.cache_directory,
                prefix=f".{self.expected_filename}.",
                delete=False,
            ) as destination:
                temporary_path = Path(destination.name)
                with urlopen(self.table_url, timeout=60) as source:
                    shutil.copyfileobj(source, destination)
            if _sha256(temporary_path) != self.checksum:
                raise ValueError(f"Checksum mismatch for {self.expected_filename}.")
            temporary_path.replace(table_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return table_path

    def load(self) -> "MarcumSilicateHydrogenEOS":
        """Return an EOS backed by the verified table."""

        return MarcumSilicateHydrogenEOS.from_file(self.fetch())


def _scalar_array(value: ArrayLike, name: str) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar; use jax.vmap for batches.")
    return array


def _composition_array(value: ArrayLike) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.shape != (2,):
        raise ValueError(
            "x must contain the two mole fractions (MgSiO3, MgSiO3H4) "
            f"with shape (2,); received {array.shape}."
        )
    return array


def _table_array(value: ArrayLike) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    expected_shape = (
        _COMPOSITION_COUNT,
        _TEMPERATURE_COUNT,
        _PRESSURE_COUNT,
        _FIELD_COUNT,
    )
    if array.shape != expected_shape:
        raise ValueError(
            f"fields must have shape {expected_shape}; received {array.shape}."
        )
    return array


def _read_table(path: Path) -> np.ndarray:
    with path.open("r", encoding="ascii", newline="") as stream:
        header = stream.readline().rstrip("\r\n")
    expected_header = ",".join(_HEADER)
    if header != expected_header:
        raise ValueError(
            f"{path} must have header {expected_header!r}; received {header!r}."
        )

    try:
        rows = np.loadtxt(path, delimiter=",", skiprows=1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid Marcum silicate-hydrogen table {path}: {exc}"
        ) from exc

    expected_shape = (_ROW_COUNT, _COLUMN_COUNT)
    if rows.shape != expected_shape:
        raise ValueError(
            f"{path} must contain {_ROW_COUNT} data rows and {_COLUMN_COUNT} "
            f"columns; received shape {rows.shape}."
        )
    if not np.all(np.isfinite(rows)):
        raise ValueError(f"{path} contains non-finite values.")

    expected_weight_percent = (
        100.0
        * 4.0
        * (_HYDROGEN_ATOMIC_MOLAR_MASS * 1.0e3)
        * _ENDMEMBER_FRACTION_AXIS
        / (
            _MGSIO3_MOLAR_MASS * 1.0e3
            + 4.0
            * (_HYDROGEN_ATOMIC_MOLAR_MASS * 1.0e3)
            * _ENDMEMBER_FRACTION_AXIS
        )
    )
    if not np.allclose(
        _HYDROGEN_WEIGHT_PERCENT_AXIS,
        expected_weight_percent,
        rtol=0.0,
        atol=5.0e-4,
    ):
        raise ValueError("Hydrogen grid is inconsistent with paper equation (9).")

    fields = np.full(
        (_COMPOSITION_COUNT, _TEMPERATURE_COUNT, _PRESSURE_COUNT, _FIELD_COUNT),
        np.nan,
        dtype=rows.dtype,
    )
    cursor = 0
    for composition_index, weight_percent in enumerate(
        _HYDROGEN_WEIGHT_PERCENT_AXIS
    ):
        for temperature_index, temperature in enumerate(_TEMPERATURE_K_AXIS):
            pressure_start = (
                0
                if temperature <= _LOW_PRESSURE_TEMPERATURE_MAX
                else _LOW_PRESSURE_COUNT
            )
            pressures = _PRESSURE_GPA_AXIS[pressure_start:]
            stop = cursor + pressures.size
            block = rows[cursor:stop]
            if not np.allclose(block[:, 0], weight_percent, rtol=0.0, atol=5.0e-7):
                raise ValueError(
                    f"{path} does not contain the expected composition-major "
                    "hydrogen grid."
                )
            if not np.allclose(block[:, 1], pressures, rtol=0.0, atol=5.0e-5):
                raise ValueError(
                    f"{path} does not contain the expected temperature-major "
                    "pressure grid."
                )
            if not np.allclose(block[:, 2], temperature, rtol=0.0, atol=5.0e-3):
                raise ValueError(
                    f"{path} does not contain the expected temperature grid."
                )
            fields[
                composition_index,
                temperature_index,
                pressure_start:,
            ] = block[:, 3:]
            cursor = stop

    if cursor != _ROW_COUNT:
        raise ValueError(f"{path} contains unexpected trailing rows.")
    return fields


def _axis_bracket(axis: Array, coordinate: Array) -> tuple[Array, Array, Array]:
    clipped = jnp.clip(coordinate, axis[0], axis[-1])
    upper_index = jnp.searchsorted(axis, clipped, side="right")
    upper_index = jnp.clip(upper_index, 1, axis.shape[0] - 1).astype(jnp.int32)
    lower_index = upper_index - 1
    lower = axis[lower_index]
    upper = axis[upper_index]
    fraction = (clipped - lower) / (upper - lower)
    return lower_index, upper_index, fraction


def _safe_lerp(lower: Array, upper: Array, fraction: Array) -> Array:
    safe_lower = jnp.where(
        (fraction == 1.0) & ~jnp.isfinite(lower),
        upper,
        lower,
    )
    safe_upper = jnp.where(
        (fraction == 0.0) & ~jnp.isfinite(upper),
        lower,
        upper,
    )
    return safe_lower + fraction * (safe_upper - safe_lower)


def _trilinear_interpolate(
    fields: Array,
    endmember_fraction: Array,
    temperature: Array,
    pressure_gpa: Array,
) -> Array:
    dtype = jnp.result_type(
        fields,
        endmember_fraction,
        temperature,
        pressure_gpa,
    )
    composition_axis = jnp.asarray(_ENDMEMBER_FRACTION_AXIS, dtype=dtype)
    temperature_axis = jnp.asarray(_TEMPERATURE_K_AXIS, dtype=dtype)
    pressure_axis = jnp.asarray(_PRESSURE_GPA_AXIS, dtype=dtype)
    fields = fields.astype(dtype)

    c0, c1, fc = _axis_bracket(composition_axis, endmember_fraction)
    t0, t1, ft = _axis_bracket(temperature_axis, temperature)
    p0, p1, fp = _axis_bracket(pressure_axis, pressure_gpa)

    c0t0 = _safe_lerp(fields[c0, t0, p0], fields[c0, t0, p1], fp)
    c0t1 = _safe_lerp(fields[c0, t1, p0], fields[c0, t1, p1], fp)
    c1t0 = _safe_lerp(fields[c1, t0, p0], fields[c1, t0, p1], fp)
    c1t1 = _safe_lerp(fields[c1, t1, p0], fields[c1, t1, p1], fp)
    c0_value = _safe_lerp(c0t0, c0t1, ft)
    c1_value = _safe_lerp(c1t0, c1t1, ft)
    return _safe_lerp(c0_value, c1_value, fc)


@tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class MarcumSilicateHydrogenEOS:
    """Composition-dependent MgSiO3-MgSiO3H4 lookup-table EOS.

    ``fields`` uses the nine native CSV columns after ``T_K`` and has shape
    ``(11, 141, 164, 9)``. Missing 1--4 GPa cells above 6000 K are ``nan``.
    Loading and table validation occur on the host; evaluation is pure JAX.
    Values that require extrapolation or a missing ragged-grid corner return
    an all-``nan`` state.
    """

    fields: Array

    def __init__(self, fields: ArrayLike) -> None:
        object.__setattr__(self, "fields", _table_array(fields))

    @classmethod
    def from_file(
        cls,
        path: Union[str, PathLike[str]],
    ) -> "MarcumSilicateHydrogenEOS":
        """Load and strictly validate one published CSV file."""

        return cls(_read_table(Path(path)))

    @property
    def endmembers(self) -> tuple[str, str]:
        """Endmember names in composition-vector order."""

        return _ENDMEMBERS

    @property
    def molar_masses(self) -> Array:
        """Endmember molar masses in kg mol^-1."""

        return jnp.asarray(_MOLAR_MASSES)

    @property
    def endmember_fraction_grid(self) -> Array:
        """Published MgSiO3H4 endmember mole-fraction grid."""

        return jnp.asarray(_ENDMEMBER_FRACTION_AXIS)

    def state_tp(
        self,
        T: ArrayLike,
        P: ArrayLike,
        x: ArrayLike,
    ) -> SilicateHydrogenState:
        """Evaluate one state from temperature, pressure, and mole fractions."""

        temperature = _scalar_array(T, "T")
        pressure = _scalar_array(P, "P")
        composition = _composition_array(x)
        dtype = jnp.result_type(
            temperature,
            pressure,
            composition,
            self.fields,
            jnp.float32,
        )
        temperature = temperature.astype(dtype)
        pressure = pressure.astype(dtype)
        composition = composition.astype(dtype)
        endmember_fraction = composition[1]
        fraction_minimum = jnp.asarray(_ENDMEMBER_FRACTION_AXIS[0], dtype=dtype)

        composition_is_valid = (
            jnp.all(jnp.isfinite(composition))
            & jnp.all(composition >= 0.0)
            & jnp.isclose(jnp.sum(composition), 1.0, rtol=1.0e-6, atol=1.0e-7)
            & (endmember_fraction >= fraction_minimum)
            & (endmember_fraction <= 1.0)
        )
        coordinate_is_valid = (
            jnp.isfinite(temperature)
            & jnp.isfinite(pressure)
            & (temperature >= _TEMPERATURE_K_AXIS[0])
            & (temperature <= _TEMPERATURE_K_AXIS[-1])
            & (pressure >= _PRESSURE_GPA_AXIS[0] * 1.0e9)
            & (pressure <= _PRESSURE_GPA_AXIS[-1] * 1.0e9)
        )
        interpolated = _trilinear_interpolate(
            self.fields,
            endmember_fraction,
            temperature,
            pressure / 1.0e9,
        )
        is_valid = (
            composition_is_valid
            & coordinate_is_valid
            & jnp.all(jnp.isfinite(interpolated))
        )
        nan = jnp.asarray(jnp.nan, dtype=dtype)

        def valid(value: Array) -> Array:
            return jnp.where(is_valid, value, nan)

        return SilicateHydrogenState(
            pressure=valid(pressure),
            mass_density=valid(1.0e3 * interpolated[0]),
            specific_enthalpy=valid(1.0e6 * interpolated[1]),
            specific_entropy=valid(1.0e3 * interpolated[2]),
            thermal_expansion=valid(1.0e-5 * interpolated[3]),
            specific_heat_capacity_cp=valid(1.0e3 * interpolated[4]),
            adiabatic_bulk_modulus=valid(1.0e9 * interpolated[5]),
            reference_mass_density=valid(1.0e3 * interpolated[6]),
            compression_ratio=valid(interpolated[7]),
            gruneisen_parameter=valid(interpolated[8]),
        )

    def mass_density_tp(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mole_fractions: ArrayLike,
    ) -> Array:
        """Return mass density in kg m^-3 for one temperature-pressure state."""

        return self.state_tp(temperature, pressure, mole_fractions).mass_density

    def tree_flatten(self):
        return (self.fields,), None

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        del auxiliary_data
        (fields,) = children
        return cls(fields)
