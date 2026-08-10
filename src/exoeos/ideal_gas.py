"""Calorically perfect ideal-gas mixture backend."""

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import tree_util
from jax.scipy.special import xlogy
from jax.typing import ArrayLike

from exoeos.constants import BOLTZMANN_CONSTANT, MOLAR_GAS_CONSTANT
from exoeos.state import ThermodynamicState


Array = jax.Array


def _component_array(value: ArrayLike, name: str) -> Array:
    array = jnp.atleast_1d(jnp.asarray(value))
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional array.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one component.")
    return array


def _reference_array(
    value: Optional[ArrayLike],
    template: Array,
    name: str,
) -> Array:
    if value is None:
        return jnp.zeros_like(template)
    array = _component_array(value, name)
    if array.shape != template.shape:
        raise ValueError(
            f"{name} must have shape {template.shape}; received {array.shape}."
        )
    return array


def _scalar_array(value: ArrayLike, name: str) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    return array


def _broadcast_inputs(
    T: ArrayLike,
    P: ArrayLike,
    x: ArrayLike,
    component_count: int,
) -> Tuple[Array, Array, Array]:
    temperature = jnp.asarray(T)
    pressure = jnp.asarray(P)
    mole_fractions = jnp.asarray(x)

    if mole_fractions.ndim == 0:
        raise ValueError("x must have a trailing component axis.")
    if mole_fractions.shape[-1] != component_count:
        raise ValueError(
            "The trailing dimension of x must match the number of components "
            f"({component_count}); received {mole_fractions.shape[-1]}."
        )

    dtype = jnp.result_type(temperature, pressure, mole_fractions, jnp.float32)
    temperature = temperature.astype(dtype)
    pressure = pressure.astype(dtype)
    mole_fractions = mole_fractions.astype(dtype)

    batch_shape = jnp.broadcast_shapes(
        temperature.shape,
        pressure.shape,
        mole_fractions.shape[:-1],
    )
    temperature = jnp.broadcast_to(temperature, batch_shape)
    pressure = jnp.broadcast_to(pressure, batch_shape)
    mole_fractions = jnp.broadcast_to(
        mole_fractions,
        batch_shape + (component_count,),
    )
    return temperature, pressure, mole_fractions


def ideal_gas_state(
    T: ArrayLike,
    P: ArrayLike,
    x: ArrayLike,
    molar_masses: ArrayLike,
    molar_heat_capacities: ArrayLike,
    reference_enthalpies: ArrayLike,
    reference_entropies: ArrayLike,
    reference_temperature: ArrayLike,
    reference_pressure: ArrayLike,
) -> ThermodynamicState:
    """Evaluate the pure-JAX ideal-gas mixture equations.

    This function performs static shape checks but deliberately leaves numerical
    domain validation to the caller so it remains safe under JAX transformations.
    """

    masses = _component_array(molar_masses, "molar_masses")
    heat_capacities = _component_array(
        molar_heat_capacities,
        "molar_heat_capacities",
    )
    enthalpies = _component_array(reference_enthalpies, "reference_enthalpies")
    entropies = _component_array(reference_entropies, "reference_entropies")
    component_shape = masses.shape
    for name, array in (
        ("molar_heat_capacities", heat_capacities),
        ("reference_enthalpies", enthalpies),
        ("reference_entropies", entropies),
    ):
        if array.shape != component_shape:
            raise ValueError(
                f"{name} must have shape {component_shape}; received {array.shape}."
            )

    reference_T = _scalar_array(reference_temperature, "reference_temperature")
    reference_P = _scalar_array(reference_pressure, "reference_pressure")
    temperature, pressure, mole_fractions = _broadcast_inputs(
        T,
        P,
        x,
        component_shape[0],
    )

    dtype = jnp.result_type(
        temperature,
        pressure,
        mole_fractions,
        masses,
        heat_capacities,
        enthalpies,
        entropies,
        reference_T,
        reference_P,
        jnp.float32,
    )
    temperature = temperature.astype(dtype)
    pressure = pressure.astype(dtype)
    mole_fractions = mole_fractions.astype(dtype)
    masses = masses.astype(dtype)
    heat_capacities = heat_capacities.astype(dtype)
    enthalpies = enthalpies.astype(dtype)
    entropies = entropies.astype(dtype)
    reference_T = reference_T.astype(dtype)
    reference_P = reference_P.astype(dtype)

    component_axis_temperature = temperature[..., None]
    mean_molar_mass = jnp.sum(mole_fractions * masses, axis=-1)
    mixture_cp = jnp.sum(mole_fractions * heat_capacities, axis=-1)

    component_enthalpy = enthalpies + heat_capacities * (
        component_axis_temperature - reference_T
    )
    mixture_enthalpy = jnp.sum(mole_fractions * component_enthalpy, axis=-1)

    component_entropy = entropies + heat_capacities * jnp.log(
        component_axis_temperature / reference_T
    )
    thermal_entropy = jnp.sum(mole_fractions * component_entropy, axis=-1)
    pressure_entropy = -MOLAR_GAS_CONSTANT * jnp.log(pressure / reference_P)
    mixing_entropy = -MOLAR_GAS_CONSTANT * jnp.sum(
        xlogy(mole_fractions, mole_fractions),
        axis=-1,
    )
    mixture_entropy = thermal_entropy + pressure_entropy + mixing_entropy

    mixture_cv = mixture_cp - MOLAR_GAS_CONSTANT
    batch_shape = temperature.shape
    zeros = jnp.zeros(batch_shape, dtype=temperature.dtype)

    return ThermodynamicState(
        compressibility_factor=jnp.ones(batch_shape, dtype=temperature.dtype),
        mass_density=(pressure * mean_molar_mass / (MOLAR_GAS_CONSTANT * temperature)),
        number_density=pressure / (BOLTZMANN_CONSTANT * temperature),
        molar_enthalpy=mixture_enthalpy,
        molar_entropy=mixture_entropy,
        molar_heat_capacity_cp=mixture_cp,
        molar_heat_capacity_cv=mixture_cv,
        adiabatic_gradient=MOLAR_GAS_CONSTANT / mixture_cp,
        log_fugacity_coefficients=jnp.zeros_like(mole_fractions),
        residual_gibbs=zeros,
        residual_enthalpy=zeros,
        thermal_expansion=jnp.reciprocal(temperature),
    )


@tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class IdealGas:
    """Calorically perfect ideal-gas mixture.

    Component data are ordered along the trailing axis of every composition
    passed to :meth:`state`. Values are stored as JAX leaves, so model parameters
    can also participate in JAX transformations.
    """

    molar_masses: Array
    molar_heat_capacities: Array
    reference_enthalpies: Array
    reference_entropies: Array
    reference_temperature: Array
    reference_pressure: Array

    def __init__(
        self,
        molar_masses: ArrayLike,
        molar_heat_capacities: ArrayLike,
        *,
        reference_enthalpies: Optional[ArrayLike] = None,
        reference_entropies: Optional[ArrayLike] = None,
        reference_temperature: ArrayLike = 298.15,
        reference_pressure: ArrayLike = 1.0e5,
    ) -> None:
        masses = _component_array(molar_masses, "molar_masses")
        heat_capacities = _component_array(
            molar_heat_capacities,
            "molar_heat_capacities",
        )
        if heat_capacities.shape != masses.shape:
            raise ValueError(
                "molar_heat_capacities must have shape "
                f"{masses.shape}; received {heat_capacities.shape}."
            )

        object.__setattr__(self, "molar_masses", masses)
        object.__setattr__(self, "molar_heat_capacities", heat_capacities)
        object.__setattr__(
            self,
            "reference_enthalpies",
            _reference_array(reference_enthalpies, masses, "reference_enthalpies"),
        )
        object.__setattr__(
            self,
            "reference_entropies",
            _reference_array(reference_entropies, masses, "reference_entropies"),
        )
        object.__setattr__(
            self,
            "reference_temperature",
            _scalar_array(reference_temperature, "reference_temperature"),
        )
        object.__setattr__(
            self,
            "reference_pressure",
            _scalar_array(reference_pressure, "reference_pressure"),
        )

    @property
    def component_count(self) -> int:
        """Number of mixture components."""

        return self.molar_masses.shape[0]

    def state(
        self,
        T: ArrayLike,
        P: ArrayLike,
        x: ArrayLike,
    ) -> ThermodynamicState:
        """Return the ideal-gas state for ``T`` [K], ``P`` [Pa], and mole fractions."""

        return ideal_gas_state(
            T,
            P,
            x,
            self.molar_masses,
            self.molar_heat_capacities,
            self.reference_enthalpies,
            self.reference_entropies,
            self.reference_temperature,
            self.reference_pressure,
        )

    def tree_flatten(self):
        children = (
            self.molar_masses,
            self.molar_heat_capacities,
            self.reference_enthalpies,
            self.reference_entropies,
            self.reference_temperature,
            self.reference_pressure,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        (
            molar_masses,
            molar_heat_capacities,
            reference_enthalpies,
            reference_entropies,
            reference_temperature,
            reference_pressure,
        ) = children
        return cls(
            molar_masses,
            molar_heat_capacities,
            reference_enthalpies=reference_enthalpies,
            reference_entropies=reference_entropies,
            reference_temperature=reference_temperature,
            reference_pressure=reference_pressure,
        )
