"""Mass-density closures and additive-volume density providers."""

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import jax
import jax.numpy as jnp
from jax import tree_util
from jax.typing import ArrayLike

from exoeos.contracts import MassDensityProvider, TPHelmholtzEOS


Array = jax.Array


class _MassDensityState(Protocol):
    mass_density: Array


class _FixedCompositionTPEOS(Protocol):
    def state_tp(
        self,
        T: ArrayLike,
        P: ArrayLike,
    ) -> _MassDensityState: ...


def _scalar_array(value: ArrayLike, name: str) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar; use jax.vmap for batches.")
    return array


def _component_vector(value: ArrayLike, name: str) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; use jax.vmap for batches.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one component.")
    return array


def mass_density_tp(
    eos: TPHelmholtzEOS,
    temperature: ArrayLike,
    pressure: ArrayLike,
    mole_fractions: ArrayLike,
    molar_masses: ArrayLike,
    *,
    phase: str = "vapor",
) -> Array:
    """Return mass density in kg m^-3 at temperature and pressure.

    ``molar_masses`` must be ordered like ``mole_fractions`` and use
    kg mol^-1. Numerical domain validation and composition normalization are
    left to the caller.
    """

    temperature_array = _scalar_array(temperature, "temperature")
    pressure_array = _scalar_array(pressure, "pressure")
    composition = _component_vector(mole_fractions, "mole_fractions")
    masses = _component_vector(molar_masses, "molar_masses")
    if composition.shape != masses.shape:
        raise ValueError(
            "molar_masses must have the same shape as mole_fractions; "
            f"received {masses.shape} and {composition.shape}."
        )

    dtype = jnp.result_type(
        temperature_array,
        pressure_array,
        composition,
        masses,
        jnp.float32,
    )
    temperature_array = temperature_array.astype(dtype)
    pressure_array = pressure_array.astype(dtype)
    composition = composition.astype(dtype)
    masses = masses.astype(dtype)

    molar_density = jnp.asarray(
        eos.molar_density(
            temperature_array,
            pressure_array,
            composition,
            phase=phase,
        )
    )
    if molar_density.ndim != 0:
        raise ValueError("eos.molar_density must return a scalar for a single state.")

    dtype = jnp.result_type(molar_density, composition, masses)
    molar_density = molar_density.astype(dtype)
    composition = composition.astype(dtype)
    masses = masses.astype(dtype)
    return molar_density * jnp.sum(composition * masses)


def additive_volume_mass_density(
    mass_fractions: ArrayLike,
    component_mass_densities: ArrayLike,
) -> Array:
    """Return mixture density from ``1 / rho = sum_i(w_i / rho_i)``.

    Component mass densities must use kg m^-3. Numerical domain validation and
    mass-fraction normalization are left to the caller.
    """

    fractions = _component_vector(mass_fractions, "mass_fractions")
    densities = _component_vector(
        component_mass_densities,
        "component_mass_densities",
    )
    if fractions.shape != densities.shape:
        raise ValueError(
            "component_mass_densities must have the same shape as mass_fractions; "
            f"received {densities.shape} and {fractions.shape}."
        )

    dtype = jnp.result_type(fractions, densities, jnp.float32)
    fractions = fractions.astype(dtype)
    densities = densities.astype(dtype)
    return jnp.reciprocal(jnp.sum(fractions / densities))


@tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class TPHelmholtzDensityProvider:
    """Mass-density adapter for a temperature-pressure Helmholtz EOS."""

    eos: TPHelmholtzEOS
    molar_masses: Array
    phase: str

    def __init__(
        self,
        eos: TPHelmholtzEOS,
        molar_masses: ArrayLike,
        *,
        phase: str = "vapor",
    ) -> None:
        object.__setattr__(self, "eos", eos)
        object.__setattr__(
            self,
            "molar_masses",
            _component_vector(molar_masses, "molar_masses"),
        )
        object.__setattr__(self, "phase", phase)

    def mass_density_tp(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mole_fractions: ArrayLike,
    ) -> Array:
        """Return mass density in kg m^-3 from the wrapped molar EOS."""

        return mass_density_tp(
            self.eos,
            temperature,
            pressure,
            mole_fractions,
            self.molar_masses,
            phase=self.phase,
        )

    def tree_flatten(self):
        return (self.eos, self.molar_masses), self.phase

    @classmethod
    def tree_unflatten(cls, phase, children):
        eos, molar_masses = children
        return cls(eos, molar_masses, phase=phase)


@tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class FixedCompositionDensityProvider:
    """Mass-density adapter for an EOS tabulated at one mass composition.

    A composition mismatch returns ``nan`` so evaluation remains compatible
    with JAX transformations. The relative tolerance is static model metadata.
    """

    eos: _FixedCompositionTPEOS
    molar_masses: Array
    expected_mass_fractions: Array
    composition_rtol: float

    def __init__(
        self,
        eos: _FixedCompositionTPEOS,
        molar_masses: ArrayLike,
        expected_mass_fractions: ArrayLike,
        *,
        composition_rtol: float,
    ) -> None:
        masses = _component_vector(molar_masses, "molar_masses")
        expected = _component_vector(
            expected_mass_fractions,
            "expected_mass_fractions",
        )
        if expected.shape != masses.shape:
            raise ValueError(
                "expected_mass_fractions must have the same shape as "
                f"molar_masses; received {expected.shape} and {masses.shape}."
            )
        tolerance = float(composition_rtol)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("composition_rtol must be finite and non-negative.")

        object.__setattr__(self, "eos", eos)
        object.__setattr__(self, "molar_masses", masses)
        object.__setattr__(self, "expected_mass_fractions", expected)
        object.__setattr__(self, "composition_rtol", tolerance)

    def mass_density_tp(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mole_fractions: ArrayLike,
    ) -> Array:
        """Return table density when the supplied mass composition matches."""

        temperature_array = _scalar_array(temperature, "temperature")
        pressure_array = _scalar_array(pressure, "pressure")
        composition = _component_vector(mole_fractions, "mole_fractions")
        if composition.shape != self.molar_masses.shape:
            raise ValueError(
                "mole_fractions must have the same shape as molar_masses; "
                f"received {composition.shape} and {self.molar_masses.shape}."
            )

        dtype = jnp.result_type(
            temperature_array,
            pressure_array,
            composition,
            self.molar_masses,
            self.expected_mass_fractions,
            jnp.float32,
        )
        temperature_array = temperature_array.astype(dtype)
        pressure_array = pressure_array.astype(dtype)
        composition = composition.astype(dtype)
        masses = self.molar_masses.astype(dtype)
        expected = self.expected_mass_fractions.astype(dtype)

        mass_amounts = composition * masses
        total_mass = jnp.sum(mass_amounts)
        has_mass = total_mass > 0.0
        safe_total_mass = jnp.where(has_mass, total_mass, jnp.ones_like(total_mass))
        actual = mass_amounts / safe_total_mass
        composition_matches = has_mass & jnp.all(
            jnp.isclose(
                actual,
                expected,
                rtol=self.composition_rtol,
                atol=0.0,
            )
        )

        density = jnp.asarray(
            self.eos.state_tp(temperature_array, pressure_array).mass_density
        )
        if density.ndim != 0:
            raise ValueError("eos.state_tp must return scalar mass_density.")
        result_dtype = jnp.result_type(density, dtype)
        density = density.astype(result_dtype)
        return jnp.where(
            composition_matches, density, jnp.asarray(jnp.nan, result_dtype)
        )

    def tree_flatten(self):
        children = (
            self.eos,
            self.molar_masses,
            self.expected_mass_fractions,
        )
        return children, self.composition_rtol

    @classmethod
    def tree_unflatten(cls, composition_rtol, children):
        eos, molar_masses, expected_mass_fractions = children
        return cls(
            eos,
            molar_masses,
            expected_mass_fractions,
            composition_rtol=composition_rtol,
        )


@tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class DensityComponent:
    """Associate an ordered species group with a mass-density provider."""

    species: tuple[str, ...]
    provider: MassDensityProvider

    def __init__(
        self,
        species: Sequence[str],
        provider: MassDensityProvider,
    ) -> None:
        formulas = tuple(species)
        if not formulas:
            raise ValueError("component species must be a non-empty sequence.")
        if len(set(formulas)) != len(formulas):
            raise ValueError("component species must not contain duplicates.")
        try:
            masses = _component_vector(provider.molar_masses, "provider.molar_masses")
        except AttributeError as exc:
            raise TypeError("provider must expose molar_masses.") from exc
        if masses.shape != (len(formulas),):
            raise ValueError(
                "provider.molar_masses must match component species; "
                f"received {masses.shape} for {len(formulas)} species."
            )

        object.__setattr__(self, "species", formulas)
        object.__setattr__(self, "provider", provider)

    def tree_flatten(self):
        return (self.provider,), self.species

    @classmethod
    def tree_unflatten(cls, species, children):
        (provider,) = children
        return cls(species, provider)


@tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class AdditiveVolumeCompositeDensityProvider:
    """Combine disjoint density components with the additive-volume law."""

    species: tuple[str, ...]
    components: tuple[DensityComponent, ...]
    _component_indices: tuple[tuple[int, ...], ...]

    def __init__(
        self,
        species: Sequence[str],
        components: Sequence[DensityComponent],
    ) -> None:
        formulas = tuple(species)
        grouped_components = tuple(components)
        if not formulas:
            raise ValueError("species must be a non-empty sequence.")
        if len(set(formulas)) != len(formulas):
            raise ValueError("species must not contain duplicates.")
        if not grouped_components:
            raise ValueError("components must be a non-empty sequence.")

        positions = {formula: index for index, formula in enumerate(formulas)}
        assigned = set()
        component_indices = []
        for component in grouped_components:
            indices = []
            for formula in component.species:
                if formula not in positions:
                    raise ValueError(
                        f"Unknown component species {formula!r}; it is not in species."
                    )
                if formula in assigned:
                    raise ValueError(
                        f"Species {formula!r} is assigned to more than one component."
                    )
                assigned.add(formula)
                indices.append(positions[formula])
            component_indices.append(tuple(indices))

        missing = tuple(formula for formula in formulas if formula not in assigned)
        if missing:
            raise ValueError(f"Species are not assigned to a component: {missing}.")

        object.__setattr__(self, "species", formulas)
        object.__setattr__(self, "components", grouped_components)
        object.__setattr__(self, "_component_indices", tuple(component_indices))

    @property
    def molar_masses(self) -> Array:
        """Return component molar masses in the composite species order."""

        ordered = [None] * len(self.species)
        for component, indices in zip(self.components, self._component_indices):
            local_masses = component.provider.molar_masses
            for local_index, global_index in enumerate(indices):
                ordered[global_index] = local_masses[local_index]
        return jnp.stack(tuple(ordered))

    def validate_species(self, species: Sequence[str]) -> None:
        """Raise when external species metadata does not exactly match."""

        received = tuple(species)
        if received != self.species:
            raise ValueError(
                f"species order mismatch: expected {self.species}, received {received}."
            )

    def mass_density_tp(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mole_fractions: ArrayLike,
    ) -> Array:
        """Return additive-volume mass density in kg m^-3 for one state."""

        temperature_array = _scalar_array(temperature, "temperature")
        pressure_array = _scalar_array(pressure, "pressure")
        composition = _component_vector(mole_fractions, "mole_fractions")
        masses = self.molar_masses
        if composition.shape != masses.shape:
            raise ValueError(
                "mole_fractions must match the composite species; "
                f"received {composition.shape} for {len(self.species)} species."
            )

        dtype = jnp.result_type(
            temperature_array,
            pressure_array,
            composition,
            masses,
            jnp.float32,
        )
        temperature_array = temperature_array.astype(dtype)
        pressure_array = pressure_array.astype(dtype)
        composition = composition.astype(dtype)
        masses = masses.astype(dtype)

        component_masses = []
        component_densities = []
        for component, indices in zip(self.components, self._component_indices):
            index_array = jnp.asarray(indices, dtype=jnp.int32)
            local_amounts = composition[index_array]
            local_total = jnp.sum(local_amounts)
            is_present = local_total > 0.0
            safe_total = jnp.where(is_present, local_total, jnp.ones_like(local_total))
            normalized = local_amounts / safe_total
            fallback = jnp.zeros_like(normalized).at[0].set(1.0)
            local_composition = jnp.where(is_present, normalized, fallback)

            component_density = jnp.asarray(
                component.provider.mass_density_tp(
                    temperature_array,
                    pressure_array,
                    local_composition,
                )
            )
            if component_density.ndim != 0:
                raise ValueError("component provider must return scalar mass density.")
            component_density = jnp.where(
                is_present,
                component_density,
                jnp.ones_like(component_density),
            )
            component_densities.append(component_density)
            component_masses.append(jnp.sum(local_amounts * masses[index_array]))

        grouped_masses = jnp.stack(component_masses)
        mass_fractions = grouped_masses / jnp.sum(grouped_masses)
        return additive_volume_mass_density(
            mass_fractions,
            jnp.stack(component_densities),
        )

    def tree_flatten(self):
        return (self.components,), self.species

    @classmethod
    def tree_unflatten(cls, species, children):
        (components,) = children
        return cls(species, components)
