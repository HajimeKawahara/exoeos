"""Zhang-Duan 2009 equation of state for C-O-H fluids."""

from dataclasses import dataclass
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from jax import lax, tree_util
from jax.typing import ArrayLike

from exoeos.constants import MOLAR_GAS_CONSTANT


Array = jax.Array

# Zhang and Duan (2009), Geochimica et Cosmochimica Acta 73, 2089-2102,
# Tables 3 and 4. The universal coefficients are kept as Python values so they
# adopt the input dtype when traced.
_EOS_COEFFICIENTS = (
    2.95177298930e-2,
    -6.33756452413e3,
    -2.75265428882e5,
    1.29128089283e-3,
    -1.45797416153e2,
    7.65938947237e4,
    2.58661493537e-6,
    5.21265321460e-1,
    -1.39839523753e2,
    -2.36335007175e-8,
    5.35026383543e-3,
    -2.71106499510e-1,
    2.50387836486e4,
    7.32267260410e-1,
    1.54833359970e-2,
)

_REFERENCE_DIAMETER = 3.691e-10
_REDUCED_TEMPERATURE_FACTOR = 154.0
_REDUCED_DENSITY_LIMIT = 128.0
_BRACKET_STEPS = 1024
_BISECTION_STEPS = 64

_PUBLISHED_COMPONENT_PARAMETERS = {
    "CH4": (154.0, 3.691e-10),
    "H2O": (510.0, 2.88e-10),
    "CO2": (235.0, 3.79e-10),
    "H2": (31.2, 2.93e-10),
    "CO": (105.6, 3.66e-10),
    "O2": (124.5, 3.36e-10),
    "C2H6": (246.1, 4.35e-10),
}

_PUBLISHED_BINARY_PARAMETERS = {
    frozenset(("H2O", "CO2")): (0.85, 1.02),
    frozenset(("H2O", "CH4")): (0.8, 1.0),
}


def _scalar_array(value: ArrayLike, name: str) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar; use jax.vmap for batches.")
    return array


def _component_array(value: ArrayLike, name: str) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.ndim != 1 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    return array


def _composition_array(value: ArrayLike, component_count: int) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(jnp.asarray(1.0).dtype)
    if array.ndim != 1 or array.shape[0] != component_count:
        raise ValueError(f"x must have shape ({component_count},).")
    return array


def _first_stable_root(function, initial_guess: Array) -> Array:
    """Bracket and refine the first negative-to-positive scalar root."""

    zero = jnp.zeros_like(initial_guess)
    limit = jnp.asarray(_REDUCED_DENSITY_LIMIT, dtype=initial_guess.dtype)
    initial_state = (
        zero,
        function(zero),
        zero,
        limit,
        jnp.asarray(False),
    )

    def scan(index, state):
        previous, previous_residual, lower, upper, found = state
        candidate = limit * (index + 1) / _BRACKET_STEPS
        residual = function(candidate)
        crossing = (~found) & (previous_residual <= 0.0) & (residual >= 0.0)
        lower = jnp.where(crossing, previous, lower)
        upper = jnp.where(crossing, candidate, upper)
        return candidate, residual, lower, upper, found | crossing

    _, _, lower, upper, found = lax.fori_loop(
        0,
        _BRACKET_STEPS,
        scan,
        initial_state,
    )

    def bisect(_, bracket):
        lower, upper = bracket
        midpoint = 0.5 * (lower + upper)
        residual = function(midpoint)
        lower = jnp.where(residual < 0.0, midpoint, lower)
        upper = jnp.where(residual >= 0.0, midpoint, upper)
        return lower, upper

    lower, upper = lax.fori_loop(
        0,
        _BISECTION_STEPS,
        bisect,
        (lower, upper),
    )
    root = 0.5 * (lower + upper)
    return jnp.where(found, root, jnp.asarray(jnp.nan, dtype=root.dtype))


def _scalar_tangent_solve(linear_function, value: Array) -> Array:
    """Solve the scalar linearization used by ``lax.custom_root``."""

    return value / linear_function(jnp.ones_like(value))


@tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class ZhangDuanEOS:
    """Zhang-Duan 2009 corresponding-states EOS.

    ``epsilon_over_k`` is in K and ``molecular_diameters`` is in m. The
    multiplicative energy and size interaction matrices are ``k1`` and ``k2``
    from the paper, respectively; both default to one. Numerical validity,
    normalized compositions, and matrix symmetry are caller contracts.

    The model describes a homogeneous fluid. Pressure inversion returns the
    first mechanically stable root connected to the low-density branch and
    supports only ``phase="vapor"`` through the common ExoEOS interface.
    The principal mixture calibration range is 673--2573 K and 1 MPa--10 GPa
    (10 MPa--10 GPa for H2O-CH4). Pure-species data ranges differ; evaluation
    outside the ranges reported by Zhang and Duan (2009) is extrapolation.
    The physical compressibility is ``P V / (R T)``; the scaled left-hand side
    printed in Equation 8 is not used because it does not reproduce Table 6.
    """

    epsilon_over_k: Array
    molecular_diameters: Array
    energy_interaction_parameters: Array
    size_interaction_parameters: Array

    def __init__(
        self,
        epsilon_over_k: ArrayLike,
        molecular_diameters: ArrayLike,
        energy_interaction_parameters: Optional[ArrayLike] = None,
        size_interaction_parameters: Optional[ArrayLike] = None,
    ) -> None:
        energies = _component_array(epsilon_over_k, "epsilon_over_k")
        diameters = _component_array(
            molecular_diameters,
            "molecular_diameters",
        )
        if diameters.shape != energies.shape:
            raise ValueError(
                f"molecular_diameters must have shape {energies.shape}."
            )

        component_count = energies.shape[0]
        shape = (component_count, component_count)
        dtype = jnp.result_type(energies, diameters, jnp.float32)

        if energy_interaction_parameters is None:
            energy_interactions = jnp.ones(shape, dtype=dtype)
        else:
            energy_interactions = jnp.asarray(energy_interaction_parameters)
            if not jnp.issubdtype(energy_interactions.dtype, jnp.inexact):
                energy_interactions = energy_interactions.astype(
                    jnp.asarray(1.0).dtype
                )
            if energy_interactions.shape != shape:
                raise ValueError(
                    "energy_interaction_parameters must have shape "
                    f"({component_count}, {component_count})."
                )

        if size_interaction_parameters is None:
            size_interactions = jnp.ones(shape, dtype=dtype)
        else:
            size_interactions = jnp.asarray(size_interaction_parameters)
            if not jnp.issubdtype(size_interactions.dtype, jnp.inexact):
                size_interactions = size_interactions.astype(jnp.asarray(1.0).dtype)
            if size_interactions.shape != shape:
                raise ValueError(
                    "size_interaction_parameters must have shape "
                    f"({component_count}, {component_count})."
                )

        object.__setattr__(self, "epsilon_over_k", energies)
        object.__setattr__(self, "molecular_diameters", diameters)
        object.__setattr__(
            self,
            "energy_interaction_parameters",
            energy_interactions,
        )
        object.__setattr__(
            self,
            "size_interaction_parameters",
            size_interactions,
        )

    @classmethod
    def from_species(cls, species: Sequence[str]) -> "ZhangDuanEOS":
        """Construct the published model for an ordered sequence of species.

        Supported formulas are ``CH4``, ``H2O``, ``CO2``, ``H2``, ``CO``,
        ``O2``, and ``C2H6``. Published H2O-CO2 and H2O-CH4 binary parameters
        are inserted automatically; all other binary parameters are one.
        """

        formulas = tuple(species)
        if not formulas:
            raise ValueError("species must be a non-empty sequence.")
        try:
            parameters = tuple(
                _PUBLISHED_COMPONENT_PARAMETERS[formula] for formula in formulas
            )
        except KeyError as exc:
            available = ", ".join(_PUBLISHED_COMPONENT_PARAMETERS)
            raise KeyError(
                f"Unknown species {exc.args[0]!r}; available species: {available}."
            ) from exc

        component_count = len(formulas)
        energy_interactions = [
            [1.0 for _ in range(component_count)]
            for _ in range(component_count)
        ]
        size_interactions = [
            [1.0 for _ in range(component_count)]
            for _ in range(component_count)
        ]
        for i, formula_i in enumerate(formulas):
            for j in range(i + 1, component_count):
                pair = frozenset((formula_i, formulas[j]))
                energy, size = _PUBLISHED_BINARY_PARAMETERS.get(pair, (1.0, 1.0))
                energy_interactions[i][j] = energy
                energy_interactions[j][i] = energy
                size_interactions[i][j] = size
                size_interactions[j][i] = size

        energies, diameters = zip(*parameters)
        return cls(
            energies,
            diameters,
            energy_interactions,
            size_interactions,
        )

    @property
    def component_count(self) -> int:
        """Number of mixture components."""

        return self.epsilon_over_k.shape[0]

    def _inputs(
        self,
        T: ArrayLike,
        value: ArrayLike,
        value_name: str,
        x: ArrayLike,
    ) -> tuple[Array, Array, Array]:
        temperature = _scalar_array(T, "T")
        scalar_value = _scalar_array(value, value_name)
        mole_fractions = _composition_array(x, self.component_count)
        dtype = jnp.result_type(
            temperature,
            scalar_value,
            mole_fractions,
            self.epsilon_over_k,
            self.molecular_diameters,
            self.energy_interaction_parameters,
            self.size_interaction_parameters,
            jnp.float32,
        )
        return (
            temperature.astype(dtype),
            scalar_value.astype(dtype),
            mole_fractions.astype(dtype),
        )

    def _mixture_parameters(self, mole_fractions: Array) -> tuple[Array, Array]:
        dtype = mole_fractions.dtype
        energies = self.epsilon_over_k.astype(dtype)
        diameters = self.molecular_diameters.astype(dtype)
        energy_interactions = self.energy_interaction_parameters.astype(dtype)
        size_interactions = self.size_interaction_parameters.astype(dtype)

        pair_energies = energy_interactions * jnp.sqrt(
            energies[:, None] * energies[None, :]
        )
        pair_diameters = size_interactions * 0.5 * (
            diameters[:, None] + diameters[None, :]
        )
        mixture_energy = jnp.einsum(
            "i,ij,j->",
            mole_fractions,
            pair_energies,
            mole_fractions,
        )
        mixture_diameter = jnp.einsum(
            "i,ij,j->",
            mole_fractions,
            pair_diameters,
            mole_fractions,
        )
        return mixture_energy, mixture_diameter

    @staticmethod
    def _temperature_coefficients(
        reduced_temperature: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        coefficients = jnp.asarray(
            _EOS_COEFFICIENTS,
            dtype=reduced_temperature.dtype,
        )
        inverse_temperature_squared = reduced_temperature**-2
        inverse_temperature_cubed = reduced_temperature**-3
        values = tuple(
            coefficients[index]
            + coefficients[index + 1] * inverse_temperature_squared
            + coefficients[index + 2] * inverse_temperature_cubed
            for index in (0, 3, 6, 9)
        )
        return (*values, coefficients)

    @staticmethod
    def _compressibility_factor(
        temperature: Array,
        reduced_density: Array,
        mixture_energy: Array,
    ) -> Array:
        reduced_temperature = (
            _REDUCED_TEMPERATURE_FACTOR * temperature / mixture_energy
        )
        b, c, d, e, coefficients = ZhangDuanEOS._temperature_coefficients(
            reduced_temperature
        )
        squared_density = reduced_density**2
        exponential = jnp.exp(-coefficients[14] * squared_density)
        return (
            1.0
            + b * reduced_density
            + c * squared_density
            + d * reduced_density**4
            + e * reduced_density**5
            + coefficients[12]
            / reduced_temperature**3
            * squared_density
            * (coefficients[13] + coefficients[14] * squared_density)
            * exponential
        )

    @staticmethod
    def _density_scale(mixture_diameter: Array) -> Array:
        """Return reduced density per molar density in m3 mol^-1.

        The logarithmic form prevents XLA from reassociating powers of the SI
        diameter into underflowing intermediates in float32.
        """

        diameter_ratio = mixture_diameter / _REFERENCE_DIAMETER
        return jnp.exp(3.0 * jnp.log(diameter_ratio)) / 1000.0

    def alphar(
        self,
        T: ArrayLike,
        rho: ArrayLike,
        x: ArrayLike,
    ) -> Array:
        """Return the reduced residual molar Helmholtz energy."""

        temperature, molar_density, mole_fractions = self._inputs(
            T,
            rho,
            "rho",
            x,
        )
        mixture_energy, mixture_diameter = self._mixture_parameters(mole_fractions)
        reduced_temperature = (
            _REDUCED_TEMPERATURE_FACTOR * temperature / mixture_energy
        )
        reduced_density = molar_density * self._density_scale(mixture_diameter)
        b, c, d, e, coefficients = self._temperature_coefficients(
            reduced_temperature
        )
        squared_density = reduced_density**2
        exponential_argument = coefficients[14] * squared_density
        exponential = jnp.exp(-exponential_argument)
        integral_exponential = (
            -(coefficients[13] + 1.0) * jnp.expm1(-exponential_argument)
            - exponential_argument * exponential
        )
        return (
            b * reduced_density
            + 0.5 * c * squared_density
            + 0.25 * d * reduced_density**4
            + 0.2 * e * reduced_density**5
            + coefficients[12]
            / (2.0 * coefficients[14] * reduced_temperature**3)
            * integral_exponential
        )

    def molar_density(
        self,
        T: ArrayLike,
        P: ArrayLike,
        x: ArrayLike,
        phase: str = "vapor",
    ) -> Array:
        """Return the stable low-density-branch root in mol m^-3."""

        if phase != "vapor":
            raise ValueError("ZhangDuanEOS supports only phase='vapor'.")
        temperature, pressure, mole_fractions = self._inputs(T, P, "P", x)
        mixture_energy, mixture_diameter = self._mixture_parameters(mole_fractions)
        density_scale = self._density_scale(mixture_diameter)
        reduced_pressure = (
            pressure * density_scale / (MOLAR_GAS_CONSTANT * temperature)
        )

        def residual(reduced_density):
            compressibility = self._compressibility_factor(
                temperature,
                reduced_density,
                mixture_energy,
            )
            return reduced_density * compressibility - reduced_pressure

        reduced_density = lax.custom_root(
            residual,
            reduced_pressure,
            _first_stable_root,
            _scalar_tangent_solve,
        )
        return reduced_density / density_scale

    def tree_flatten(self):
        """Return the JAX PyTree representation."""

        children = (
            self.epsilon_over_k,
            self.molecular_diameters,
            self.energy_interaction_parameters,
            self.size_interaction_parameters,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Restore a Zhang-Duan EOS from its JAX PyTree leaves."""

        del aux_data
        return cls(*children)
