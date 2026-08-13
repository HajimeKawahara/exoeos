"""Peng-Robinson cubic equation of state."""

from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax, tree_util
from jax.typing import ArrayLike

from exoeos.constants import MOLAR_GAS_CONSTANT


Array = jax.Array

_ATTRACTION_CONSTANT = 0.45724
_COVOLUME_CONSTANT = 0.07780
_SQRT_TWO = 2.0**0.5


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


def _compressibility_factor_impl(A: Array, B: Array, phase: str) -> Array:
    """Return the selected real root of the Peng-Robinson cubic."""

    if phase not in ("vapor", "liquid"):
        raise ValueError("phase must be 'vapor' or 'liquid'.")

    quadratic = B - 1.0
    linear = A - 3.0 * B**2 - 2.0 * B
    constant = B**3 + B**2 - A * B
    depressed_linear = linear - quadratic**2 / 3.0
    depressed_constant = (
        2.0 * quadratic**3 / 27.0 - quadratic * linear / 3.0 + constant
    )
    cubic_discriminant = (
        18.0 * quadratic * linear * constant
        - 4.0 * quadratic**3 * constant
        + quadratic**2 * linear**2
        - 4.0 * linear**3
        - 27.0 * constant**2
    )
    discriminant = -cubic_discriminant / 108.0

    def one_real_root(_) -> Array:
        half_constant = -depressed_constant / 2.0
        root_term = jnp.cbrt(
            half_constant + jnp.copysign(jnp.sqrt(discriminant), half_constant)
        )
        depressed_root = root_term - depressed_linear / (3.0 * root_term)
        return depressed_root - quadratic / 3.0

    def repeated_or_three_real_roots(_) -> Array:
        def repeated_root(_) -> Array:
            return -quadratic / 3.0

        def three_real_roots(_) -> Array:
            radius = jnp.sqrt(-depressed_linear / 3.0)
            angle = (
                jnp.arctan2(
                    jnp.sqrt(jnp.maximum(-discriminant, 0.0)),
                    -depressed_constant / 2.0,
                )
                / 3.0
            )
            shift = -quadratic / 3.0
            common_small_root = (
                linear / (3.0 * (shift + radius))
                + 2.0 * radius * jnp.sin(angle / 2.0) ** 2
            )
            root_separation = jnp.sqrt(3.0) * radius * jnp.sin(angle)
            roots = jnp.stack(
                (
                    shift + 2.0 * radius * jnp.cos(angle),
                    common_small_root - root_separation,
                    common_small_root + root_separation,
                )
            )
            if phase == "vapor":
                return jnp.max(jnp.where(roots > B, roots, -jnp.inf))
            return jnp.min(jnp.where(roots > B, roots, jnp.inf))

        return lax.cond(
            depressed_linear < 0.0,
            three_real_roots,
            repeated_root,
            operand=None,
        )

    return lax.cond(
        discriminant > 0.0,
        one_real_root,
        repeated_or_three_real_roots,
        operand=None,
    )


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _compressibility_factor(A: Array, B: Array, phase: str) -> Array:
    return _compressibility_factor_impl(A, B, phase)


@_compressibility_factor.defjvp
def _compressibility_factor_jvp(
    phase: str,
    primals: tuple[Array, Array],
    tangents: tuple[Array, Array],
) -> tuple[Array, Array]:
    """Differentiate a simple cubic root by the implicit function theorem."""

    A, B = primals
    A_tangent, B_tangent = tangents
    root = _compressibility_factor(A, B, phase)
    polynomial_derivative = (
        3.0 * root**2 + 2.0 * (B - 1.0) * root + A - 3.0 * B**2 - 2.0 * B
    )
    A_derivative = root - B
    B_derivative = root**2 - (6.0 * B + 2.0) * root + 3.0 * B**2 + 2.0 * B - A
    root_tangent = (
        -(A_derivative * A_tangent + B_derivative * B_tangent) / polynomial_derivative
    )
    return root, root_tangent


@tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class PengRobinsonEOS:
    """Classical Peng-Robinson EOS with quadratic mixture parameters.

    Critical temperatures are in K and critical pressures are in Pa. Binary
    interaction parameters use ``k_ij`` in ``a_ij = (1 - k_ij) sqrt(a_i a_j)``.
    Numerical validity, normalized compositions, and matrix symmetry are caller
    contracts; static shapes are checked by the implementation.
    """

    critical_temperatures: Array
    critical_pressures: Array
    acentric_factors: Array
    binary_interaction_parameters: Array

    def __init__(
        self,
        critical_temperatures: ArrayLike,
        critical_pressures: ArrayLike,
        acentric_factors: ArrayLike,
        binary_interaction_parameters: Optional[ArrayLike] = None,
    ) -> None:
        temperatures = _component_array(
            critical_temperatures,
            "critical_temperatures",
        )
        pressures = _component_array(critical_pressures, "critical_pressures")
        factors = _component_array(acentric_factors, "acentric_factors")
        if pressures.shape != temperatures.shape:
            raise ValueError(
                f"critical_pressures must have shape {temperatures.shape}."
            )
        if factors.shape != temperatures.shape:
            raise ValueError(f"acentric_factors must have shape {temperatures.shape}.")

        component_count = temperatures.shape[0]
        if binary_interaction_parameters is None:
            interactions = jnp.zeros(
                (component_count, component_count),
                dtype=jnp.result_type(temperatures, pressures, factors),
            )
        else:
            interactions = jnp.asarray(binary_interaction_parameters)
            if not jnp.issubdtype(interactions.dtype, jnp.inexact):
                interactions = interactions.astype(jnp.asarray(1.0).dtype)
            if interactions.shape != (component_count, component_count):
                raise ValueError(
                    "binary_interaction_parameters must have shape "
                    f"({component_count}, {component_count})."
                )

        object.__setattr__(self, "critical_temperatures", temperatures)
        object.__setattr__(self, "critical_pressures", pressures)
        object.__setattr__(self, "acentric_factors", factors)
        object.__setattr__(self, "binary_interaction_parameters", interactions)

    @property
    def component_count(self) -> int:
        """Number of mixture components."""

        return self.critical_temperatures.shape[0]

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
            self.critical_temperatures,
            self.critical_pressures,
            self.acentric_factors,
            self.binary_interaction_parameters,
            jnp.float32,
        )
        return (
            temperature.astype(dtype),
            scalar_value.astype(dtype),
            mole_fractions.astype(dtype),
        )

    def _mixture_parameters(
        self,
        temperature: Array,
        mole_fractions: Array,
    ) -> tuple[Array, Array]:
        dtype = temperature.dtype
        critical_temperatures = self.critical_temperatures.astype(dtype)
        critical_pressures = self.critical_pressures.astype(dtype)
        acentric_factors = self.acentric_factors.astype(dtype)
        interactions = self.binary_interaction_parameters.astype(dtype)

        kappa = 0.37464 + 1.54226 * acentric_factors - 0.26992 * acentric_factors**2
        alpha = (
            1.0 + kappa * (1.0 - jnp.sqrt(temperature / critical_temperatures))
        ) ** 2
        component_attraction = (
            _ATTRACTION_CONSTANT
            * MOLAR_GAS_CONSTANT**2
            * critical_temperatures**2
            * alpha
            / critical_pressures
        )
        component_covolume = (
            _COVOLUME_CONSTANT
            * MOLAR_GAS_CONSTANT
            * critical_temperatures
            / critical_pressures
        )
        pair_attraction = (1.0 - interactions) * jnp.sqrt(
            component_attraction[:, None] * component_attraction[None, :]
        )
        mixture_attraction = jnp.einsum(
            "i,ij,j->",
            mole_fractions,
            pair_attraction,
            mole_fractions,
        )
        mixture_covolume = jnp.dot(mole_fractions, component_covolume)
        return mixture_attraction, mixture_covolume

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
        attraction, covolume = self._mixture_parameters(
            temperature,
            mole_fractions,
        )
        reduced_density = covolume * molar_density
        attraction_log = jnp.log1p((1.0 + _SQRT_TWO) * reduced_density) - jnp.log1p(
            (1.0 - _SQRT_TWO) * reduced_density
        )
        return -jnp.log1p(-reduced_density) - attraction * attraction_log / (
            2.0 * _SQRT_TWO * covolume * MOLAR_GAS_CONSTANT * temperature
        )

    def molar_density(
        self,
        T: ArrayLike,
        P: ArrayLike,
        x: ArrayLike,
        phase: str = "vapor",
    ) -> Array:
        """Return the molar density of the selected cubic root in mol m^-3."""

        if phase not in ("vapor", "liquid"):
            raise ValueError("phase must be 'vapor' or 'liquid'.")
        temperature, pressure, mole_fractions = self._inputs(T, P, "P", x)
        attraction, covolume = self._mixture_parameters(
            temperature,
            mole_fractions,
        )
        reduced_attraction = (
            attraction * pressure / (MOLAR_GAS_CONSTANT * temperature) ** 2
        )
        reduced_covolume = covolume * pressure / (MOLAR_GAS_CONSTANT * temperature)
        compressibility = _compressibility_factor(
            reduced_attraction,
            reduced_covolume,
            phase,
        )
        return pressure / (compressibility * MOLAR_GAS_CONSTANT * temperature)

    def tree_flatten(self):
        """Return the JAX PyTree representation."""

        children = (
            self.critical_temperatures,
            self.critical_pressures,
            self.acentric_factors,
            self.binary_interaction_parameters,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Restore a Peng-Robinson EOS from its JAX PyTree leaves."""

        del aux_data
        return cls(*children)
