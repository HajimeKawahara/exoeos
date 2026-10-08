"""Fixed-composition thermodynamics from a total molar Helmholtz energy."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from exoeos._arrays import common_dtype, scalar_array, vector_array
from exoeos.constants import MOLAR_GAS_CONSTANT
from exoeos.contracts import HelmholtzEOS, MolarHelmholtzEOS
from exoeos.state import HelmholtzThermodynamicState


def _inputs(model, T, density_or_pressure, x, coordinate_name):
    temperature = scalar_array(T, "T")
    coordinate = scalar_array(density_or_pressure, coordinate_name)
    composition = vector_array(x, "x")
    masses = vector_array(model.molar_masses, "molar_masses")
    if masses.shape != composition.shape:
        raise ValueError("x and molar_masses must have the same component shape.")
    dtype = common_dtype(model, temperature, coordinate, composition, masses)
    return tuple(
        value.astype(dtype) for value in (temperature, coordinate, composition, masses)
    )


def thermodynamic_state_trho(
    model: MolarHelmholtzEOS,
    T: ArrayLike,
    rho: ArrayLike,
    x: ArrayLike,
) -> HelmholtzThermodynamicState:
    """Differentiate a total Helmholtz potential at fixed composition.

    Args:
        model: Model providing ``molar_helmholtz(T, rho, x)`` in J mol^-1
            and component ``molar_masses`` in kg mol^-1. The energy must
            include the complete ideal contribution, including mixing.
        T: Scalar temperature in K.
        rho: Scalar total molar density in mol m^-3.
        x: Normalized mole fractions with shape ``(K,)``.

    Returns:
        Total caloric and response properties in SI units. Heat capacities
        and energies are molar; sound speed is in m s^-1.

    Only temperature and density are differentiated internally. Use
    ``jax.vmap`` for batches. Numerical validity is a caller contract:
    positive temperature, density and masses, normalized nonnegative
    composition, and a smooth potential. Stable response quantities require
    positive cv and ``(dP/drho)_(T,x)``. No clipping or phase equilibration is
    performed. Further JAX derivatives require a correspondingly smooth model.
    """

    temperature, molar_density, composition, masses = _inputs(model, T, rho, x, "rho")

    def energy(coordinates):
        value = jnp.asarray(
            model.molar_helmholtz(coordinates[0], coordinates[1], composition)
        )
        if value.ndim != 0:
            raise ValueError("model.molar_helmholtz must return a scalar for one state.")
        return value

    def gradient_with_value(coordinates):
        value, gradient = jax.value_and_grad(energy)(coordinates)
        return gradient, (value, gradient)

    hessian, (value, gradient) = jax.jacfwd(gradient_with_value, has_aux=True)(
        jnp.stack((temperature, molar_density))
    )
    return HelmholtzThermodynamicState(
        temperature=temperature,
        molar_density=molar_density,
        mean_molar_mass=jnp.dot(composition, masses),
        molar_helmholtz=value,
        molar_entropy=-gradient[0],
        molar_heat_capacity_cv=-temperature * hessian[0, 0],
        pressure=molar_density**2 * gradient[1],
        pressure_temperature_derivative=molar_density**2 * hessian[0, 1],
        pressure_density_derivative=(
            2.0 * molar_density * gradient[1] + molar_density**2 * hessian[1, 1]
        ),
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HelmholtzThermodynamics:
    """Combine a residual EOS with an ideal caloric Helmholtz model.

    ``residual.alphar`` supplies the reduced residual energy. ``ideal``
    supplies ``molar_helmholtz`` and ``molar_masses``, for example an
    :class:`~exoeos.IdealGas`. Its density dependence must be that of an ideal
    gas: ``da0/drho = R T / rho``. Both models must use the same component
    ordering. Their registered PyTree leaves remain differentiable.

    Temperature-pressure evaluation requires the residual model's
    ``molar_density`` hook. Root selection is delegated to that model.
    """

    residual: HelmholtzEOS
    ideal: MolarHelmholtzEOS

    @property
    def molar_masses(self) -> jax.Array:
        """Component molar masses in kg mol^-1."""
        return self.ideal.molar_masses

    def molar_helmholtz(
        self, T: ArrayLike, rho: ArrayLike, x: ArrayLike
    ) -> jax.Array:
        """Return total molar Helmholtz energy in J mol^-1."""
        ideal = jnp.asarray(self.ideal.molar_helmholtz(T, rho, x))
        residual = jnp.asarray(self.residual.alphar(T, rho, x))
        if ideal.ndim != 0 or residual.ndim != 0:
            raise ValueError("Helmholtz models must return a scalar for one state.")
        return ideal + MOLAR_GAS_CONSTANT * T * residual

    def state_trho(
        self, T: ArrayLike, rho: ArrayLike, x: ArrayLike
    ) -> HelmholtzThermodynamicState:
        """Evaluate one state at temperature [K] and molar density [mol m^-3]."""
        return thermodynamic_state_trho(self, T, rho, x)

    def state_tp(
        self,
        T: ArrayLike,
        P: ArrayLike,
        x: ArrayLike,
        phase: str = "vapor",
    ) -> HelmholtzThermodynamicState:
        """Evaluate one state at temperature [K] and pressure [Pa].

        ``phase`` is a static root selector interpreted by the residual EOS.
        The Helmholtz derivatives hold density fixed after inversion; they
        do not differentiate along the temperature-pressure solution curve.
        """
        temperature, pressure, composition, _ = _inputs(self, T, P, x, "P")
        density = self.residual.molar_density(
            temperature, pressure, composition, phase=phase
        )
        return self.state_trho(temperature, density, composition)

    state = state_tp

    def tree_flatten(self):
        return (self.residual, self.ideal), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)
