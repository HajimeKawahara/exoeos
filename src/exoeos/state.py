"""Thermodynamic state returned by equation-of-state backends."""

from typing import NamedTuple

import jax


Array = jax.Array


class ThermodynamicState(NamedTuple):
    """Immutable, JAX-compatible thermodynamic state.

    Energies and heat capacities are molar quantities. Densities and all other
    fields use SI units. A ``NamedTuple`` is used so the state is automatically
    a JAX pytree and can cross ``jit``, ``vmap``, and differentiation boundaries.
    """

    compressibility_factor: Array
    mass_density: Array
    number_density: Array
    molar_enthalpy: Array
    molar_entropy: Array
    molar_heat_capacity_cp: Array
    molar_heat_capacity_cv: Array
    adiabatic_gradient: Array
    log_fugacity_coefficients: Array
    residual_gibbs: Array
    residual_enthalpy: Array
    thermal_expansion: Array

    @property
    def Z(self) -> Array:
        """Compressibility factor."""

        return self.compressibility_factor

    @property
    def h(self) -> Array:
        """Molar enthalpy in J mol^-1."""

        return self.molar_enthalpy

    @property
    def s(self) -> Array:
        """Molar entropy in J mol^-1 K^-1."""

        return self.molar_entropy

    @property
    def cp(self) -> Array:
        """Constant-pressure molar heat capacity in J mol^-1 K^-1."""

        return self.molar_heat_capacity_cp

    @property
    def cv(self) -> Array:
        """Constant-volume molar heat capacity in J mol^-1 K^-1."""

        return self.molar_heat_capacity_cv
