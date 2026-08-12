"""Thermodynamic states returned by ExoEOS evaluators."""

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


class TRhoState(NamedTuple):
    """Residual state evaluated at temperature and molar density."""

    molar_density: Array
    pressure: Array
    compressibility_factor: Array
    reduced_residual_helmholtz: Array
    reduced_residual_chemical_potentials: Array
    log_fugacity_coefficients: Array
    reduced_residual_gibbs: Array

    @property
    def rho(self) -> Array:
        """Total molar density in mol m^-3."""

        return self.molar_density

    @property
    def P(self) -> Array:
        """Pressure in Pa."""

        return self.pressure

    @property
    def Z(self) -> Array:
        """Compressibility factor."""

        return self.compressibility_factor

    @property
    def alphar(self) -> Array:
        """Reduced residual molar Helmholtz energy."""

        return self.reduced_residual_helmholtz

    @property
    def mu_res_RT(self) -> Array:
        """Reduced residual chemical potentials."""

        return self.reduced_residual_chemical_potentials

    @property
    def lnphi(self) -> Array:
        """Logarithmic fugacity coefficients."""

        return self.log_fugacity_coefficients

    @property
    def gres_RT(self) -> Array:
        """Reduced residual molar Gibbs energy."""

        return self.reduced_residual_gibbs


class SolutionState(NamedTuple):
    """Excess state evaluated at temperature, pressure, and composition."""

    reduced_excess_gibbs: Array
    log_activity_coefficients: Array

    @property
    def gex_RT(self) -> Array:
        """Reduced molar excess Gibbs energy."""

        return self.reduced_excess_gibbs

    @property
    def lngamma(self) -> Array:
        """Logarithmic activity coefficients."""

        return self.log_activity_coefficients
