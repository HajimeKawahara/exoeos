"""Thermodynamic states returned by ExoEOS evaluators."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from exoeos.constants import AVOGADRO_CONSTANT, MOLAR_GAS_CONSTANT


Array = jax.Array


class HelmholtzThermodynamicState(NamedTuple):
    """Total single-phase state from a molar Helmholtz potential.

    All derivatives hold composition fixed. Stored fields use SI units;
    ``rho`` is molar density and heat capacities are molar. Derived
    properties reuse the stored first and second potential derivatives.
    No stability clipping is performed at singular or unstable states.
    """

    temperature: Array
    molar_density: Array
    mean_molar_mass: Array
    molar_helmholtz: Array
    molar_entropy: Array
    molar_heat_capacity_cv: Array
    pressure: Array
    pressure_temperature_derivative: Array
    pressure_density_derivative: Array

    @property
    def molar_internal_energy(self) -> Array:
        """Internal energy in J mol^-1."""
        return self.molar_helmholtz + self.temperature * self.molar_entropy

    @property
    def molar_enthalpy(self) -> Array:
        """Enthalpy in J mol^-1."""
        return self.molar_internal_energy + self.pressure / self.molar_density

    @property
    def molar_gibbs(self) -> Array:
        """Gibbs energy in J mol^-1."""
        return self.molar_helmholtz + self.pressure / self.molar_density

    @property
    def molar_heat_capacity_cp(self) -> Array:
        """Constant-pressure heat capacity in J mol^-1 K^-1."""
        return self.molar_heat_capacity_cv + (
            self.temperature * self.pressure_temperature_derivative**2
            / (self.molar_density**2 * self.pressure_density_derivative)
        )

    @property
    def compressibility_factor(self) -> Array:
        """Dimensionless compressibility factor."""
        return self.pressure / (
            self.molar_density * MOLAR_GAS_CONSTANT * self.temperature
        )

    @property
    def mass_density(self) -> Array:
        """Mass density in kg m^-3."""
        return self.molar_density * self.mean_molar_mass

    @property
    def number_density(self) -> Array:
        """Particle number density in m^-3."""
        return self.molar_density * AVOGADRO_CONSTANT

    @property
    def thermal_expansion(self) -> Array:
        """Isobaric volumetric expansion coefficient in K^-1."""
        return self.pressure_temperature_derivative / (
            self.molar_density * self.pressure_density_derivative
        )

    @property
    def isothermal_compressibility(self) -> Array:
        """Isothermal compressibility in Pa^-1."""
        return 1.0 / (self.molar_density * self.pressure_density_derivative)

    @property
    def sound_speed_squared(self) -> Array:
        """Squared frozen-composition isentropic sound speed in m^2 s^-2."""
        return (
            self.pressure_density_derivative
            + self.temperature * self.pressure_temperature_derivative**2
            / (self.molar_density**2 * self.molar_heat_capacity_cv)
        ) / self.mean_molar_mass

    @property
    def sound_speed(self) -> Array:
        """Frozen-composition isentropic sound speed in m s^-1."""
        return jnp.sqrt(self.sound_speed_squared)

    @property
    def isentropic_compressibility(self) -> Array:
        """Isentropic compressibility in Pa^-1."""
        return 1.0 / (self.mass_density * self.sound_speed_squared)

    @property
    def adiabatic_gradient(self) -> Array:
        """Dimensionless ``(d ln T / d ln P)_(s,x)``."""
        return self.pressure * self.thermal_expansion / (
            self.molar_density * self.molar_heat_capacity_cp
        )

    rho = property(lambda self: self.molar_density, doc="Molar density in mol m^-3.")
    P = property(lambda self: self.pressure, doc="Pressure in Pa.")
    a = property(lambda self: self.molar_helmholtz, doc="Helmholtz energy in J mol^-1.")
    u = property(lambda self: self.molar_internal_energy, doc="Internal energy in J mol^-1.")
    h = property(lambda self: self.molar_enthalpy, doc="Enthalpy in J mol^-1.")
    g = property(lambda self: self.molar_gibbs, doc="Gibbs energy in J mol^-1.")
    s = property(lambda self: self.molar_entropy, doc="Entropy in J mol^-1 K^-1.")
    cv = property(lambda self: self.molar_heat_capacity_cv, doc="Molar cv in J mol^-1 K^-1.")
    cp = property(lambda self: self.molar_heat_capacity_cp, doc="Molar cp in J mol^-1 K^-1.")
    Z = property(lambda self: self.compressibility_factor, doc="Compressibility factor.")
    nabla_ad = property(lambda self: self.adiabatic_gradient, doc="Adiabatic gradient.")


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


class MassThermodynamicState(NamedTuple):
    """Immutable mass-specific thermodynamic state in SI units."""

    pressure: Array
    mass_density: Array
    specific_internal_energy: Array
    specific_entropy: Array
    dlnrho_dlnT_P: Array
    dlnrho_dlnP_T: Array
    dlns_dlnT_P: Array
    dlns_dlnP_T: Array
    adiabatic_gradient: Array

    @property
    def P(self) -> Array:
        """Pressure in Pa."""

        return self.pressure

    @property
    def rho(self) -> Array:
        """Mass density in kg m^-3."""

        return self.mass_density

    @property
    def u(self) -> Array:
        """Specific internal energy in J kg^-1."""

        return self.specific_internal_energy

    @property
    def s(self) -> Array:
        """Specific entropy in J kg^-1 K^-1."""

        return self.specific_entropy

    @property
    def nabla_ad(self) -> Array:
        """Adiabatic logarithmic temperature gradient."""

        return self.adiabatic_gradient


class SilicateHydrogenState(NamedTuple):
    """Immutable mass-specific silicate-hydrogen state in SI units."""

    pressure: Array
    mass_density: Array
    specific_enthalpy: Array
    specific_entropy: Array
    thermal_expansion: Array
    specific_heat_capacity_cp: Array
    adiabatic_bulk_modulus: Array
    reference_mass_density: Array
    compression_ratio: Array
    gruneisen_parameter: Array

    @property
    def P(self) -> Array:
        """Pressure in Pa."""

        return self.pressure

    @property
    def rho(self) -> Array:
        """Mass density in kg m^-3."""

        return self.mass_density

    @property
    def h(self) -> Array:
        """Specific enthalpy in J kg^-1."""

        return self.specific_enthalpy

    @property
    def s(self) -> Array:
        """Specific entropy in J kg^-1 K^-1."""

        return self.specific_entropy

    @property
    def alpha(self) -> Array:
        """Thermal expansion coefficient in K^-1."""

        return self.thermal_expansion

    @property
    def cp(self) -> Array:
        """Specific heat capacity in J kg^-1 K^-1."""

        return self.specific_heat_capacity_cp

    @property
    def Ks(self) -> Array:
        """Adiabatic bulk modulus in Pa."""

        return self.adiabatic_bulk_modulus

    @property
    def rho0(self) -> Array:
        """Reference mass density in kg m^-3."""

        return self.reference_mass_density

    @property
    def eta(self) -> Array:
        """Compression ratio."""

        return self.compression_ratio

    @property
    def gamma(self) -> Array:
        """Grueneisen parameter."""

        return self.gruneisen_parameter

    @property
    def specific_internal_energy(self) -> Array:
        """Specific internal energy in J kg^-1."""

        return self.specific_enthalpy - self.pressure / self.mass_density

    @property
    def u(self) -> Array:
        """Specific internal energy in J kg^-1."""

        return self.specific_internal_energy

    @property
    def adiabatic_gradient(self) -> Array:
        """Adiabatic logarithmic temperature gradient."""

        return (
            self.thermal_expansion
            * self.pressure
            / (self.mass_density * self.specific_heat_capacity_cp)
        )

    @property
    def nabla_ad(self) -> Array:
        """Adiabatic logarithmic temperature gradient."""

        return self.adiabatic_gradient


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


class TotalSolutionState(NamedTuple):
    """Extensive solution G/(RT) [mol] and component mu/(RT) [dimensionless]."""

    gibbs_RT: Array
    mu_RT: Array
