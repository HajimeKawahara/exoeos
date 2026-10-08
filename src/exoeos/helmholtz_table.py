"""Local C2 interpolation of a mass-specific Helmholtz potential."""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from exoeos._arrays import common_dtype, scalar_array
from exoeos.state import MassHelmholtzThermodynamicState


def _axis(values, name):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real.")
    axis = np.asarray(values, dtype=np.float64)
    if (
        axis.ndim != 1
        or axis.size < 3
        or not np.all(np.isfinite(axis))
        or np.any(axis <= 0.0)
        or np.any(np.diff(axis) <= 0.0)
    ):
        raise ValueError(
            f"{name} must contain at least three finite, positive, "
            "strictly increasing values."
        )
    return axis


def _field(values, shape, name):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real.")
    field = np.asarray(values, dtype=np.float64)
    if field.shape != shape or not np.all(np.isfinite(field)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return field


def _complete_derivatives(derivatives, log_temperature, log_density):
    """Share mixed third/fourth derivatives between neighboring cells."""
    derivatives[..., 2, 1] = 0.5 * (
        np.gradient(
            derivatives[..., 2, 0], log_density, axis=1, edge_order=2
        )
        + np.gradient(
            derivatives[..., 1, 1], log_temperature, axis=0, edge_order=2
        )
    )
    derivatives[..., 1, 2] = 0.5 * (
        np.gradient(
            derivatives[..., 0, 2], log_temperature, axis=0, edge_order=2
        )
        + np.gradient(
            derivatives[..., 1, 1], log_density, axis=1, edge_order=2
        )
    )
    derivatives[..., 2, 2] = 0.5 * (
        np.gradient(
            derivatives[..., 2, 1], log_density, axis=1, edge_order=2
        )
        + np.gradient(
            derivatives[..., 1, 2], log_temperature, axis=0, edge_order=2
        )
    )
    return derivatives


def _hermite_weights(fraction, width):
    """Quintic endpoint weights for values and first/second derivatives."""
    t = fraction
    u = 1.0 - t
    weights = jnp.stack(
        (
            jnp.stack(
                (
                    u**3 * (1.0 + 3.0 * t + 6.0 * t**2),
                    u**3 * t * (1.0 + 3.0 * t),
                    0.5 * u**3 * t**2,
                )
            ),
            jnp.stack(
                (
                    t**3 * (1.0 + 3.0 * u + 6.0 * u**2),
                    -t**3 * u * (1.0 + 3.0 * u),
                    0.5 * t**3 * u**2,
                )
            ),
        )
    )
    return weights * jnp.stack((jnp.ones_like(width), width, width**2))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HelmholtzTable:
    """Fixed-composition, mass-specific Helmholtz table in SI units.

    The host factories build shared nodal derivatives of ``q = a / T`` in
    ``(log(T), log(rho))``. Each cell uses a tensor-product quintic Hermite
    polynomial, giving a C2 potential across grid lines. Runtime evaluation
    and its derivatives use pure JAX. The potential includes the complete
    ideal and residual contributions, with no composition coordinate.

    Derivatives beyond second order can jump at cell boundaries. Smoothness
    and thermodynamic identities do not ensure stability or accuracy against
    the source data. Inconsistent source values can produce oscillations;
    reconstructed tables must be checked against their source, including
    inside cells. No stability projection or extrapolation is performed.

    ``temperatures`` and ``mass_densities`` are one-dimensional grids in K
    and kg m^-3. ``derivatives[i,j,k,l]`` stores the derivative of ``a/T``
    of logarithmic temperature order ``k`` and density order ``l``. Use the
    factories to validate and construct these arrays on the host.
    """

    temperatures: jax.Array
    mass_densities: jax.Array
    derivatives: jax.Array

    @classmethod
    def _from_derivatives(cls, temperatures, mass_densities, derivatives):
        arrays = tuple(
            jnp.asarray(value)
            for value in (temperatures, mass_densities, derivatives)
        )
        if not all(np.all(np.isfinite(np.asarray(value))) for value in arrays):
            raise ValueError(
                "Table data overflow in the active JAX dtype; enable "
                "jax_enable_x64 or restrict the table domain."
            )
        if any(
            np.any(np.diff(np.asarray(jnp.log(axis))) <= 0.0)
            or np.any(np.asarray(axis) <= 0.0)
            for axis in arrays[:2]
        ):
            raise ValueError(
                "Logarithmic grid values must remain strictly increasing "
                "in the active JAX dtype."
            )
        return cls(*arrays)

    @classmethod
    def from_helmholtz(
        cls,
        temperatures: ArrayLike,
        mass_densities: ArrayLike,
        specific_helmholtz: ArrayLike,
    ) -> "HelmholtzTable":
        """Interpolate a rectangular potential table [J kg^-1].

        The potential values are preserved at nodes. Shared derivatives are
        estimated locally using second-order finite differences on the log
        grids, including one-sided estimates at the outer boundaries.
        """
        temperature = _axis(temperatures, "temperatures")
        density = _axis(mass_densities, "mass_densities")
        shape = (temperature.size, density.size)
        potential = _field(specific_helmholtz, shape, "specific_helmholtz")
        x, y = np.log(temperature), np.log(density)
        derivatives = np.zeros((*shape, 3, 3), dtype=np.float64)
        derivatives[..., 0, 0] = potential / temperature[:, None]
        for order in (1, 2):
            derivatives[..., order, 0] = np.gradient(
                derivatives[..., order - 1, 0], x, axis=0, edge_order=2
            )
            derivatives[..., 0, order] = np.gradient(
                derivatives[..., 0, order - 1], y, axis=1, edge_order=2
            )
        derivatives[..., 1, 1] = np.gradient(
            derivatives[..., 0, 1], x, axis=0, edge_order=2
        )
        return cls._from_derivatives(
            temperature, density, _complete_derivatives(derivatives, x, y)
        )

    @classmethod
    def from_thermodynamic_data(
        cls,
        temperatures: ArrayLike,
        mass_densities: ArrayLike,
        pressure: ArrayLike,
        specific_internal_energy: ArrayLike,
        specific_entropy: ArrayLike,
        *,
        dlnrho_dlnT_P: Optional[ArrayLike] = None,
        dlnrho_dlnP_T: Optional[ArrayLike] = None,
        dlns_dlnT_P: Optional[ArrayLike] = None,
    ) -> "HelmholtzTable":
        """Reconstruct from P [Pa], u [J kg^-1], and s [J kg^-1 K^-1].

        Constrain nodal ``q=u/T-s``, ``q_x=-u/T`` and ``q_y=P/(rho*T)``.
        Thus P, u, and s are reproduced at nodes. If all three logarithmic
        response arrays are supplied, they determine the second derivatives;
        otherwise those derivatives are estimated locally from ``q_x,q_y``.
        Maxwell identities determine the remaining responses, which may
        differ from independently tabulated source values even at nodes.

        This is a local Hermite reconstruction with nodal constraints, not
        a least-squares fit or a guarantee of fidelity between nodes.
        """
        temperature = _axis(temperatures, "temperatures")
        density = _axis(mass_densities, "mass_densities")
        shape = (temperature.size, density.size)
        pressure = _field(pressure, shape, "pressure")
        energy = _field(specific_internal_energy, shape, "specific_internal_energy")
        entropy = _field(specific_entropy, shape, "specific_entropy")
        responses = (dlnrho_dlnT_P, dlnrho_dlnP_T, dlns_dlnT_P)
        if any(value is None for value in responses) and not all(
            value is None for value in responses
        ):
            raise ValueError("Supply all three response arrays or none of them.")
        x, y = np.log(temperature), np.log(density)
        energy_over_temperature = energy / temperature[:, None]
        pressure_over_rho_temperature = pressure / (
            temperature[:, None] * density[None, :]
        )
        derivatives = np.zeros((*shape, 3, 3), dtype=np.float64)
        derivatives[..., 0, 0] = energy_over_temperature - entropy
        derivatives[..., 1, 0] = -energy_over_temperature
        derivatives[..., 0, 1] = pressure_over_rho_temperature
        if all(value is None for value in responses):
            derivatives[..., 2, 0] = np.gradient(
                derivatives[..., 1, 0], x, axis=0, edge_order=2
            )
            derivatives[..., 0, 2] = np.gradient(
                derivatives[..., 0, 1], y, axis=1, edge_order=2
            )
            derivatives[..., 1, 1] = 0.5 * (
                np.gradient(derivatives[..., 1, 0], y, axis=1, edge_order=2)
                + np.gradient(derivatives[..., 0, 1], x, axis=0, edge_order=2)
            )
        else:
            alpha, beta, entropy_temperature = (
                _field(value, shape, name)
                for value, name in zip(
                    responses,
                    ("dlnrho_dlnT_P", "dlnrho_dlnP_T", "dlns_dlnT_P"),
                )
            )
            if np.any(beta == 0.0):
                raise ValueError("dlnrho_dlnP_T must be nonzero.")
            chi_temperature, chi_density = -alpha / beta, 1.0 / beta
            cv = entropy * entropy_temperature - (
                pressure_over_rho_temperature * chi_temperature**2 / chi_density
            )
            derivatives[..., 2, 0] = energy_over_temperature - cv
            derivatives[..., 1, 1] = pressure_over_rho_temperature * (
                chi_temperature - 1.0
            )
            derivatives[..., 0, 2] = pressure_over_rho_temperature * (
                chi_density - 1.0
            )
        return cls._from_derivatives(
            temperature, density, _complete_derivatives(derivatives, x, y)
        )

    def _inputs(self, T, mass_density):
        temperature = scalar_array(T, "T")
        density = scalar_array(mass_density, "mass_density")
        dtype = common_dtype(self, temperature, density)
        temperature, density = temperature.astype(dtype), density.astype(dtype)
        valid = (
            jnp.isfinite(temperature)
            & jnp.isfinite(density)
            & (temperature >= self.temperatures[0])
            & (temperature <= self.temperatures[-1])
            & (density >= self.mass_densities[0])
            & (density <= self.mass_densities[-1])
        )
        return temperature, density, valid

    def _scaled_potential(self, coordinates):
        x, y = coordinates
        log_temperature = jnp.log(self.temperatures)
        log_density = jnp.log(self.mass_densities)
        i = jnp.clip(
            jnp.searchsorted(log_temperature, x, side="right") - 1,
            0,
            log_temperature.size - 2,
        )
        j = jnp.clip(
            jnp.searchsorted(log_density, y, side="right") - 1,
            0,
            log_density.size - 2,
        )
        dx, dy = log_temperature[i + 1] - log_temperature[i], (
            log_density[j + 1] - log_density[j]
        )
        tx, ty = (x - log_temperature[i]) / dx, (y - log_density[j]) / dy
        temperature_weights = _hermite_weights(tx, dx)
        density_weights = _hermite_weights(ty, dy)
        patch = self.derivatives[
            i + jnp.arange(2)[:, None], j + jnp.arange(2)[None, :]
        ]
        # The value basis sums to one. Removing a nearby constant before AD
        # avoids cancellation of huge entropy offsets in pressure derivatives.
        anchor = patch[(tx > 0.5).astype(jnp.int32), (ty > 0.5).astype(jnp.int32), 0, 0]
        patch = patch.at[:, :, 0, 0].add(-anchor)
        return anchor + jnp.einsum("ik,jl,ijkl->", temperature_weights, density_weights, patch)

    def specific_helmholtz(self, T: ArrayLike, mass_density: ArrayLike) -> jax.Array:
        """Return total specific Helmholtz energy [J kg^-1], or NaN outside."""
        temperature, density, valid = self._inputs(T, mass_density)
        coordinates = jnp.log(jnp.stack((temperature, density)))
        potential = temperature * self._scaled_potential(coordinates)
        return jnp.where(valid & jnp.isfinite(potential), potential, jnp.nan)

    def state_trho(
        self, T: ArrayLike, mass_density: ArrayLike
    ) -> MassHelmholtzThermodynamicState:
        """Differentiate the single interpolated potential for one SI state."""
        temperature, density, valid = self._inputs(T, mass_density)

        def gradient_with_value(coordinates):
            value, gradient = jax.value_and_grad(self._scaled_potential)(coordinates)
            return gradient, (value, gradient)

        hessian, (value, gradient) = jax.jacfwd(
            gradient_with_value, has_aux=True
        )(jnp.log(jnp.stack((temperature, density))))
        state = MassHelmholtzThermodynamicState(
            temperature=temperature,
            mass_density=density,
            specific_helmholtz=temperature * value,
            specific_entropy=-value - gradient[0],
            specific_heat_capacity_cv=-gradient[0] - hessian[0, 0],
            pressure=density * temperature * gradient[1],
            pressure_temperature_derivative=density * (gradient[1] + hessian[0, 1]),
            pressure_density_derivative=temperature * (gradient[1] + hessian[1, 1]),
            specific_internal_energy=-temperature * gradient[0],
        )
        valid = valid & jnp.all(jnp.isfinite(jnp.stack(state)))
        return MassHelmholtzThermodynamicState(
            *(jnp.where(valid, field, jnp.nan) for field in state)
        )

    def state_tp(
        self, T: ArrayLike, P: ArrayLike, *, density_bounds: ArrayLike
    ) -> MassHelmholtzThermodynamicState:
        """Invert the same potential on an explicitly supplied density bracket.

        ``density_bounds=(lower, upper)`` uses kg m^-3 and must lie inside
        the table. The caller must choose a single-phase interval where
        pressure increases with density. This is not a phase/root selector.
        Failed brackets and nonpositive compressibility at the root return
        an all-NaN state. Bisection uses log density; JAX derivatives use the
        implicit pressure equation, including derivatives of table leaves.
        """
        temperature = scalar_array(T, "T")
        pressure = scalar_array(P, "P")
        bounds = jnp.asarray(density_bounds)
        if bounds.shape != (2,):
            raise ValueError("density_bounds must have shape (2,).")
        dtype = common_dtype(self, temperature, pressure, bounds)
        temperature, pressure, bounds = (
            value.astype(dtype) for value in (temperature, pressure, bounds)
        )
        low, high = jnp.log(bounds)
        scale = jnp.maximum(jnp.abs(pressure), 1.0)

        def equation(log_density):
            gradient = jax.grad(self._scaled_potential)(
                jnp.stack((jnp.log(temperature), log_density))
            )
            return (jnp.exp(log_density) * temperature * gradient[1] - pressure) / scale

        def solve(function, initial):
            del initial

            def step(_, interval):
                lower, upper = interval
                midpoint = lower + 0.5 * (upper - lower)
                below = function(midpoint) < 0.0
                return (jnp.where(below, midpoint, lower),
                        jnp.where(below, upper, midpoint))

            lower, upper = jax.lax.fori_loop(0, 64, step, (low, high))
            return lower + 0.5 * (upper - lower)

        root = jax.lax.custom_root(
            equation, 0.5 * (low + high), solve,
            lambda linear, value: value / linear(jnp.ones_like(value)),
        )
        density = jnp.exp(root)
        # Correct endpoint roundoff without clipping implicit derivatives.
        density += jax.lax.stop_gradient(jnp.clip(density, bounds[0], bounds[1]) - density)
        state = self.state_trho(temperature, density)
        tolerance = 128.0 * jnp.finfo(dtype).eps
        valid = (
            jnp.isfinite(pressure) & (pressure > 0.0)
            & jnp.all(jnp.isfinite(bounds))
            & (bounds[0] >= self.mass_densities[0])
            & (bounds[1] <= self.mass_densities[-1])
            & (bounds[0] < bounds[1])
            & (equation(low) <= tolerance)
            & (equation(high) >= -tolerance)
            & (state.pressure_density_derivative > 0.0)
            & (jnp.abs(state.pressure - pressure) <= tolerance * scale)
        )
        return MassHelmholtzThermodynamicState(
            *(jnp.where(valid, field, jnp.nan) for field in state)
        )

    def tree_flatten(self):
        return (self.temperatures, self.mass_densities, self.derivatives), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del auxiliary
        return cls(*children)
