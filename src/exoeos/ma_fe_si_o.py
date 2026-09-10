"""Ma Fe-Si-O liquid with Young (2023)'s printed interaction parameters."""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from jax.scipy.special import xlog1py
from jax.typing import ArrayLike

from exoeos._arrays import as_inexact_array, common_dtype, scalar_array


@tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class MaFeSiOLiquid:
    """Fe-rich alloy in symmetric, formally continued endmember standards.

    ``interaction_K`` contains ``(Si-Si, O-O, Si-O)`` coefficients in K;
    the dimensionless interactions are ``interaction_K / T``. Defaults use
    Young's printed equations, completed with the Ma solvent Fe term.
    Custom coefficients are uncalibrated parameter experiments.

    Call ``validate_state`` at the eager input boundary. Traced evaluation
    checks shapes only and requires normalized, nonnegative ``(Fe, Si, O)``
    mole fractions with Fe > 0 and finite positive T/P. Pressure dependence
    and phase stability are not modeled. Pure Si/O scalar limits are zero,
    but their full activity vectors and composition derivatives are undefined.
    """

    interaction_K: jax.Array

    components: ClassVar[tuple[str, ...]] = ("Fe", "Si", "O")
    activity_basis: ClassVar[str] = "mole_fraction"
    standard_state_convention: ClassVar[str] = "symmetric"
    reference_model_id: ClassVar[str] = "ma2001_fe_si_o_young2023_printed_v1"
    reference_pressure_Pa: ClassVar[float] = 1.0e5

    def __init__(
        self,
        interaction_K: ArrayLike = (12.41 * 1873.0, -16500.0, -5.0 * 1873.0),
    ) -> None:
        coefficients = as_inexact_array(interaction_K)
        if coefficients.shape != (3,):
            raise ValueError("interaction_K must have shape (3,) in Si-Si, O-O, Si-O order.")
        object.__setattr__(self, "interaction_K", coefficients)

    def _inputs(self, T: ArrayLike, P: ArrayLike, x: ArrayLike):
        temperature = scalar_array(T, "T")
        pressure = scalar_array(P, "P")
        composition = as_inexact_array(x)
        if composition.shape != (3,):
            raise ValueError("x must have shape (3,) in Fe, Si, O order.")
        dtype = common_dtype(self, temperature, pressure, composition)
        return tuple(value.astype(dtype) for value in (temperature, pressure, composition))

    def gex_RT(self, T: ArrayLike, P: ArrayLike, x: ArrayLike) -> jax.Array:
        """Return scalar molar ``gE/(RT)`` at T [K], P [Pa], and mole fractions.

        The formal symmetric free energy excludes ideal mixing and standard
        potentials. ``solution_state`` differentiates its extensive form to
        recover all three natural-log activity coefficients, including Fe.
        """
        temperature, _, composition = self._inputs(T, P, x)
        solutes = composition[1:]
        si, oxygen = solutes
        self_terms = xlog1py(1 - solutes, -solutes)

        # At a pure solute, (si*oxygen)^2/(1-solute) has limit zero within
        # the nonnegative simplex. Only its exact endpoint denominator is
        # replaced; fractions are never clipped or given artificial traces.
        complements = jnp.where(solutes == 1, jnp.ones_like(solutes), 1 - solutes)
        product = si * oxygen
        pair = (
            -product - xlog1py(si, -oxygen) - xlog1py(oxygen, -si)
            + product**2 * (jnp.sum(1 / complements) - 1) / 2
        )
        terms = jnp.concatenate((self_terms, jnp.asarray([pair])))
        return jnp.dot(self.interaction_K / temperature, terms)

    def standard_state_shift_RT(self, T: ArrayLike) -> jax.Array:
        """Return ``(mu0_formal - mu0_source)/(RT)`` in Fe, Si, O order.

        Add this shift to source standard potentials divided by RT. Add it
        to formal ``lngamma`` to reconstruct the completed source convention.
        The source infinite-dilution constants remain those of Young (2023);
        the self interactions are taken from this instance. No absolute
        chemical potentials or Henry/weight-percent inputs are supplied.
        """
        temperature = scalar_array(T, "T")
        dtype = common_dtype(self, temperature)
        temperature = temperature.astype(dtype)
        source_limits = jnp.asarray([-6.65 * 1873.0, -16500.0], dtype=dtype) / temperature
        source_limits = source_limits + jnp.asarray([0.0, 4.29], dtype=dtype)
        shifts = source_limits + self.interaction_K[:2] / temperature
        return jnp.concatenate((jnp.zeros((1,), dtype=dtype), shifts))

    def validate_state(self, T: ArrayLike, P: ArrayLike, x: ArrayLike) -> None:
        """Raise ValueError for inputs outside the activity domain, before tracing.

        This eager NumPy check accepts normalization within eight machine
        epsilons of the input composition dtype. It does not normalize or
        alter inputs, check phase stability, or certify experimental validity.
        It is not called implicitly by ``gex_RT`` or ``solution_state``.
        """
        temperature, pressure, composition = map(np.asarray, self._inputs(T, P, x))
        coefficients = np.asarray(self.interaction_K)
        values = (temperature, pressure, composition, coefficients)
        if not all(np.isrealobj(value) and np.all(np.isfinite(value)) for value in values):
            raise ValueError("T, P, x, and interaction_K must be finite and real.")
        if temperature <= 0 or pressure <= 0:
            raise ValueError("T and P must be positive.")
        if np.any(composition < 0) or composition[0] <= 0:
            raise ValueError("Activity evaluation requires nonnegative mole fractions and Fe > 0.")
        # Tiny positive Fe can still leave a solute at a represented singularity.
        if np.any(composition[1:] >= 1) or np.any(composition[1:] >= composition.sum()):
            raise ValueError("Solute fractions must remain below one before and after normalization.")
        tolerance = 8 * jnp.finfo(as_inexact_array(x).dtype).eps
        if not np.isclose(np.sum(composition), 1.0, rtol=0.0, atol=tolerance):
            raise ValueError("x must sum to one; inputs are not normalized by validate_state.")

    def tree_flatten(self):
        """Return the single differentiable interaction-parameter leaf."""
        return (self.interaction_K,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Restore a model without validating JAX transformation placeholders."""
        del aux_data
        model = object.__new__(cls)
        object.__setattr__(model, "interaction_K", children[0])
        return model
