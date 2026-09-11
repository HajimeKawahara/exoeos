"""Formal Fe-Si-O-H control with no excess interactions involving H."""

from dataclasses import dataclass, field
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from jax.typing import ArrayLike

from exoeos._arrays import as_inexact_array, common_dtype, scalar_array
from exoeos.ma_fe_si_o import MaFeSiOLiquid


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MaFeSiOHLiquid:
    """Extend a dry Ma alloy by ideal H dilution on an atomic amount basis.

    The scalar is ``gE/(RT) = (1-x_H) * dry_model.gex_RT(T, P, y)``,
    where ``y = x[:3] / sum(x[:3])``. All four ideal activities use ``x``.
    This integrable control has no fitted H interactions, pressure response,
    absolute standard potentials, or established H partition calibration.

    Call ``validate_state`` before tracing. The domain requires positive
    dry amount and the dry model's Fe-rich activity domain; pure H has no
    supported scalar or activity endpoint. Only static shapes are checked
    inside ``gex_RT`` and the shared Gibbs-excess kernel.
    """

    dry_model: MaFeSiOLiquid = field(default_factory=MaFeSiOLiquid)

    components: ClassVar[tuple[str, ...]] = ("Fe", "Si", "O", "H")
    activity_basis: ClassVar[str] = "mole_fraction"
    standard_state_convention: ClassVar[str] = "symmetric"
    reference_model_id: ClassVar[str] = "ma2001_fe_si_o_h_no_h_interaction_v1"
    reference_pressure_Pa: ClassVar[float] = MaFeSiOLiquid.reference_pressure_Pa

    def _inputs(self, T: ArrayLike, P: ArrayLike, x: ArrayLike):
        temperature = scalar_array(T, "T")
        pressure = scalar_array(P, "P")
        composition = as_inexact_array(x)
        if composition.shape != (4,):
            raise ValueError("x must have shape (4,) in Fe, Si, O, H order.")
        dtype = common_dtype(self, temperature, pressure, composition)
        return tuple(value.astype(dtype) for value in (temperature, pressure, composition))

    def gex_RT(self, T: ArrayLike, P: ArrayLike, x: ArrayLike) -> jax.Array:
        """Return molar excess energy, excluding ideal mixing and standards."""
        temperature, pressure, composition = self._inputs(T, P, x)
        dry_fraction = jnp.sum(composition[:3])
        dry_composition = composition[:3] / dry_fraction
        return dry_fraction * self.dry_model.gex_RT(temperature, pressure, dry_composition)

    def standard_state_shift_RT(self, T: ArrayLike) -> jax.Array:
        """Return the dry formal-minus-source shifts followed by zero for H.

        Add to the consumer's source standard potentials divided by RT.
        The zero H shift preserves its supplied standard; it does not assign
        zero absolute H potential or provide Okuchi partition thermochemistry.
        """
        dry_shift = self.dry_model.standard_state_shift_RT(T)
        return jnp.concatenate((dry_shift, jnp.zeros((1,), dtype=dry_shift.dtype)))

    def validate_state(self, T: ArrayLike, P: ArrayLike, x: ArrayLike) -> None:
        """Check normalized four-component fractions and the dry domain eagerly.

        No clipping, normalization, phase-stability check, or calibration
        check is performed. Normalization uses eight input-dtype epsilons.
        """
        temperature, pressure, composition = map(np.asarray, self._inputs(T, P, x))
        if not np.isrealobj(composition) or not np.all(np.isfinite(composition)):
            raise ValueError("x must be finite and real.")
        if np.any(composition < 0):
            raise ValueError("x must contain nonnegative mole fractions.")
        tolerance = 8 * jnp.finfo(as_inexact_array(x).dtype).eps
        if not np.isclose(composition.sum(), 1.0, rtol=0.0, atol=tolerance):
            raise ValueError("x must sum to one; inputs are not normalized by validate_state.")
        dry_fraction = composition[:3].sum()
        if dry_fraction <= 0:
            raise ValueError("Activity evaluation requires a positive dry amount; pure H is unsupported.")
        self.dry_model.validate_state(temperature, pressure, composition[:3] / dry_fraction)

    def tree_flatten(self):
        """Expose the dry model's differentiable interaction parameters."""
        return (self.dry_model,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Restore the immutable model through JAX transformations."""
        del aux_data
        return cls(children[0])
