"""Public contracts shared by equation-of-state backends."""

from typing import Protocol

import jax
from jax.typing import ArrayLike

from exoeos.state import ThermodynamicState


class HelmholtzEOS(Protocol):
    """Residual Helmholtz free-energy model."""

    def alphar(
        self,
        T: ArrayLike,
        rho: ArrayLike,
        x: ArrayLike,
    ) -> jax.Array:
        """Return the reduced residual molar Helmholtz energy.

        Args:
            T: Temperature in K.
            rho: Total molar density in mol m^-3.
            x: Mole fractions with shape ``(K,)``.

        Returns:
            The dimensionless value ``A^r / (n R T)``.
        """

        ...


class EquationOfState(Protocol):
    """Structural interface implemented by an equation-of-state backend."""

    def state(
        self,
        T: ArrayLike,
        P: ArrayLike,
        x: ArrayLike,
    ) -> ThermodynamicState:
        """Evaluate a thermodynamic state from temperature, pressure, and composition."""

        ...
