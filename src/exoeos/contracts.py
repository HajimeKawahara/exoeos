"""Public contracts shared by thermodynamic model backends."""

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


class GibbsExcessModel(Protocol):
    """Symmetric mole-fraction excess Gibbs-energy model.

    A valid model has zero excess Gibbs energy at each pure-component or
    pure-endmember composition.
    """

    def gex_RT(
        self,
        T: ArrayLike,
        P: ArrayLike,
        x: ArrayLike,
    ) -> jax.Array:
        """Return the reduced molar excess Gibbs energy.

        Args:
            T: Temperature in K.
            P: Absolute pressure in Pa.
            x: Mole fractions with shape ``(K,)``.

        Returns:
            The dimensionless value ``g^E / (R T)`` under the symmetric
            mole-fraction convention, referenced to pure components or
            specified pure endmembers at the same ``T`` and ``P``.
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
