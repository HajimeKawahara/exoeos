"""Public contracts shared by equation-of-state backends."""

from typing import Protocol

from jax.typing import ArrayLike

from exoeos.state import ThermodynamicState


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
