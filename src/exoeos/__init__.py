"""Differentiable equations of state for planetary atmospheres and fluids."""

from importlib.metadata import PackageNotFoundError, version

from exoeos.contracts import EquationOfState
from exoeos.ideal_gas import IdealGas
from exoeos.state import ThermodynamicState


try:
    __version__ = version("exoeos")
except PackageNotFoundError:
    __version__ = "0.0.0"


__all__ = [
    "EquationOfState",
    "IdealGas",
    "ThermodynamicState",
    "__version__",
]
