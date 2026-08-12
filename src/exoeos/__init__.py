"""Differentiable equations of state for planetary atmospheres and fluids."""

from importlib.metadata import PackageNotFoundError, version

from exoeos.contracts import EquationOfState, HelmholtzEOS
from exoeos.helmholtz import psir, state_trho
from exoeos.ideal import IdealEOS
from exoeos.ideal_gas import IdealGas
from exoeos.state import TRhoState, ThermodynamicState


try:
    __version__ = version("exoeos")
except PackageNotFoundError:
    __version__ = "0.0.0"


__all__ = [
    "EquationOfState",
    "HelmholtzEOS",
    "IdealEOS",
    "IdealGas",
    "TRhoState",
    "ThermodynamicState",
    "__version__",
    "psir",
    "state_trho",
]
