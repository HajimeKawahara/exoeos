"""Differentiable equations of state and excess free-energy models."""

from importlib.metadata import PackageNotFoundError, version

from exoeos.contracts import EquationOfState, GibbsExcessModel, HelmholtzEOS
from exoeos.gibbs_excess import solution_state, total_gex_RT
from exoeos.helmholtz import psir, state_trho
from exoeos.ideal import IdealEOS
from exoeos.ideal_gas import IdealGas
from exoeos.ideal_solution import IdealSolution
from exoeos.state import SolutionState, TRhoState, ThermodynamicState


try:
    __version__ = version("exoeos")
except PackageNotFoundError:
    __version__ = "0.0.0"


__all__ = [
    "EquationOfState",
    "GibbsExcessModel",
    "HelmholtzEOS",
    "IdealEOS",
    "IdealGas",
    "IdealSolution",
    "SolutionState",
    "TRhoState",
    "ThermodynamicState",
    "__version__",
    "psir",
    "solution_state",
    "state_trho",
    "total_gex_RT",
]
