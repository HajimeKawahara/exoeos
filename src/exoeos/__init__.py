"""Differentiable equations of state and excess free-energy models."""

from importlib.metadata import PackageNotFoundError, version

from exoeos.contracts import (
    EquationOfState,
    GibbsExcessModel,
    HelmholtzEOS,
    TPHelmholtzEOS,
)
from exoeos.critical_properties import (
    FluidCriticalProperties,
    available_critical_properties,
    get_critical_properties,
)
from exoeos.density import additive_volume_mass_density, mass_density_tp
from exoeos.gibbs_excess import solution_state, total_gex_RT
from exoeos.helmholtz import psir, state_tp, state_trho
from exoeos.ideal import IdealEOS
from exoeos.ideal_gas import IdealGas
from exoeos.ideal_solution import IdealSolution
from exoeos.peng_robinson import PengRobinsonEOS
from exoeos.second_virial import SecondVirialEOS
from exoeos.state import SolutionState, TRhoState, ThermodynamicState
from exoeos.zhang_duan import ZhangDuanEOS


try:
    __version__ = version("exoeos")
except PackageNotFoundError:
    __version__ = "0.0.0"


__all__ = [
    "EquationOfState",
    "FluidCriticalProperties",
    "GibbsExcessModel",
    "HelmholtzEOS",
    "IdealEOS",
    "IdealGas",
    "IdealSolution",
    "PengRobinsonEOS",
    "SolutionState",
    "SecondVirialEOS",
    "TPHelmholtzEOS",
    "TRhoState",
    "ThermodynamicState",
    "ZhangDuanEOS",
    "__version__",
    "additive_volume_mass_density",
    "available_critical_properties",
    "get_critical_properties",
    "mass_density_tp",
    "psir",
    "solution_state",
    "state_tp",
    "state_trho",
    "total_gex_RT",
]
