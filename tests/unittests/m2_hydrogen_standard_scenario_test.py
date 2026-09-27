"""Check atomic/molecular stoichiometry and Henry H convention conversion."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODULE = importlib.import_module("examples.m2_material.hydrogen_standard_scenario")
sys.path.pop(0)


def supplied():
    return dict(hydrogen_standard_rt=-3.,hydrogen_lngamma_infinite_dilution=2.,
                hydrogen_gas_standard_rt=-20.)


def test_atomic_h_target_closes_half_molecular_gas_reaction_in_mass_basis():
    result = MODULE.atomic_h_standard_scenario(1873.15,101325.,**supplied())
    w = 10**(-1874./1873.15-1.601)/100.
    x_limit = w*55.845/1.00794
    h = result["target_H_standard_rt"]+2.+np.log(x_limit)
    gas = .5*(-20.+np.log(101325./100000.))
    assert h == pytest.approx(gas,abs=1e-14)
    assert result["accepted_coupled_material_domain"] is None


def test_consistent_element_gauge_and_one_atm_gas_convention_cancel():
    original = MODULE.atomic_h_standard_scenario(1873.15,101325.,**supplied())
    changed = supplied()
    changed["hydrogen_standard_rt"] += 7.
    changed["hydrogen_gas_standard_rt"] += 14.
    # Change ideal-gas standard pressure from 1 bar to 1 atm consistently.
    changed["hydrogen_gas_standard_rt"] += np.log(101325./100000.)
    actual = MODULE.atomic_h_standard_scenario(1873.15,101325.,**changed,standard_pressure_Pa=101325.)
    assert actual["standard_offsets_rt"] == pytest.approx(original["standard_offsets_rt"])


def test_extrapolation_is_explicit_and_low_pressure_raw_replicates_remain_visible():
    with pytest.raises(ValueError,match="explicit opt-in"):
        MODULE.atomic_h_standard_scenario(2173.15,267.2e5,**supplied())
    result = MODULE.atomic_h_standard_scenario(2173.15,267.2e5,**supplied(),allow_temperature_extrapolation=True)
    assert result["temperature_distance_K"] == pytest.approx(160.)
    assert result["pressure_transfer_error_bound"] is None
    assert result["FeSiOH_composition_transfer_error_bound"] is None
    data = MODULE.kato_reference_observations()
    assert data[0]["observed_H_mass_ppm"] == 25.
    assert data[1]["replicates_ppm"] == [7.26,7.18,7.16,7.51,7.61]
    assert data[1]["reported_sigma_ppm"] == .18
