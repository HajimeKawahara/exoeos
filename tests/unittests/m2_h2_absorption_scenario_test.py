"""Keep measured spectral calibration separate from finite concentration G."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.m2_material.h2_absorption_scenario import absorption_scenario_report


def test_absorption_recalibration_retains_one_H2_per_oxidized_water_and_original_scenario():
    base = {"standard_offsets_rt": {"O_metal": 1.6, "H_metal": -.4}}
    result = absorption_scenario_report(base)
    assert base == {"standard_offsets_rt": {"O_metal": 1.6, "H_metal": -.4}}
    assert result["scenario"]["standard_offsets_rt"]["O_metal"] == 1.6
    assert result["scenario"]["standard_offsets_rt"]["H_metal"] == -.4
    shift = result["standard_shift_RT"]
    # Equilibrium changes through one dissolved-molecule standard, without
    # an erroneous factor of two for the two H atoms in each molecule.
    assert np.exp(-shift)*2.12 == pytest.approx(.26)
    assert len(result["concentration_receipts"]) == 10
    pc12 = result["concentration_receipts"][0]
    assert pc12["stoichiometric_H2_mass_ppm"] == pytest.approx(1.04e4*2.01588/18.01528)
    assert pc12["difference_minus_rounded_columns_wt_percent"] == pytest.approx(-.003)
    assert result["accepted_coupled_material_domain"] is None


def test_dilute_and_finite_denominators_are_distinguished_and_duplicate_offset_rejected():
    result = absorption_scenario_report({})
    rows = result["fixed_dry_host_moles_remapping"]
    assert rows[0]["constant_Henry_scenario_x"] == pytest.approx(
        rows[0]["rescaling_solute_amount_at_fixed_host_x"], rel=1e-6)
    assert rows[-1]["rescaling_solute_amount_at_fixed_host_x"] > rows[-1]["constant_Henry_scenario_x"]
    with pytest.raises(ValueError, match="double-apply"):
        absorption_scenario_report(result["scenario"])
