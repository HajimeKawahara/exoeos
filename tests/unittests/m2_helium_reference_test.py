"""Reference units and domain restrictions for a previously omitted solute."""

import importlib.util
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "examples/m2_material/helium_reference.py"
SPEC = importlib.util.spec_from_file_location("helium_reference", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_published_morb_point_and_independent_stp_unit_conversion():
    result = MODULE.helium_henry_coefficient("MORB", 1673.)
    assert result["S_cm3_STP_per_g_bar"] == pytest.approx(59.6e-5)
    # One kg at unit fugacity: 0.596 cm3 STP divided by 22.414 litre/mol.
    assert result["He_mol_per_kg_dry_host_per_bar_fugacity"] == pytest.approx(
        .596 / 22413.969545014137)
    assert not result["is_BSE_error_bound"]
    assert not result["supports_material_admission"]


def test_reciprocal_temperature_midpoint_is_geometric_not_arithmetic():
    temperature = 2 / (1 / 1873 + 1 / 2273)
    result = MODULE.helium_henry_coefficient("olivine", temperature)
    assert result["S_cm3_STP_per_g_bar"] == pytest.approx(math.sqrt(30.5 * 64.8) * 1e-5)


@pytest.mark.parametrize("host,temperature", [("BSE", 2173.), ("enstatite", 2173.),
                                                ("MORB", 1600.), ("MORB", float("nan"))])
def test_no_silent_host_replacement_or_temperature_extrapolation(host, temperature):
    with pytest.raises(ValueError):
        MODULE.helium_henry_coefficient(host, temperature)


def test_high_pressure_same_run_metal_partition_is_not_zero():
    data = json.loads(MODULE.DATA.read_text())
    rows = data["references"]["bouhifd_2013"]["S2_same_run_Fe80Ni20"]
    for row in rows:
        ratio = row["metal_He_mass_ppm"] / row["silicate_He_mass_ppm"]
        assert ratio == pytest.approx(row["reported_D"], rel=.025)
    assert data["references"]["gessmann_wood_2002"]["Na_detection_limit_mass_ppm"] is None
    assert not data["coverage_policy"]["all_13_element_transfer_coverage_established"]
    assert all(not value["insolubility_established"] for value in data["paths"].values())
