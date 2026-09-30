"""Check Mg reaction convention, published coefficients, and data scope."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODULE = importlib.import_module("examples.m2_material.magnesium_reference")
sys.path.pop(0)


def test_janaf_uses_absolute_enthalpy_gauge_and_entropy_not_formation_g():
    for t, s, dh in [(1800.,113.426,75.558),(2200.,124.587,97.816),(2400.,129.524,109.164)]:
        g = MODULE.mgo_crystal_standard_rt(t)*MODULE.R*t
        assert g == pytest.approx(1000.*(-601.241+dh)-t*s, abs=1e-9)
    t, step = 2000., 1e-3
    potential = lambda value: MODULE.mgo_crystal_standard_rt(value)*MODULE.R*value
    assert (potential(t+step)-potential(t-step))/(2*step) == pytest.approx(-119.249, abs=1e-5)
    with pytest.raises(ValueError, match="1700-2400"):
        MODULE.mgo_crystal_standard_rt(2500.)


def test_mg_henry_standard_closes_deoxidation_reaction_and_mass_conversion():
    t = 1873.
    result = MODULE.magnesium_henry_standard(t, oxygen_henry_standard_rt=-20.,
                                            mgo_crystal_standard_rt=-40.)
    mu_mg = result["Mg_Henry_mass_percent_standard_rt"]
    a_o = .0001
    a_mg = 10.**(-4.28-4700./t)/a_o
    reaction = mu_mg+np.log(a_mg)-20.+np.log(a_o)-(-40.)
    assert reaction == pytest.approx(0., abs=1e-14)
    # One microgram Mg in 100 g Fe provides an independent dilute x/wt% check.
    weight_percent = 1e-8
    x = (weight_percent/24.305)/(weight_percent/24.305+(100.-weight_percent)/55.845)
    molar = result["Mg_infinite_dilution_mole_fraction_standard_rt"]+np.log(x)
    assert molar == pytest.approx(mu_mg+np.log(weight_percent), abs=2e-10)
    shifted = MODULE.magnesium_henry_standard(t, oxygen_henry_standard_rt=-13.,
                                             mgo_crystal_standard_rt=-33.)
    assert shifted["Mg_Henry_mass_percent_standard_rt"] == pytest.approx(mu_mg)


def test_actual_high_oxygen_and_temperature_require_explicit_extrapolation():
    with pytest.raises(ValueError, match="opt-in"):
        MODULE.magnesium_henry_standard(2173.15, oxygen_henry_standard_rt=-20.,mgo_crystal_standard_rt=-40.)
    with pytest.raises(ValueError, match="opt-in"):
        MODULE.itoh_activity_reference(2173.15,.001,.480264)
    result = MODULE.itoh_activity_reference(2173.15,.001,.480264,allow_extrapolation=True)
    assert not result["inside_observed_coordinate_envelope"]
    assert result["coordinate_distances_T_K_Mg_wt_percent_O_wt_percent"][2] == pytest.approx(.480264-.00428)
    assert result["accepted_coupled_material_domain"] is None


def test_published_second_order_coefficients_are_replayed_without_fitting():
    t, mg, o = 1873., .0002, .00068
    result = MODULE.itoh_activity_reference(t,mg,o)
    # Direct Equations 5/29/30/33 arithmetic, independent of helper intermediates.
    expected = (958.-2590000./t)*o+(-1900000.+4220000000./t)*o**2+(214000.-516000000./t)*o*mg
    assert result["log10_f_Mg"] == pytest.approx(expected)
    report = MODULE.reference_report()
    assert report["observed_sample_count"] == 37
    assert report["refitted_parameter_count"] == 0
    assert report["summary"]["1873.0"]["sample_count"] == 28
    assert report["summary"]["2023.0"]["sample_count"] == 9
    assert report["summary"]["1873.0"]["rmse_log10_K"] == pytest.approx(.42394940030745815)
    assert report["summary"]["2023.0"]["max_absolute_residual_log10_K"] > 2.
    assert report["accepted_coupled_material_domain"] is None
