"""Dimensionless K, formula amounts and source/reference separation."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
r = importlib.import_module("examples.m2_material.reaction_calibration")


def test_molecular_source_pressure_sign_and_dimensionless_standard():
    result = r.source_h2_standard_audit(2173.15, 2e7)
    expected = -11.403 - .76*.02
    assert result["ln_K_dimensionless"] == pytest.approx(expected)
    assert result["delta_G0_RT"] == pytest.approx(-expected)
    assert result["experimental_mole_basis_to_MELTS_verified"] is False
    assert result["temperature_extrapolation_error_bound"] is None
    alternate = r.source_h2_standard_audit(2173.15, 2e7, standard_pressure_Pa=2e5)
    assert alternate["delta_G0_RT"] == pytest.approx(result["delta_G0_RT"]-np.log(2))


def test_illustrative_mass_standard_preserves_the_molecular_reaction_and_gauge():
    gas = -10.
    std = r.illustrative_low_pressure_h2_standard_RT(gas)
    fugacity = 3e7
    # 515 mass ppm at 1GPa; this low-P scaling remains a descriptive model.
    w_h2 = 515e-6 * fugacity/1e9
    assert std + np.log(w_h2) == pytest.approx(gas+np.log(fugacity/1e5))
    assert r.illustrative_low_pressure_h2_standard_RT(gas+np.log(2), standard_pressure_Pa=2e5) == pytest.approx(std)


def test_kato_reaction_uses_atomic_amounts_and_half_a_molecule_of_gas():
    result = r.kato_reaction_audit(1873.15, gas_h2_standard_RT=-8.)
    assert result["role"] == "independent_low_pressure_reference"
    assert abs(result["equilibrium_reaction_residual_RT"]) < 1e-14
    step = .01
    forward = r.kato_reaction_audit(1873.15+step)["ln_K_dimensionless"]
    backward = r.kato_reaction_audit(1873.15-step)["ln_K_dimensionless"]
    assert (forward-backward)/(2*step) == pytest.approx(result["d_ln_K_d_temperature_K"], rel=1e-9)
    assert result["FeSiOH_composition_transfer_error_bound"] is None
    with pytest.raises(ValueError):
        r.kato_reaction_audit(2173.15)


def test_inherited_exchange_detects_host_change_and_is_element_gauge_invariant():
    t, fe, feo, water = 2173.15, -8., -15., -20.
    target = (143589.7-69.1*t)/(r.R_J_MOL_K*t)
    h = .5*(target-feo+fe+water)
    audit = lambda fe,feo,water,h: r.audit_metal_water_exchange(
        t, fe_metal_mu0_RT=fe, feo_liquid_mu0_RT=feo,
        water_liquid_mu0_RT=water, h_metal_mu0_RT=h)
    assert abs(audit(fe,feo,water,h)["standard_exchange_residual_RT"]) < 1e-13
    # A different native host water standard breaks the inherited exchange.
    assert audit(fe,feo,water+.3,h)["standard_exchange_residual_RT"] == pytest.approx(-.3)
    # An element gauge is not a reaction recalibration.
    gauge_fe, gauge_o, gauge_h = .2, -.5, 3.
    shifted = audit(fe+gauge_fe, feo+gauge_fe+gauge_o, water+2*gauge_h+gauge_o, h+gauge_h)
    assert abs(shifted["standard_exchange_residual_RT"]) < 1e-13


def test_combined_report_does_not_promote_fit_replay_or_reference_laws_to_admission():
    report = r.calibration_report()
    assert report["water"]["fitted_observation_count"] == 74
    assert report["measured_hydrogen_references"]["kato_pure_iron"]["observed_atomic_H_mass_ppm"] == 25.
    assert report["low_pressure_illustration_h2"]["role"] == "descriptive_extrapolation_not_calibration"
    assert report["native_MELTS_standard_exchange"]["status"] == "not_evaluated"
    assert report["accepted_coupled_material_domain"] is None


@pytest.mark.parametrize("t,p,p0", [(0,1e5,1e5),(1800,-1,1e5),(1800,1e5,np.inf)])
def test_invalid_reaction_standard_inputs_fail(t,p,p0):
    with pytest.raises(ValueError):
        r.source_h2_standard_audit(t,p,standard_pressure_Pa=p0)


def formal_standard_fixture():
    return ({"sio2_liquid": -30., "fe2sio4_liquid": -52., "Fe_metal": -4.,
             "Si_metal": -7., "H2_gas": -3., "H2O_gas": -12.},
            {"liquid": "Synthetic native-endmember test", "metal": "Synthetic applied alloy standard",
             "gas": "Synthetic common gas standard at 1bar"})


def test_formal_native_reductions_keep_virtual_oxide_and_balanced_element_gauge():
    standards, conventions = formal_standard_fixture()
    result = r.audit_formal_reduction_standards(2173.15, 2e7,
        standards_rt=standards, standard_conventions=conventions)
    assert result["virtual_FeO_mu0_RT"] == -11.
    assert [item["delta_G0_RT"] for item in result["reactions"]] == pytest.approx([-4., 5., -2.])
    assert result["empirical_alignment_accepted"] is False
    assert result["virtual_FeO_is_native_or_measured_pure_standard"] is False
    assert result["reactions"][2]["uses_virtual_FeO"] is True
    # Use independently written atomic compositions for the gauge shift.
    atoms = {"sio2_liquid": {"Si":1,"O":2}, "fe2sio4_liquid": {"Fe":2,"Si":1,"O":4},
             "Fe_metal": {"Fe":1}, "Si_metal": {"Si":1}, "H2_gas": {"H":2},
             "H2O_gas": {"H":2,"O":1}}
    gauge = {"Fe":1.7,"Si":-.3,"O":8.,"H":-2.}
    shifted = {name: value+sum(count*gauge[element] for element,count in atoms[name].items())
               for name,value in standards.items()}
    alternate = r.audit_formal_reduction_standards(2173.15, 2e7,
        standards_rt=shifted, standard_conventions=conventions)
    for a,b in zip(result["reactions"], alternate["reactions"]):
        assert a["delta_G0_RT"] == pytest.approx(b["delta_G0_RT"], abs=1e-13)
        assert a["K_dimensionless"] == pytest.approx(np.exp(-a["delta_G0_RT"]))
    # Both reductions have equal gas stoichiometry on each side: a common
    # standard-pressure change cannot alter their formal reaction sums.
    for name in ("H2_gas", "H2O_gas"):
        standards[name] += np.log(2.)
    rescaled = r.audit_formal_reduction_standards(2173.15, 2e7, standards_rt=standards,
        standard_conventions=conventions, standard_pressure_Pa=2e5)
    assert [x["ln_K_dimensionless"] for x in rescaled["reactions"]] == pytest.approx([4.,-5.,2.])


def test_formal_reduction_preserves_extreme_log_K_without_false_zero_or_infinity():
    standards, conventions = formal_standard_fixture()
    for value in (-1000.,1000.):
        standards["Fe_metal"] = value
        result = r.audit_formal_reduction_standards(2173.15, 1e5,
            standards_rt=standards, standard_conventions=conventions)
        assert result["reactions"][0]["K_dimensionless"] is None
        assert np.isfinite(result["reactions"][0]["ln_K_dimensionless"])


@pytest.mark.parametrize("bad", [None, np.nan, True, [1.], "1"])
def test_formal_reductions_reject_missing_or_ambiguous_standard_values(bad):
    standards, conventions = formal_standard_fixture()
    standards["Fe_metal"] = bad
    with pytest.raises(ValueError):
        r.audit_formal_reduction_standards(2173.15, 1e5,
            standards_rt=standards, standard_conventions=conventions)


def test_formal_reductions_require_each_phase_convention_and_each_standard():
    standards, conventions = formal_standard_fixture()
    conventions["metal"] = ""
    with pytest.raises(ValueError):
        r.audit_formal_reduction_standards(2173.15, 1e5,
            standards_rt=standards, standard_conventions=conventions)
    _, conventions = formal_standard_fixture()
    del standards["H2_gas"]
    with pytest.raises(ValueError):
        r.audit_formal_reduction_standards(2173.15, 1e5,
            standards_rt=standards, standard_conventions=conventions)
