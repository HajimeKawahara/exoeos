"""Na projection bookkeeping and independent chemical-equilibrium constraints."""
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'examples'), str(ROOT/'examples/m2_material')]
import sodium_host_calibration as calibration
import sodium_reference as reference
import melts_liquid_evaluator as native


def test_raw_host_cannot_be_silently_used_as_nonnegative_native_components():
    data = json.loads(reference.DATA.read_text())
    for row in reference.replay(data):
        oxides = calibration.measured_oxide_masses(data, row)
        q = np.array([oxides.get(o, 0)/m for o, m in zip(native.OXIDES, native.OXIDE_MASSES)])
        n = np.linalg.solve(native.NU.T, q)
        assert n[native.COMPONENTS.index('al2o3')] < 0
        with pytest.raises(ValueError, match='nonnegative'):
            native.validate_request(row['temperature_K'], 1e9, n, native.COMMON_R)


@pytest.mark.parametrize('method', ['remove_potassium', 'add_aluminum_silica'])
def test_projection_preserves_original_na_fe_atoms_and_records_mass_change(method):
    data = json.loads(reference.DATA.read_text())
    for row in reference.replay(data):
        original = calibration.measured_oxide_masses(data, row)
        result = calibration.projected_host(native, original, method=method, potassium_aluminum_ratio=.9)
        q = np.asarray(result['projected_oxide_moles'])
        assert q[native.OXIDES.index('na2o')]*native.OXIDE_MASSES[native.OXIDES.index('na2o')] == pytest.approx(original['na2o'])
        assert q[native.OXIDES.index('feo')]*native.OXIDE_MASSES[native.OXIDES.index('feo')] == pytest.approx(original['feo'])
        assert sum(result['changed_oxide_masses_g'].values()) == pytest.approx(
            result['projected_total_oxide_mass_g']-result['original_total_oxide_mass_g'],abs=2e-13)
        assert min(result['component_moles']) >= 0
        assert not result['measured_host_represented']
        assert not result['empirical_uncertainty_bound']


def test_virtual_standard_is_one_na_and_covariant_under_element_gauge():
    order = native.COMPONENTS
    baseline = np.arange(len(order), dtype=float)
    gauge = dict(Na=3.2, Fe=-1.7, O=.8, Si=.1)
    vector = np.array([gauge.get(e, 0.) for e in native.ELEMENTS])
    shift = native.FORMULA_MATRIX@vector
    before = calibration.virtual_reference_rt(order, baseline)+.5*(-9.)
    after = calibration.virtual_reference_rt(order, baseline+shift)+.5*(-9.+gauge['Fe'])
    assert after-before == pytest.approx(gauge['Na'], abs=2e-14)


def test_lower_constraint_replays_independent_equilibrium_and_inequality_direction():
    properties = dict(component_order=['na2sio3','sio2','fe2sio4'],
                      mu_RT=[-20.,-30.,-40.],mu0_RT=[-19.,-26.,-35.])
    result = calibration.henry_offset_constraint(properties,alloy_x_na_upper=1e-4,alloy_x_fe=.8,gamma_fe=.7)
    mu_fe_zero=-12.
    mu_fe=mu_fe_zero+math.log(.8*.7)
    virtual_actual=calibration.virtual_reference_rt(properties['component_order'],properties['mu_RT'])
    source_mu_na=virtual_actual+.5*mu_fe
    virtual_zero=calibration.virtual_reference_rt(properties['component_order'],properties['mu0_RT'])
    standard=virtual_zero+.5*mu_fe_zero+result['delta_lower_rt']
    assert math.exp(source_mu_na-standard)/.7 == pytest.approx(1e-4)
    assert math.exp(source_mu_na-(standard+2.))/.7 < 1e-4


@pytest.mark.parametrize('slope', [0.,14340.*math.log(10.)])
def test_one_sided_envelope_keeps_unbounded_upper_and_all_constraints(slope):
    rows=[dict(run='a',temperature_K=1683.,delta_lower_rt=6.),
          dict(run='b',temperature_K=1883.,delta_lower_rt=7.)]
    fit=calibration.lower_envelope(rows,inverse_temperature_coefficient_K=slope)
    assert fit['active_runs']==['b']
    assert fit['allowed_intercept_interval'][1] is None
    assert all(x['predicted_to_reported_Na_detection_limit']<=1 for x in fit['constraints'])
    assert fit['constraints'][1]['predicted_to_reported_Na_detection_limit']==1.


def test_saved_native_and_published_replay_is_bound_to_same_proxy_inputs():
    path=ROOT/'examples/m2_material/validation/20260928_sodium_host_calibration/assessment.json'
    data=json.loads(path.read_text())
    assert len(data['rows'])==20
    for row in data['rows']:
        native_result,published=(row['models'][k] for k in ('native','published'))
        assert native_result['status']==published['status']=='available'
        assert native_result['properties']['component_moles']==published['properties']['component_moles']==row['projection']['component_moles']
        assert abs(native_result['conditional_constraint']['delta_lower_rt']-published['conditional_constraint']['delta_lower_rt'])<1e-8
    assert all(x['constraint_count']==5 for x in data['conditional_lower_envelopes'])
    assert not data['physical_domain_accepted']
