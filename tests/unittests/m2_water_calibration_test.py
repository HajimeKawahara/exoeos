"""Source concentration basis, experimental scatter, and integrable water G."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq


_PATH = Path(__file__).resolve().parents[2] / "examples/m2_material/water_calibration.py"
_SPEC = importlib.util.spec_from_file_location("water_calibration", _PATH)
w = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(w)


def conditional_water_state(*args, **kwargs):
    return w.water_dissolution_state(*args, response_interpretation="literal_OH_group_mass", **kwargs)


def test_published_rows_are_replayed_without_silently_renormalizing_oxides():
    result = w.water_calibration_report()
    assert result["fitted_observation_count"] == 74
    assert result["blank_count"] == 7
    assert len(result["observations"]) == 81
    first = result["observations"][0]
    assert first["observed_OH_mass_ppm"] == 226.4545345
    # Independent transcription of Table4 applied to author CSV row0.
    expected = 719 * np.exp((-1511.1*.408216929 + 886.5*.484637301
                            - 1015.2*.065795495 + 890.6*.038587228) / 1673.15) * np.sqrt(.156)
    assert first["predicted_OH_mass_ppm"] == pytest.approx(expected, rel=1e-12)
    assert result["max_absolute_residual_over_observed_sigma"] > 3
    assert result["accepted_coupled_material_domain"] is None
    assert len([v for v in result["observations"] if v["study"] == "Sossi2023" and v["used_in_author_fit"]]) == 10


def test_oh_mass_basis_water_equivalent_and_uncertain_covariance():
    row = w.load_observations()[0]
    x = np.array([float(row[name + "_MF"]) for name in w.OXIDES[:5]])
    result = w.water_reference(1673.15, .156e5, x)
    assert result["matched_observations"] == ["F1-A1"]
    # OH includes host oxygen: two OH groups per added H2O component.
    assert result["H2O_equivalent_mass_fraction_if_literal_OH"] / result["author_OH_response"] == pytest.approx(18.0153/(2*17.008))
    bound = result["uncertainty"]["sigma_ln_capacity_upper_from_parameter_marginals"]
    jac = x / 1673.15
    sigma = np.asarray(w.B_SIGMA_K)
    # Fully correlated coefficients attain the reported worst-case SD.
    assert bound**2 == pytest.approx(jac @ np.outer(sigma, sigma) @ jac)
    assert result["uncertainty"]["parameter_covariance"] is None
    assert result["uncertainty"]["extrapolation_error_bound"] is None


def test_sossi_water_regression_retains_both_ir_branches_and_h2_as_oh():
    first = w.sossi2023_water_response(.027e5, .012e5, absorption_basis="basalt_epsilon_6.3")
    other = w.sossi2023_water_response(.027e5, .012e5, absorption_basis="peridotite_epsilon_5.1")
    assert first == pytest.approx((524*np.sqrt(.027)+183*np.sqrt(.012))*1e-6)
    assert other == pytest.approx((647*np.sqrt(.027)+225*np.sqrt(.012))*1e-6)
    assert other > first
    result = w.sossi_regression_replay()
    assert len(result["observations"]) == 24
    assert result["absorption_branches_are_independent_observations"] is False
    row = next(v for v in result["observations"] if v["sample"] == "Per-fO2(10)" and v["absorption_basis"] == "basalt_epsilon_6.3")
    assert row["observed_water_equivalent_mass_ppm"] == 103.3
    assert row["predicted_water_equivalent_mass_ppm"] == pytest.approx(first*1e6)


def test_no_joint_temperature_pressure_composition_domain_is_invented():
    row = w.load_observations()[0]
    x = np.array([float(row[name + "_MF"]) for name in w.OXIDES[:5]])
    with pytest.raises(ValueError, match="extrapolation"):
        w.water_reference(2173.15, .156e5, x, total_pressure_Pa=3e7)
    result = w.water_reference(2173.15, .156e5, x, total_pressure_Pa=3e7, allow_extrapolation=True)
    assert result["evaluation_scope"] == "explicit_extrapolation"
    assert result["accepted_coupled_material_domain"] is None


def test_temperature_fugacity_composition_derivatives_agree_with_finite_differences():
    x = np.array([.2, .5, .15, .1, .04])
    t, f = 1800., 8e3
    result = w.water_reference(t, f, x, allow_extrapolation=True)
    params = np.r_[t, f, x]
    scalar = lambda q: w.thompson_oh_response(q[0], q[1], q[2:])
    actual = np.asarray(jax.grad(scalar)(jnp.asarray(params)))
    expected = np.r_[result["d_author_OH_response_d_T_K"], result["d_author_OH_response_d_fugacity_Pa"],
                     result["d_author_OH_response_d_oxide_fraction"]]
    np.testing.assert_allclose(actual, expected, rtol=2e-12)
    for i in range(len(params)):
        step = np.zeros_like(params)
        step[i] = 1e-5 * max(abs(params[i]), .01)
        finite = (float(scalar(params+step)) - float(scalar(params-step))) / (2*step[i])
        assert finite == pytest.approx(actual[i], rel=2e-7)


def test_dissolution_scalar_includes_host_composition_derivatives_and_euler_identity():
    host = jnp.array([.2, .5, .15, .1, .04, .01])
    amounts = jnp.r_[host, .0005]
    state = conditional_water_state(1800., host, amounts[-1], -20.)
    scalar = lambda n: conditional_water_state(1800., n[:-1], n[-1], -20.).gibbs_RT
    mu = jnp.r_[state.host_mu_RT, state.water_mu_RT]
    np.testing.assert_allclose(jax.grad(scalar)(amounts), mu, atol=2e-12)
    assert float(amounts @ mu) == pytest.approx(float(state.gibbs_RT), rel=2e-12)
    scaled = conditional_water_state(1800., 7*host, 7*amounts[-1], -20.)
    assert float(scaled.gibbs_RT) == pytest.approx(7*float(state.gibbs_RT), rel=2e-12)
    np.testing.assert_allclose(scaled.host_mu_RT, state.host_mu_RT, atol=2e-12)
    assert float(scaled.water_mu_RT) == pytest.approx(float(state.water_mu_RT))
    hessian = np.asarray(jax.jacfwd(jax.grad(scalar))(amounts))
    np.testing.assert_allclose(hessian, hessian.T, atol=2e-11)


def test_standalone_gibbs_equilibrium_recovers_mass_law_and_standard_pressure_gauge():
    host = np.array([.2, .5, .15, .1, .04, .01])
    t, f, gas_standard = 1800., 1.7e4, -20.
    gas_mu = gas_standard + np.log(f/1e5)
    evaluate = lambda h: conditional_water_state(t, host, h, gas_standard)
    n_water = brentq(lambda h: float(evaluate(h).water_mu_RT)-gas_mu, 1e-9, .01, xtol=1e-15)
    state = evaluate(n_water)
    assert float(state.OH_mass_fraction) == pytest.approx(float(w.thompson_oh_response(t, f, host[:5])), rel=1e-9)
    total_mass = np.dot(host, w.OXIDE_MASSES_KG_MOL) + n_water*w.WATER_KG_MOL
    # Primitive H and O atom counts: host oxygen is not added a second time.
    h_atoms = 2*n_water
    assert float(state.OH_mass_fraction)*total_mass/w.OH_KG_MOL == pytest.approx(h_atoms)
    other = conditional_water_state(t, host, n_water, gas_standard+np.log(2), standard_pressure_Pa=2e5)
    assert float(other.gibbs_RT) == pytest.approx(float(state.gibbs_RT))
    assert float(other.water_mu_RT) == pytest.approx(float(state.water_mu_RT))


def test_exact_zero_water_is_not_replaced_by_a_trace():
    host = np.array([.2, .5, .15, .1, .04, .01])
    state = conditional_water_state(1800., host, 0., -20.)
    assert float(state.gibbs_RT) == 0
    assert float(state.OH_mass_fraction) == 0
    assert np.isneginf(state.water_mu_RT)
    np.testing.assert_array_equal(state.host_mu_RT, np.zeros(6))
    assert float(w.thompson_oh_response(1800., 0., host[:5])) == 0


@pytest.mark.parametrize("t,f,x", [(0.,1.,[.2,.5,.1,.1,.1]), (1800.,-1.,[.2,.5,.1,.1,.1]),
    (1800.,1.,[.3,.5,.1,.1,.1]), (1800.,1.,[-.1,.5,.1,.1,.1]), (np.nan,1.,[.2,.5,.1,.1,.1])])
def test_invalid_state_has_no_finite_physical_value(t, f, x):
    assert np.isnan(w.thompson_oh_response(t, f, x))
    with pytest.raises(ValueError):
        w.water_reference(t, f, x, allow_extrapolation=True)


def test_source_file_changes_fail_before_replay(tmp_path, monkeypatch):
    import shutil
    shutil.copytree(w.DATA_DIR, tmp_path/'data')
    monkeypatch.setattr(w, "DATA_DIR", tmp_path/'data')
    monkeypatch.setattr(w, "METADATA_PATH", tmp_path/'data/provenance.json')
    path = tmp_path/'data/CombinedDataset_withS23N17.csv'
    path.write_bytes(path.read_bytes().replace(b'226.4545345', b'226.4545346'))
    with pytest.raises(ValueError, match="hash"):
        w.water_calibration_report()


def test_pooled_sossi_conversion_is_not_silently_treated_as_atom_validated():
    audit = w.sossi_basis_audit()
    assert audit["status"] == "physical_atomic_normalization_unresolved"
    assert len(audit["observations"]) == 12
    for row in audit["observations"]:
        assert row["combined_over_literal_two_group_prediction"] == pytest.approx(.5, rel=2e-4)
    host = np.array([.2,.5,.15,.1,.04,.01])
    with pytest.raises(TypeError):
        w.water_dissolution_state(1800.,host,.001,-20.)
    with pytest.raises(ValueError, match="interpretation"):
        w.water_dissolution_state(1800.,host,.001,-20.,response_interpretation="calibrated_atomic_basis")
    assert np.isnan(conditional_water_state(1800.,host,3.,-20.).gibbs_RT)
