"""Published amount bases and measured-domain rejection for M2 references."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from exoeos import mass_fraction_solute_state


_PATH = Path(__file__).resolve().parents[2] / "examples/m2_material/hydrogen_reference.py"
_SPEC = importlib.util.spec_from_file_location("m2_hydrogen_reference", _PATH)
_REFERENCE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REFERENCE)
ATM_PA = _REFERENCE.ATM_PA
chaudhari_buffered_h2_mass_ppm = _REFERENCE.chaudhari_buffered_h2_mass_ppm
kato1970_hydrogen_mass_ppm = _REFERENCE.kato1970_hydrogen_mass_ppm
kato_atomic_h_standard_RT = _REFERENCE.kato_atomic_h_standard_RT
reference_report = _REFERENCE.reference_report


def test_buffered_regressions_retain_observed_scatter():
    record = reference_report()
    first = next(row for row in record["chaudhari_regression_residuals"] if row["run_id"] == "AMPC38")
    assert first["predicted_H2_mass_ppm"] == 103.
    assert first["H2_mass_ppm"] == 75.
    assert first["residual_over_observation_sigma"] == 7.
    assert record["accepted_coupled_material_domain"] is None
    assert record["author_excluded_run_ids"] == ["AMPC5", "AMPC6"]
    assert len(record["chaudhari_regression_residuals"]) == 19
    result = chaudhari_buffered_h2_mass_ppm("andesite", 1673.15, 2e9, buffer="Fe-FeO-H2O")
    assert result["molecular_h2_mass_ppm"] == 724.
    assert result["slope_uncertainty_ppm"] == 70.
    assert chaudhari_buffered_h2_mass_ppm("haplogranite",1473.15,1e9,buffer="Fe-FeO-H2O")["molecular_h2_mass_ppm"] == 500.


def test_atomic_h_reference_reproduces_one_atmosphere_and_silicon_effect():
    assert kato1970_hydrogen_mass_ppm(1873.15) == pytest.approx(25.0349206343)
    ratio = kato1970_hydrogen_mass_ppm(1873.15, silicon_mass_percent=2.) / kato1970_hydrogen_mass_ppm(1873.15)
    assert ratio == pytest.approx(.8590135215)
    receipt = reference_report()["kato_low_partial_pressure_receipt"]
    assert receipt["predicted_atomic_H_mass_ppm"] == pytest.approx(7.5483125818)
    assert receipt["residual_over_reported_sigma"] > 1.


def test_atom_balanced_standard_and_mass_scalar_reproduce_dilute_pure_iron():
    gas_mu0 = -8.
    standard = kato_atomic_h_standard_RT(1873.15, gas_mu0)
    gas_mu = gas_mu0 + np.log(ATM_PA / 1e5)
    weights = np.array([.055845])
    h_weight = .00100794  # Atomic H; no molecular-H2 amount or factor-of-two ambiguity.
    energy = lambda n: float(mass_fraction_solute_state(
        0., np.zeros(1), np.ones(1), weights, n, h_weight, standard).gibbs_RT) - .5 * gas_mu * n
    minimum = minimize_scalar(energy, method="bounded", bounds=(1e-5, .01), options={"xatol": 1e-12})
    ppm = 1e6 * minimum.x * h_weight / (weights[0] + minimum.x * h_weight)
    assert minimum.success
    assert ppm == pytest.approx(25.0349206343, rel=3e-7)
    # Changing p0 must change the gas mu0 consistently and preserve this liquid standard.
    other_p0 = 2e5
    other_gas_mu0 = gas_mu0 + np.log(other_p0 / 1e5)
    assert kato_atomic_h_standard_RT(1873.15, other_gas_mu0, standard_pressure_Pa=other_p0) == pytest.approx(standard)


@pytest.mark.parametrize("host,t,p,buffer", [("BSE",1673.15,1e9,"Fe-FeO-H2O"),
    ("basalt",2173.15,1e5,"Fe-FeO-H2O"), ("basalt",1673.15,3e9,"Fe-FeO-H2O"),
    ("andesite",1673.15,1e9,"IW"), ("andesite",np.nan,1e9,"Fe-FeO-H2O"),
    ("haplogranite",1500.,1e9,"Fe-FeO-H2O"), ("haplogranite",1473.15,2.5e9,"Fe-FeO-H2O")])
def test_unsupported_silicate_host_and_buffer_are_rejected(host,t,p,buffer):
    with pytest.raises(ValueError):
        chaudhari_buffered_h2_mass_ppm(host,t,p,buffer=buffer)


@pytest.mark.parametrize("t,p,si", [(2173.15,ATM_PA,0.), (1873.15,1e5,0.),
    (1973.15,ATM_PA,1.), (1873.15,ATM_PA,2.5), (1873.15,np.inf,0.)])
def test_unsupported_iron_temperature_pressure_or_composition_is_rejected(t,p,si):
    with pytest.raises(ValueError):
        kato1970_hydrogen_mass_ppm(t,hydrogen_partial_pressure_Pa=p,silicon_mass_percent=si)
