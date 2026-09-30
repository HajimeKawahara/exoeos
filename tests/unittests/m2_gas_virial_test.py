"""Check source transcriptions and incomplete-mixture behavior."""

import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest


jax.config.update("jax_enable_x64", True)
_PATH = (Path(__file__).resolve().parents[2] / "examples" / "m2_material"
         / "gas_virial.py")
_SPEC = importlib.util.spec_from_file_location("m2_gas_virial", _PATH)
virial = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(virial)


@pytest.mark.parametrize("temperature,cross,water", [
    (1000, 13.91, -22.22), (1400, 14.90, -3.37),
    (1800, 15.13, 4.49), (2000, 15.13, 6.85),
])
def test_published_water_and_hydrogen_water_tables(temperature, cross, water):
    # Independent printed values, not values generated from the coefficient fit.
    pairs = virial.pair_coefficients_cm3_mol(temperature)
    assert pairs["H2-H2O"] == pytest.approx(cross, abs=0.005)
    assert pairs["H2O-H2O"] == pytest.approx(water, abs=0.005)


def test_missing_pair_does_not_become_a_zero_coefficient():
    report = virial.major_gas_diagnostic(2173.15, 27e6, [0.765, 0.170, 0.064])
    assert report["pair_coefficients_cm3_mol"]["H2-He"] is None
    assert report["state"] is None
    assert report["full_mixture_physical_error_bound"] is None
    assert report["omitted_gas_mole_fraction"] == pytest.approx(0.001)
    assert report["uncovered_pair_weight_in_original_B"] == pytest.approx(
        1 - 0.999**2)
    assert report["water_cross_pairs_extrapolated"]


def test_explicit_scenario_preserves_thermodynamic_euler_identity():
    report = virial.major_gas_diagnostic(2000, 27e6, [0.765, 0.170, 0.064],
                                         h2_he_cm3_mol=10.0)
    x = np.asarray(report["conditioned_mole_fractions"])
    state = report["state"]
    assert x @ state["log_fugacity_coefficients"] == pytest.approx(
        state["reduced_residual_gibbs"], abs=1e-14)
    assert state["compressibility_factor"] > 1
    assert not report["water_cross_pairs_extrapolated"]
    assert report["full_mixture_physical_error_bound"] is None


def test_helium_table_and_interpolation():
    assert virial.pair_coefficients_cm3_mol(2000)["He-He"] == 7.9559
    assert virial.pair_coefficients_cm3_mol(2500)["He-He"] == 7.4448
    assert 7.4448 < virial.pair_coefficients_cm3_mol(2173.15)["He-He"] < 7.9559


@pytest.mark.parametrize("temperature", [999, 3001, float("nan")])
def test_reject_unsupported_temperature(temperature):
    with pytest.raises(ValueError, match="1000 <= T <= 3000"):
        virial.pair_coefficients_cm3_mol(temperature)


@pytest.mark.parametrize("fractions", [[0, 0, 0], [0.9, 0.2, 0],
                                      [-0.1, 0.5, 0.6], [1.0, 0.0]])
def test_reject_invalid_composition(fractions):
    with pytest.raises(ValueError, match="mole fractions"):
        virial.major_gas_diagnostic(2000, 27e6, fractions)
