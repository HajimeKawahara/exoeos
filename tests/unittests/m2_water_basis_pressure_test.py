"""Independent water mass basis, closed G derivatives and pressure holdout."""

import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.m2_material import water_basis_pressure as w
from examples.m2_material.water_calibration import (OH_KG_MOL, WATER_KG_MOL,
                                                    OXIDE_MASSES_KG_MOL,
                                                    water_dissolution_state)


def test_absorptivity_basis_reconstructs_original_water_without_second_factor_two():
    replay = w.normalization_replay()
    assert len(replay["observations"]) == 12
    assert replay["max_abs_rounding_difference_mass_ppm"] < 5e-8
    row = next(v for v in replay["observations"] if v["sample"] == "Per-fO2(10)")
    assert row["original_H2O_mass_ppm"] == pytest.approx((103.3 + 127.5) / 2)


def test_reconstructed_scalar_matches_fugacity_and_all_composition_derivatives():
    host = jnp.array([.02, .40, .49, .025, .06, .005])
    temperature, fugacity, gas_standard = 2173.15, 20e5, -3.2
    mass_fraction = float(w.water_equivalent_response(temperature, fugacity, host[:5]))
    dry_mass = float(host @ jnp.asarray(OXIDE_MASSES_KG_MOL))
    water = mass_fraction * dry_mass / (WATER_KG_MOL * (1-mass_fraction))
    n = jnp.r_[host, water]
    state = w.water_equivalent_dissolution_state(temperature, host, water, gas_standard)
    assert float(state.water_mu_RT) == pytest.approx(gas_standard + np.log(fugacity/1e5), abs=1e-12)
    energy = lambda n: w.water_equivalent_dissolution_state(temperature, n[:-1], n[-1], gas_standard).gibbs_RT
    gradient = jax.grad(energy)(n)
    assert np.asarray(gradient) == pytest.approx(np.r_[state.host_mu_RT, state.water_mu_RT], abs=1e-12)
    assert float(n @ gradient) == pytest.approx(float(state.gibbs_RT), abs=1e-12)
    legacy = lambda n: water_dissolution_state(temperature, n[:-1], n[-1], gas_standard,
                                              response_interpretation="literal_OH_group_mass").gibbs_RT
    assert float(energy(n)-legacy(n)) == pytest.approx(-2*water*np.log(2), abs=1e-12)
    assert np.asarray(jax.hessian(energy)(n)) == pytest.approx(np.asarray(jax.hessian(legacy)(n)), abs=1e-11)
    assert float(state.author_response) == pytest.approx(mass_fraction * OH_KG_MOL/WATER_KG_MOL)


def test_pressure_holdout_preserves_missing_zero_and_parallel_measurements():
    result = w.pressure_replay()
    rows = {v["sample"]: v for v in result["observations"]}
    assert len(rows) == 14
    assert rows["M13"]["NIR_total_H2O_wt_percent"] is None
    assert rows["M9"]["NIR_molecular_H2O_wt_percent"] == 0
    # Table2's OH column is already expressed in H2O-equivalent weight.
    assert rows["M51"]["NIR_molecular_H2O_wt_percent"] + rows["M51"]["NIR_OH_H2O_equivalent_wt_percent"] == pytest.approx(2.34)
    # Fourteen KFT and twelve NIR analyses concern fourteen capsules, not26.
    assert result["statistics"]["KFT_H2O_wt_percent"]["count"] == 14
    assert result["statistics"]["NIR_total_H2O_wt_percent"]["count"] == 12
    # Published coefficients, independent Table1 composition and500bar.
    assert rows["M13"]["predicted_H2O_equivalent_wt_percent"] == pytest.approx(1.8418109307155501)
    assert result["accepted_coupled_material_domain"] is None
    assert result["original_paper_gas_EOS_reproduced"] is False
    with pytest.raises(ValueError):
        w.pressure_replay(gas_model="original_paper")
