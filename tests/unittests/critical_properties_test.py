"""Bundled critical-property records and their existing EOS integration."""

import jax.numpy as jnp
import pytest

from exoeos import (
    FluidCriticalProperties,
    PengRobinsonEOS,
    available_critical_properties,
    get_critical_properties,
    state_tp,
)


def test_critical_properties_are_available_in_stable_order() -> None:
    formulas = available_critical_properties()
    records = tuple(get_critical_properties(formula) for formula in formulas)

    assert formulas == ("CO", "H2O", "CO2", "H2", "CH4", "N2", "NH3", "H2S", "SO2")
    assert all(isinstance(record, FluidCriticalProperties) for record in records)
    assert tuple(record.formula for record in records) == formulas
    assert all(record.critical_temperature > 0.0 for record in records)
    assert all(record.critical_pressure > 0.0 for record in records)
    assert all(
        record.source_url.startswith("https://coolprop.org/") for record in records
    )


def test_unknown_critical_property_formula_has_actionable_error() -> None:
    with pytest.raises(KeyError, match="available formulas: CO, H2O, CO2, H2"):
        get_critical_properties("HCN")


def _eos(formulas):
    records = tuple(get_critical_properties(formula) for formula in formulas)
    return PengRobinsonEOS(
        [record.critical_temperature for record in records],
        [record.critical_pressure for record in records],
        [record.acentric_factor for record in records],
    )


def test_cns_mixture_round_trips_pressure_and_recovers_existing_support() -> None:
    eos = _eos(available_critical_properties())
    # Zero binary interactions are a numerical control, not mixture calibration.
    x = jnp.asarray([0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
    state = state_tp(eos, 500.0, 2.0e6, x)
    assert jnp.allclose(state.P, 2.0e6, rtol=1e-10)
    assert jnp.all(jnp.isfinite(state.lnphi))
    assert jnp.allclose(state.gres_RT, x @ state.lnphi, rtol=1e-10)

    cho_x = jnp.asarray([0.4, 0.4, 0.1, 0.1])
    reduced = state_tp(_eos(available_critical_properties()[:4]), 500.0, 2.0e6, cho_x)
    extended = state_tp(eos, 500.0, 2.0e6, jnp.pad(cho_x, (0, 5)))
    assert jnp.allclose(extended.rho, reduced.rho, rtol=1e-12)
    assert jnp.allclose(extended.lnphi[:4], reduced.lnphi, rtol=1e-12)
    assert jnp.allclose(extended.gres_RT, reduced.gres_RT, rtol=1e-12)


@pytest.mark.parametrize("formula", ["CH4", "N2", "NH3", "H2S", "SO2"])
def test_cns_fluid_recovers_low_pressure_ideal_limit(formula) -> None:
    state = state_tp(_eos((formula,)), 500.0, 1.0, jnp.asarray([1.0]))
    assert jnp.allclose(state.rho, 1.0 / (8.31446261815324 * 500.0), rtol=1e-6)
    assert jnp.allclose(state.Z, 1.0, rtol=1e-6)
    assert jnp.allclose(state.lnphi, 0.0, rtol=0.0, atol=1e-6)
