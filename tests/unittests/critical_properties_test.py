"""Contracts for the bundled critical-property records."""

import pytest

from exoeos import (
    FluidCriticalProperties,
    available_critical_properties,
    get_critical_properties,
)


def test_cho_critical_properties_are_available_in_stable_order() -> None:
    formulas = available_critical_properties()
    records = tuple(get_critical_properties(formula) for formula in formulas)

    assert formulas == ("CO", "H2O", "CO2", "H2")
    assert all(isinstance(record, FluidCriticalProperties) for record in records)
    assert all(record.critical_temperature > 0.0 for record in records)
    assert all(record.critical_pressure > 0.0 for record in records)
    assert all(
        record.source_url.startswith("https://coolprop.org/") for record in records
    )


def test_unknown_critical_property_formula_has_actionable_error() -> None:
    with pytest.raises(KeyError, match="available formulas: CO, H2O, CO2, H2"):
        get_critical_properties("CH4")
