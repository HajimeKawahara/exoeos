"""Small, curated critical-property table for cubic equations of state."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FluidCriticalProperties:
    """Critical properties required by the Peng-Robinson equation of state.

    Temperatures are in K and pressures are in Pa. The source URL identifies
    the CoolProp fluid-information page from which the values were transcribed.
    """

    formula: str
    critical_temperature: float
    critical_pressure: float
    acentric_factor: float
    source_url: str


_COOLPROP_ROOT = "https://coolprop.org/fluid_properties/fluids"

_CRITICAL_PROPERTIES = {
    "CO": FluidCriticalProperties(
        formula="CO",
        critical_temperature=132.85989463386605,
        critical_pressure=3_498_194.6661988837,
        acentric_factor=0.0497,
        source_url=f"{_COOLPROP_ROOT}/CarbonMonoxide.html",
    ),
    "H2O": FluidCriticalProperties(
        formula="H2O",
        critical_temperature=647.0959999999873,
        critical_pressure=22_063_999.999997754,
        acentric_factor=0.3442920843,
        source_url=f"{_COOLPROP_ROOT}/Water.html",
    ),
    "CO2": FluidCriticalProperties(
        formula="CO2",
        critical_temperature=304.1282000029807,
        critical_pressure=7_377_298.373446752,
        acentric_factor=0.22394,
        source_url=f"{_COOLPROP_ROOT}/CarbonDioxide.html",
    ),
    "H2": FluidCriticalProperties(
        formula="H2",
        critical_temperature=33.1443326883113,
        critical_pressure=1_296_357.6060553084,
        acentric_factor=-0.219,
        source_url=f"{_COOLPROP_ROOT}/Hydrogen.html",
    ),
}


def available_critical_properties() -> tuple[str, ...]:
    """Return the molecular formulas available in the bundled table."""

    return tuple(_CRITICAL_PROPERTIES)


def get_critical_properties(formula: str) -> FluidCriticalProperties:
    """Return the critical-property record for a molecular formula.

    Args:
        formula: Case-sensitive molecular formula, such as ``"CO2"``.

    Raises:
        KeyError: If the formula is not present in the bundled table.
    """

    try:
        return _CRITICAL_PROPERTIES[formula]
    except KeyError as exc:
        available = ", ".join(available_critical_properties())
        raise KeyError(
            f"Unknown molecular formula {formula!r}; available formulas: {available}."
        ) from exc
