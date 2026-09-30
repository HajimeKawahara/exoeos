"""Conditional Na redistribution between two fixed non-Na host masses."""
import math


def partition_sodium(*, active_sodium_mol, silicate_non_sodium_mass_kg,
                     metal_non_sodium_mass_kg, sodium_molar_mass_kg,
                     metal_silicate_mass_fraction_D):
    """Close only the available Na, with D defined on total phase masses.

    No activity model, source equilibrium or pressure closure is supplied.
    """
    inputs = {key: float(value) for key, value in locals().items()}
    if not all(math.isfinite(value) for value in inputs.values()):
        raise ValueError("Every fixed-box input must be finite.")
    total = inputs["active_sodium_mol"]
    d = inputs["metal_silicate_mass_fraction_D"]
    ms = inputs["silicate_non_sodium_mass_kg"]
    mm = inputs["metal_non_sodium_mass_kg"]
    mass = inputs["sodium_molar_mass_kg"]
    if total < 0 or d < 0 or min(ms, mm, mass) <= 0:
        raise ValueError("Use nonnegative Na and D, and positive non-Na host masses.")

    def mass_fractions(fraction):
        metal_amount = total * fraction
        silicate_amount = total * (1. - fraction)
        return (mass * metal_amount / (mm + mass * metal_amount),
                mass * silicate_amount / (ms + mass * silicate_amount))

    def residual(fraction):
        wm, ws = mass_fractions(fraction)
        return wm / d - ws if d >= 1. else wm - d * ws

    lower, upper = 0., 1.
    if total and d:
        for _ in range(1080):
            middle = (lower + upper) / 2.
            if middle == lower or middle == upper:
                break
            if residual(middle) < 0:
                lower = middle
            else:
                upper = middle
        fraction = min((lower, upper), key=lambda x: abs(residual(x)))
    else:
        lower = upper = fraction = 0.
    metal = total * fraction
    silicate = total - metal
    wm, ws = mass_fractions(fraction)
    return dict(inputs=inputs, sodium_mol={"silicate": silicate, "metal": metal},
                mass_fraction_metal=wm, mass_fraction_silicate=ws,
                reconstructed_D=wm / ws if ws else None,
                fraction_of_active_Na_in_metal=fraction,
                budget_residual_mol=math.fsum((silicate, metal, -total)),
                metal_fraction_bracket=[lower, upper],
                coupled_source_equilibrium_performed=False,
                pressure_closure_performed=False, is_BSE_error_bound=False)


def critical_partition_coefficient(*, target_metal_sodium_mol,
                                   active_sodium_mol,
                                   silicate_non_sodium_mass_kg,
                                   metal_non_sodium_mass_kg,
                                   sodium_molar_mass_kg):
    """D required to put a specified absolute Na amount in the metal."""
    values = (target_metal_sodium_mol, active_sodium_mol,
              silicate_non_sodium_mass_kg, metal_non_sodium_mass_kg,
              sodium_molar_mass_kg)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Every threshold input must be finite.")
    target, total, ms, mm, mass = map(float, values)
    if not 0 <= target < total or min(ms, mm, mass) <= 0:
        raise ValueError("Require 0 <= target < active Na and positive host masses.")
    metal_fraction = mass * target / (mm + mass * target)
    silicate_fraction = mass * (total - target) / (ms + mass * (total - target))
    return metal_fraction / silicate_fraction
