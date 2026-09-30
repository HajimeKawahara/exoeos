"""Conditional finite-He partitioning in fixed gas/silicate/metal boxes.

This example tests a declared partition continuation, not a new source Gibbs
model, pressure closure, or an empirical bound for low-pressure BSE.
"""

import math


def redistribute_helium(*, total_helium_mol, other_gas_mol, pressure_bar,
                        dry_silicate_mass_kg, silicate_nonhelium_mass_kg,
                        metal_nonhelium_mass_kg, helium_molar_mass_kg,
                        henry_mol_per_kg_dry_bar, metal_silicate_mass_fraction_D):
    """Close He only, keeping T/P, other gases and all non-He hosts fixed.

    Henry's law uses dry silicate mass and ideal He partial pressure with
    the changing *complete* gas denominator. D uses He-inclusive total
    phase mass fractions, including silicate water/H2 in the host mass.
    Only 0 <= D < 1 is supported by this diagnostic.
    """
    supplied = dict(locals())
    p = {key: float(value) for key, value in supplied.items()}
    if any(not math.isfinite(value) for value in p.values()):
        raise ValueError("Every box input must be finite")
    positive = ("other_gas_mol", "pressure_bar", "silicate_nonhelium_mass_kg",
                "helium_molar_mass_kg")
    nonnegative = ("total_helium_mol", "dry_silicate_mass_kg",
                   "metal_nonhelium_mass_kg", "henry_mol_per_kg_dry_bar")
    if (any(p[key] <= 0 for key in positive) or any(p[key] < 0 for key in nonnegative)
            or p["dry_silicate_mass_kg"] > p["silicate_nonhelium_mass_kg"]
            or not 0 <= p["metal_silicate_mass_fraction_D"] < 1):
        raise ValueError("Require positive gas/phase bases, 0 <= dry <= full host, and 0 <= D < 1")
    total = p["total_helium_mol"]
    d = p["metal_silicate_mass_fraction_D"]
    silicate_mass = p["silicate_nonhelium_mass_kg"]
    he_mass = p["helium_molar_mass_kg"]
    capacity = p["dry_silicate_mass_kg"] * p["henry_mol_per_kg_dry_bar"] * p["pressure_bar"]
    # Normalize the scalar inventory equation; all three fractions increase
    # monotonically with the gas fraction, so [0, 1] brackets its unique root.
    def fractions(gas):
        silicate = capacity * gas / (p["other_gas_mol"] + total * gas)
        metal = ((d * p["metal_nonhelium_mass_kg"] / silicate_mass) * silicate
                 / (1 + (1 - d) * he_mass * total / silicate_mass * silicate))
        return gas, silicate, metal

    lower, upper = 0., 1.
    iterations = 0
    if total:
        if not all(math.isfinite(value) for value in fractions(1.)):
            raise ValueError("The supplied box scales overflow the diagnostic")
        for iterations in range(1, 1080):
            middle = (lower + upper) / 2
            if middle == lower or middle == upper:
                break
            if math.fsum(fractions(middle)) < 1:
                lower = middle
            else:
                upper = middle
        selected = min((lower, upper), key=lambda value: abs(math.fsum(fractions(value)) - 1))
        gas, silicate, metal = [value * total for value in fractions(selected)]
    else:
        gas = silicate = metal = 0.
        lower = upper = 0.
    pressure_he = p["pressure_bar"] * gas / (p["other_gas_mol"] + gas)
    ws = he_mass * silicate / (silicate_mass + he_mass * silicate)
    wm = (he_mass * metal / (p["metal_nonhelium_mass_kg"] + he_mass * metal)
          if p["metal_nonhelium_mass_kg"] else None)
    residual = math.fsum((gas, silicate, metal, -total))
    return {
        "inputs": p,
        "helium_mol": {"gas": gas, "silicate": silicate, "metal": metal},
        "fraction_of_total_helium": {name: value / total if total else 0.
                                     for name, value in zip(("gas", "silicate", "metal"), (gas, silicate, metal))},
        "He_partial_pressure_bar": pressure_he,
        "He_mass_fraction_total_silicate": ws,
        "He_mass_fraction_total_metal": wm,
        "He_mass_fraction_ratio": wm / ws if wm is not None and ws else None,
        "He_mass_per_kg_dry_silicate": he_mass * silicate / p["dry_silicate_mass_kg"]
                                        if p["dry_silicate_mass_kg"] else None,
        "helium_budget_residual_mol": residual,
        "helium_budget_relative_residual": residual / total if total else 0.,
        "numerics": {"method": "monotone binary64 bisection to adjacent gas-fraction endpoints",
                     "gas_fraction_bracket": [lower, upper], "iterations": iterations},
        "coupled_source_equilibrium_performed": False,
        "pressure_closure_performed": False,
        "supports_material_admission": False,
        "is_BSE_error_bound": False,
    }
