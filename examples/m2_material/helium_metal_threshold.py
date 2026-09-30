"""Inverse metal-partition threshold for the explicitly frozen He boxes."""

import math

from helium_metal_box import redistribute_helium


def metal_partition_threshold(box_inputs, target_metal_helium_fraction):
    """Find D for a chosen fraction of total He in metal; infer no bound on D.

    Fix n_m = eta*N first. The remaining He obeys the existing gas/silicate
    D=0 balance. Taking the ratio of the two finite phase mass fractions
    gives the required D, including when it exceeds the forward diagnostic's
    D<1 range. At this single threshold wm<1 remains guaranteed.
    """
    if "metal_silicate_mass_fraction_D" in box_inputs:
        raise ValueError("D is the unknown threshold, not a supplied input")
    target = float(target_metal_helium_fraction)
    if not math.isfinite(target) or not 0 < target < 1:
        raise ValueError("Require a finite target fraction strictly between zero and one")
    original = redistribute_helium(**box_inputs, metal_silicate_mass_fraction_D=0.)["inputs"]
    for key in ("total_helium_mol", "metal_nonhelium_mass_kg", "dry_silicate_mass_kg",
                "henry_mol_per_kg_dry_bar"):
        if original[key] <= 0:
            raise ValueError("A positive metal threshold needs He, metal and positive silicate capacity")
    total = original["total_helium_mol"]
    metal = target * total
    remaining = {**original, "total_helium_mol": total - metal}
    state = redistribute_helium(**remaining)
    gas, silicate = (state["helium_mol"][name] for name in ("gas", "silicate"))
    helium_mass = original["helium_molar_mass_kg"]
    wm = helium_mass * metal / (original["metal_nonhelium_mass_kg"] + helium_mass * metal)
    ws = state["He_mass_fraction_total_silicate"]
    if not 0 < ws < 1 or not 0 < wm < 1:
        raise ValueError("The threshold requires finite positive phase mass fractions below one")
    critical = wm / ws
    if not math.isfinite(critical):
        raise ValueError("The required threshold overflows this diagnostic")
    return {
        "box_inputs": {key: value for key, value in original.items() if key != "metal_silicate_mass_fraction_D"},
        "target_metal_helium_fraction": target,
        "critical_mass_fraction_partition_D": critical,
        "helium_mol": {"gas": gas, "silicate": silicate, "metal": metal},
        "He_partial_pressure_bar": state["He_partial_pressure_bar"],
        "He_mass_fraction_total_silicate": ws,
        "He_mass_fraction_total_metal": wm,
        "helium_budget_relative_residual": math.fsum((gas, silicate, metal, -total)) / total,
        "remaining_gas_silicate_balance": state,
        "scope": "Fixed-box break-even D only; no empirical bound on the unknown low-pressure D.",
        "critical_D_is_adopted_material_parameter": False,
        "supports_material_admission": False,
        "is_BSE_error_bound": False,
        "pressure_closure_performed": False,
    }
