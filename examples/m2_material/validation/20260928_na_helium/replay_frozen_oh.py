"""Replay a frozen-source capacity illustration; never run an equilibrium."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from helium_reference import helium_henry_coefficient


EXPECTED_SOURCE_SHA = "f7b848f0c89740ac77010b56eccdaa6c5283c17216d6aeac24d2c99ea54b9235"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SOURCE_SHA:
        raise ValueError("This replay is bound to the archived OH closure bytes")
    document = json.loads(raw)
    root = document["runs"][0]["roots"][0]
    elements = root["elements"]
    masses = dict(zip(elements, root["atomic_masses_kg_mol"]))
    inventory = dict(zip(elements, document["inventory"]["total_element_amounts_mol"]))
    names = root["source_component_order"]
    amounts = dict(zip(names, root["source_component_amounts_mol"]))
    formulas = root["source_formula_matrix"]
    component_mass = {
        name: sum(formulas[e][i] * masses[element] for e, element in enumerate(elements))
        for i, name in enumerate(names)
    }
    phase_mass = {
        phase: sum(amounts[name] * component_mass[name] for name in components)
        for phase, components in root["source_phases"].items()
    }
    dry_mass = sum(amounts[name] * component_mass[name]
                   for name in root["source_phases"]["silicate"]
                   if name not in ("h2o_melts", "H2_dissolved"))
    gas = root["source_phases"]["gas"]
    gas_denominator = sum(amounts[name] for name in gas)
    pressure = root["pressure_base_pa"] / 1e5
    p_he = pressure * amounts["He_gas"] / gas_denominator
    rows = []
    for host in ("olivine", "MORB", "rhyolite"):
        coefficient = helium_henry_coefficient(host, root["temperature_base_k"])
        amount = dry_mass * coefficient["He_mol_per_kg_dry_host_per_bar_fugacity"] * p_he
        rows.append({
            **coefficient,
            "He_mol_frozen": amount,
            "fraction_of_global_He": amount / inventory["He"],
            "He_mass_ppm_per_dry_host": 1e6 * amount * masses["He"] / dry_mass,
        })
    result = {
        "kind": "frozen_source_reference_illustration_not_a_coupled_bound",
        "source_sha256": EXPECTED_SOURCE_SHA,
        "source_archive": "ExoInventory PR 35, final_common_runtime_oh/closure.json",
        "temperature_K": root["temperature_base_k"], "pressure_bar": pressure,
        "gas_species_count": len(gas), "complete_gas_amount_mol": gas_denominator,
        "He_partial_pressure_bar": p_he, "ideal_pHe_used_as_fugacity": True,
        "dry_silicate_mass_kg": dry_mass,
        "reconstructed_source_phase_mass_kg": phase_mass,
        "global_He_inventory_mol": inventory["He"],
        "global_He_inventory_mass_kg": inventory["He"] * masses["He"],
        "rows": rows,
        "Na": {"inventory_mol": inventory["Na"],
               "inventory_mass_kg": inventory["Na"] * masses["Na"],
               "all_Na_mass_over_existing_metal_mass": inventory["Na"] * masses["Na"] / phase_mass["metal"]},
        "limitations": ["Dry simulated hosts are not a wet-BSE empirical bracket.",
                        "Solvent mass and ideal gas fugacity remain fixed.",
                        "No finite He equilibrium or coupled pressure error was calculated.",
                        "Metal He and Na constitutive continuations remain unresolved."],
        "replay_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
