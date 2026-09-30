"""Extract the unchanged OH primitive ledger and close a conditional He box."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
MATERIAL = HERE.parents[1]
sys.path.insert(0, str(MATERIAL))
from helium_metal_box import redistribute_helium
from helium_reference import DATA, helium_henry_coefficient


SOURCE_SHA = "f7b848f0c89740ac77010b56eccdaa6c5283c17216d6aeac24d2c99ea54b9235"
SOURCE_URL = ("https://github.com/HajimeKawahara/exoinventory/blob/"
              "ff13dfa994b2e1f658a46bc8f07c57560072ffac/examples/subneptune_taxonomy/"
              "finite_melt/validation/20260927_m2_finite_response/final_common_runtime_oh/closure.json")


def assess(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("Use the exact archived OH closure bytes")
    document = json.loads(raw)
    root = document["runs"][0]["roots"][0]
    if (not document["numerically_accepted"] or not all(root[key] for key in
            ("accepted", "contact_accepted", "column_numerically_accepted", "global_closure_numerically_accepted"))):
        raise ValueError("The retained original OH closure must be accepted")
    elements, names = root["elements"], root["source_component_order"]
    masses = dict(zip(elements, root["atomic_masses_kg_mol"]))
    amounts = dict(zip(names, root["source_component_amounts_mol"]))
    matrix, phases = root["source_formula_matrix"], root["source_phases"]
    inventory = dict(zip(document["inventory"]["elements"], document["inventory"]["total_element_amounts_mol"]))
    he_row = matrix[elements.index("He")]
    if any(count and name != "He_gas" for name, count in zip(names, he_row)):
        raise ValueError("This control requires the original gas-only He source")
    source_helium_residual = (amounts["He_gas"] - inventory["He"]) / inventory["He"]
    if (he_row[names.index("He_gas")] != 1 or abs(source_helium_residual) > 1e-9
            or "He_gas" not in phases["gas"]):
        raise ValueError("The entire finite He inventory must be primitive atomic He gas")
    mass = {name: sum(matrix[e][i] * masses[element] for e, element in enumerate(elements))
            for i, name in enumerate(names)}
    phase_mass = {phase: sum(amounts[name] * mass[name] for name in components)
                  for phase, components in phases.items()}
    dry_mass = sum(amounts[name] * mass[name] for name in phases["silicate"]
                   if name not in ("h2o_melts", "H2_dissolved"))
    other_gas = sum(amounts[name] for name in phases["gas"] if name != "He_gas")
    inputs = {
        "total_helium_mol": inventory["He"], "other_gas_mol": other_gas,
        "pressure_bar": root["pressure_base_pa"] / 1e5,
        "dry_silicate_mass_kg": dry_mass,
        "silicate_nonhelium_mass_kg": phase_mass["silicate"],
        "metal_nonhelium_mass_kg": phase_mass["metal"],
        "helium_molar_mass_kg": masses["He"],
    }
    old = json.loads((MATERIAL / "validation/20260928_na_helium/frozen_oh.json").read_text())
    if (old["source_sha256"] != SOURCE_SHA or old["dry_silicate_mass_kg"] != dry_mass
            or old["reconstructed_source_phase_mass_kg"] != phase_mass):
        raise ValueError("Independent primitive reconstruction differs from the previous frozen illustration")
    p_frozen = inputs["pressure_bar"] * amounts["He_gas"] / sum(amounts[name] for name in phases["gas"])
    evidence = json.loads(DATA.read_text())["references"]["bouhifd_2013"]
    rows = []
    for host in ("olivine", "MORB", "rhyolite"):
        coefficient = helium_henry_coefficient(host, root["temperature_base_k"])
        capacity = coefficient["He_mol_per_kg_dry_host_per_bar_fugacity"]
        control = redistribute_helium(**inputs, henry_mol_per_kg_dry_bar=capacity,
                                      metal_silicate_mass_fraction_D=0.)
        trials = []
        for measurement in evidence["S2_same_run_Fe80Ni20"]:
            result = redistribute_helium(**inputs, henry_mol_per_kg_dry_bar=capacity,
                                         metal_silicate_mass_fraction_D=measurement["reported_D"])
            result["primary_measurement"] = measurement
            result["delta_from_D_zero"] = {
                "gas_He_mol": result["helium_mol"]["gas"] - control["helium_mol"]["gas"],
                "silicate_He_mol": result["helium_mol"]["silicate"] - control["helium_mol"]["silicate"],
                "ideal_He_partial_pressure_bar": result["He_partial_pressure_bar"] - control["He_partial_pressure_bar"],
            }
            if abs(result["helium_budget_relative_residual"]) > 1e-14:
                raise ArithmeticError("The conditional He inventory did not close")
            trials.append(result)
        rows.append({"henry_reference": coefficient,
                     "previous_frozen_silicate_He_mol": dry_mass * capacity * p_frozen,
                     "finite_He_D_zero_control": control, "metal_partition_trials": trials})
    files = [Path(__file__), MATERIAL / "helium_metal_box.py", MATERIAL / "helium_reference.py", DATA,
             MATERIAL / "validation/20260928_na_helium/frozen_oh.json"]
    return {
        "assessment_id": "fixed_host_finite_He_metal_partition_diagnostic_v1",
        "source": {"sha256": SOURCE_SHA, "archive": SOURCE_URL,
                   "temperature_K": root["temperature_base_k"], "pressure_bar": inputs["pressure_bar"],
                   "primitive_ledger": {key: root[key] for key in ("elements", "atomic_masses_kg_mol",
                       "source_component_order", "source_component_amounts_mol", "source_formula_matrix", "source_phases")}},
        "derived_fixed_box_inputs": inputs,
        "original_primitive_He_mol": amounts["He_gas"],
        "original_primitive_relative_He_budget_residual": source_helium_residual,
        "finite_box_inventory_policy": "Use the original global He inventory; retain the tiny original primitive rounding residual separately.",
        "primitive_source_phase_mass_kg": phase_mass,
        "original_ideal_He_partial_pressure_bar": p_frozen,
        "primary_partition_reference": evidence,
        "rows": rows,
        "assumptions": [
            "Each same-run Table S2 D is held invariant in pressure, temperature and alloy/host composition.",
            "Henry capacity is a dry simulated-host interpolation, applied to the fixed OH dry silicate mass.",
            "D is a ratio of He-inclusive total phase mass fractions; the wet non-He silicate includes H2O and H2.",
            "Non-He gas amounts, dry/wet host amounts, T and total pressure remain fixed; ideal He fugacity changes with the complete gas denominator.",
            "The finite phase mass denominators are exact within this box, but the extrapolated solubility/partition laws are dilute approximations.",
        ],
        "supports_material_admission": False, "is_BSE_error_bound": False,
        "coupled_source_equilibrium_performed": False, "pressure_closure_performed": False,
        "limitations": ["The 1.9--13.1 GPa Fe80Ni20/CI-chondrite measurements do not bound the 0.027 GPa Fe-Si-O-H source.",
                        "The five reported central D values are separate scenarios, not a statistical or empirical BSE uncertainty interval.",
                        "No host/redox, non-He gas, chemical potential, density, column or bottom-pressure response is solved."],
        "input_and_code_sha256": {str(path.relative_to(MATERIAL)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.source.read_bytes())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
