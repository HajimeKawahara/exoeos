"""Conditionally transfer primary Na limits to the fixed archived OH boxes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


MATERIAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MATERIAL))
from sodium_fixed_box import partition_sodium, critical_partition_coefficient
from sodium_reference import replay

SOURCE_SHA = "f7b848f0c89740ac77010b56eccdaa6c5283c17216d6aeac24d2c99ea54b9235"
SOURCE_URL = ("https://github.com/HajimeKawahara/exoinventory/blob/"
              "ff13dfa994b2e1f658a46bc8f07c57560072ffac/examples/subneptune_taxonomy/"
              "finite_melt/validation/20260927_m2_finite_response/final_common_runtime_oh/closure.json")


def assess(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("Use the exact archived OH closure bytes.")
    document = json.loads(raw)
    root = document["runs"][0]["roots"][0]
    if not document["numerically_accepted"] or not root["accepted"]:
        raise ValueError("The original accepted root is required.")
    elements, names = root["elements"], root["source_component_order"]
    amounts = dict(zip(names, root["source_component_amounts_mol"]))
    masses = dict(zip(elements, root["atomic_masses_kg_mol"]))
    matrix, phases = root["source_formula_matrix"], root["source_phases"]
    sodium = dict(zip(names, matrix[elements.index("Na")]))
    mass = {name: math.fsum(matrix[e][i] * masses[element]
                           for e, element in enumerate(elements))
            for i, name in enumerate(names)}
    phase_mass = {phase: math.fsum(amounts[n] * mass[n] for n in components)
                  for phase, components in phases.items()}
    phase_na = {phase: math.fsum(amounts[n] * sodium[n] for n in components)
                for phase, components in phases.items()}
    inventory = dict(zip(document["inventory"]["elements"],
                         document["inventory"]["total_element_amounts_mol"]))
    other = math.fsum(n for phase, n in phase_na.items() if phase not in ("silicate", "metal"))
    active = inventory["Na"] - other
    primitive_residual = (math.fsum(phase_na.values()) - inventory["Na"]) / inventory["Na"]
    if abs(primitive_residual) > 1e-9:
        raise ValueError("The primitive Na normalization must match the global inventory.")
    inputs = dict(active_sodium_mol=active,
                  silicate_non_sodium_mass_kg=phase_mass["silicate"] - masses["Na"] * phase_na["silicate"],
                  metal_non_sodium_mass_kg=phase_mass["metal"] - masses["Na"] * phase_na["metal"],
                  sodium_molar_mass_kg=masses["Na"])
    target = .01 * inventory["Na"]
    critical = critical_partition_coefficient(**inputs, target_metal_sodium_mol=target)
    observations = [row for row in replay() if row["metal_Na_mass_ppm"]["censored_upper"]]
    rows = []
    for observed in observations:
        trials = {}
        for name, d in (("central_silicate", observed["mass_partition_ratio"]),
                        ("reported_silicate_minus_2SE", observed["mass_partition_upper_at_reported_silicate_lower"])):
            result = partition_sodium(**inputs, metal_silicate_mass_fraction_D=d)
            result["fraction_of_global_Na_in_metal"] = result["sodium_mol"]["metal"] / inventory["Na"]
            result["metal_Na_total_phase_mass_ppm"] = 1e6 * result["mass_fraction_metal"]
            result["global_Na_budget_residual_mol"] = math.fsum((result["sodium_mol"]["metal"],
                                                                result["sodium_mol"]["silicate"], other, -inventory["Na"]))
            result["D_critical_over_declared_D"] = critical / d
            trials[name] = result
        rows.append(dict(primary_observation=observed, fixed_box_trials=trials))
    threshold = partition_sodium(**inputs, metal_silicate_mass_fraction_D=critical)
    maximum = max(row["mass_partition_ratio"] for row in observations)
    files = [Path(__file__), MATERIAL / "sodium_fixed_box.py", MATERIAL / "sodium_reference.py",
             MATERIAL / "sodium_reference_sources.json"]
    return dict(
        assessment_id="fixed_host_finite_Na_partition_diagnostic_v1",
        source=dict(sha256=SOURCE_SHA, archive=SOURCE_URL,
                    temperature_K=root["temperature_base_k"], pressure_bar=root["pressure_base_pa"] / 1e5,
                    primitive_ledger={key: root[key] for key in ("elements", "atomic_masses_kg_mol",
                        "source_component_order", "source_component_amounts_mol", "source_formula_matrix", "source_phases")}),
        global_Na_inventory_mol=inventory["Na"], original_primitive_phase_Na_mol=phase_na,
        original_primitive_relative_Na_budget_residual=primitive_residual,
        original_primitive_phase_mass_kg=phase_mass, fixed_other_phase_Na_mol=other,
        fixed_box_inputs=inputs, rows=rows,
        target=dict(origin="Existing M2 engineering partition target, not an empirical uncertainty.",
                    fraction_of_global_Na=.01, metal_Na_mol=target, D_critical=critical,
                    largest_reported_D_at_central_silicate=maximum,
                    D_critical_over_largest_reported_D=critical / maximum,
                    threshold_box_check=threshold),
        assumptions=[
            "Each 1GPa experimental D upper is held invariant in pressure, temperature and composition for this diagnostic only.",
            "Only metal and silicate Na redistribute; primitive gas/cloud Na and every non-Na host amount remain fixed.",
            "Global Na minus fixed primitive other-phase Na sets the active inventory; retain the original primitive/global rounding residual separately.",
            "Both D denominators include their redistributed Na and all remaining host mass, including water and dissolved H2 in silicate.",
            "The alternative silicate-minus-2SE input is not a new joint confidence interval.",
        ],
        limitations=[
            "No low-pressure BSE empirical upper bound or absolute Na standard is established.",
            "No silicate component speciation, redox, host chemical potentials, density or bottom-pressure response is solved.",
            "The ratio Dcrit/D quantifies the required extrapolation under the fixed-host assumptions; it does not assign a probability or bound to that extrapolation.",
        ], supports_material_admission=False, is_BSE_error_bound=False,
        coupled_source_equilibrium_performed=False, pressure_closure_performed=False,
        input_and_code_sha256={str(p.relative_to(MATERIAL)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


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
