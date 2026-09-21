"""Offline checks of the pinned BSE material audit and its exact atom mapping."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


DATA = Path(__file__).resolve().parents[2] / "examples/m2_material"
spec = importlib.util.spec_from_file_location("validate_bse", DATA / "validate_bse.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_bse_mapping_preserves_atoms_and_does_not_import_hydrogen():
    ledger = json.loads((DATA / "bse_inventory.json").read_text())
    n, scale = audit.component_amounts(ledger)
    assert np.count_nonzero(n) == 10
    elements = dict(zip(audit.melts.ELEMENTS, audit.melts.FORMULA_MATRIX.T @ n))
    expected = dict(zip(ledger["elements"], np.array(ledger["rock_element_amounts_mol"]) * scale))
    for name, amount in expected.items():
        np.testing.assert_allclose(elements.get(name, 0.), amount, rtol=1e-14, atol=0.)
    assert elements["H"] == 0
    assert n[audit.melts.COMPONENTS.index("fe2o3")] == 0
    assert sum(np.array(ledger["total_element_amounts_mol"])[-2:]) > 0
    mass = np.array(ledger["rock_element_amounts_mol"]) @ ledger["atomic_masses_kg_mol"]
    np.testing.assert_allclose(mass * scale, 0.1, rtol=1e-14)


def test_native_receipt_is_pinned_and_scientific_gates_remain_pending():
    receipt = json.loads((DATA / "bse_liquid_validation.json").read_text())
    contract = json.loads((DATA / "material_contract.json").read_text())
    for filename, field in (("bse_inventory.json", "ledger_sha256"), ("bse_source.json", "source_sha256")):
        assert hashlib.sha256((DATA / filename).read_bytes()).hexdigest() == receipt["input"][field]
    assert hashlib.sha256((DATA / "validate_bse.py").read_bytes()).hexdigest() == receipt["generator_sha256"]
    assert receipt["checks"]["native_evaluations"] == 22
    assert receipt["checks"]["maximum_derivative_error_J_mol"] < 0.02
    state = receipt["state"]
    np.testing.assert_allclose(state["gibbs_RT"], state["gibbs_J"] / (state["basis"]["common_R_J_mol_K"] * state["T_K"]))
    assert contract["scientific_gates"] == {"M2_A": "pending", "M2_B": "pending"}
    assert contract["accepted_coupled_material_domain"] is None
    assert contract["omission_error_budget"]["estimated_effects"] is None
    ledger = receipt["input"]["ledger"]
    masses = np.asarray(ledger["rock_element_amounts_mol"]) * ledger["atomic_masses_kg_mol"]
    by_element = dict(zip(ledger["elements"], masses))
    bound = contract["initial_omission_inventory_bounds"]
    np.testing.assert_allclose(bound["Al_Ca_K_Ti_Cr_P_mass_kg"],
                               sum(by_element[name] for name in ("Al", "Ca", "K", "Ti", "Cr", "P")))
    np.testing.assert_allclose(bound["Mg_mass_kg"], by_element["Mg"])


def test_domain_evidence_matches_receipt_and_preserves_distinct_materials():
    from exoeos.marcum_silicate_hydrogen import MarcumSilicateHydrogenTableLoader

    contract = json.loads((DATA / "material_contract.json").read_text())
    evidence = json.loads((DATA / contract["domain_audit"]["evidence_file"]).read_text())
    receipt = json.loads((DATA / "bse_liquid_validation.json").read_text())
    candidate = evidence["candidate_assessment"]
    sources = evidence["evidence"]
    assert candidate["temperature_K"] == receipt["state"]["T_K"]
    assert candidate["pressure_Pa"] == receipt["state"]["P_Pa"]
    missing = {item["id"] for item in contract["missing_material_evidence"]}
    assert set(candidate["blocking_evidence_ids"]) <= missing
    for item in contract["missing_material_evidence"]:
        assert set(item.get("evidence_ids", ())) <= sources.keys()
    for source in sources.values():
        for filename in source.get("source_files", ()):
            assert (DATA / filename).is_file()
    loader = MarcumSilicateHydrogenTableLoader()
    marcum = sources["marcum_2026"]
    assert tuple(marcum["table_temperature_K"]) == loader.table_domain["temperature_K"]
    assert tuple(marcum["table_pressure_Pa"]) == loader.table_domain["pressure_Pa"]
    assert sources["melts_guidance"]["temperature_K"][1] < marcum["table_temperature_K"][0]
    assert sources["chaudhari_2025"]["temperature_K"][1] < candidate["temperature_K"]
    assert sources["chaudhari_2025"]["pressure_Pa"][0] > candidate["pressure_Pa"]
    assert candidate["accepted_coupled_material_domain"] is None
