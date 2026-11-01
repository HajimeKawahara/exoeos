"""Require review when files covered by the H2 evidence snapshot change."""

import hashlib
import json
from pathlib import Path


def test_h2_evidence_matches_selected_source_files() -> None:
    root = Path(__file__).resolve().parents[2]
    receipt_path = "results/mceg/h2_tests_d7681db.json"
    receipt = json.loads((root / receipt_path).read_text())
    transition = json.loads((root / "results/mceg/20261029_badro_default/evidence_transition.json").read_text())
    updates = transition["historical_receipts"][receipt_path]["file_updates"]
    api_updates = json.loads((root / "results/mceg/20261101_material_response/evidence_transition.json").read_text())["file_updates"]
    for name, expected in receipt["selected_file_sha256"].items():
        for reviewed in (updates, api_updates):
            if name in reviewed:
                # Follow exact reviewed changes without rewriting the receipt.
                assert reviewed[name]["previous_sha256"] == expected
                expected = reviewed[name]["reviewed_sha256"]
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == expected, f"Review MCEG evidence after changing {name}"
    report = root / receipt["junit_xml"]
    assert hashlib.sha256(report.read_bytes()).hexdigest() == receipt["junit_sha256"]
