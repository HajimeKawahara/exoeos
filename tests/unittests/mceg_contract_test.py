"""Require review when files covered by the H2 evidence snapshot change."""

import hashlib
import json
from pathlib import Path


def test_h2_evidence_matches_selected_source_files() -> None:
    root = Path(__file__).resolve().parents[2]
    receipt = json.loads((root / "results/mceg/h2_tests_d7681db.json").read_text())
    for name, expected in receipt["selected_file_sha256"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == expected, f"Review MCEG evidence after changing {name}"
    report = root / receipt["junit_xml"]
    assert hashlib.sha256(report.read_bytes()).hexdigest() == receipt["junit_sha256"]
