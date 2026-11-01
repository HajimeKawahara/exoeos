"""Keep the selected model-survey evidence attached to its recorded files."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "results/mceg/20261020_model_survey/receipt.json"
CURRENT = ROOT / "results/mceg/20261029_badro_default"
API_TRANSITION = ROOT / "results/mceg/20261101_material_response/evidence_transition.json"


def test_model_survey_selected_files_and_artifacts() -> None:
    receipt = json.loads(RECEIPT.read_text())
    transition = json.loads((CURRENT / "evidence_transition.json").read_text())
    updates = transition["historical_receipts"][str(RECEIPT.relative_to(ROOT))]["file_updates"]
    api_updates = json.loads(API_TRANSITION.read_text())["file_updates"]
    for name, expected in {**receipt["selected_file_sha256"], **receipt["artifacts"]}.items():
        for reviewed in (updates, api_updates):
            if name in reviewed:
                # The changed Ma model has separate evidence; API additions
                # have an explicit review. The survey itself stays historical.
                assert reviewed[name]["previous_sha256"] == expected
                expected = reviewed[name]["reviewed_sha256"]
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert actual == expected, f"Review model-survey evidence after changing {name}"


def test_provisional_evidence_is_current_and_preserves_historical_receipts() -> None:
    transition = json.loads((CURRENT / "evidence_transition.json").read_text())
    for path, record in transition["historical_receipts"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == record["sha256"]
    path = ROOT / transition["replacement_receipt"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == transition["replacement_receipt_sha256"]
    receipt = json.loads(path.read_text())
    api_updates = json.loads(API_TRANSITION.read_text())["file_updates"]
    for name, expected in receipt["selected_file_sha256"].items():
        if name in api_updates:
            assert api_updates[name]["previous_sha256"] == expected
            expected = api_updates[name]["reviewed_sha256"]
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    report = ROOT / receipt["junit_xml"]
    assert hashlib.sha256(report.read_bytes()).hexdigest() == receipt["junit_sha256"]
    suite = ET.parse(report).getroot().find("testsuite")
    assert int(suite.get("tests")) == receipt["tests_passed"] == 81
    assert all(int(suite.get(key, "0")) == 0 for key in ("failures", "errors", "skipped"))


def test_api_transition_preserves_the_previous_review():
    transition = json.loads(API_TRANSITION.read_text())
    previous = ROOT / transition["previous_transition"]
    assert previous == CURRENT / "evidence_transition.json"
    assert hashlib.sha256(previous.read_bytes()).hexdigest() == transition["previous_transition_sha256"]
    assert set(transition["file_updates"]) == {"src/exoeos/__init__.py", "src/exoeos/state.py"}


def test_model_survey_receipt_agrees_with_junit() -> None:
    receipt = json.loads(RECEIPT.read_text())
    suites = ET.parse(RECEIPT.with_name("tests.xml")).getroot().findall("testsuite")
    assert len(suites) == len(receipt["runs"])
    for suite, run in zip(suites, receipt["runs"]):
        counts = {key: int(suite.get(key, "0"))
                  for key in ("tests", "failures", "errors", "skipped")}
        assert counts == run["counts"]
        assert run["exit_code"] == 0 and counts["tests"] > 0
        assert counts["failures"] == counts["errors"] == counts["skipped"] == 0
        assert len(suite.findall("testcase")) == counts["tests"]
