"""Keep the selected model-survey evidence attached to its recorded files."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "results/mceg/20261020_model_survey/receipt.json"


def test_model_survey_selected_files_and_artifacts() -> None:
    receipt = json.loads(RECEIPT.read_text())
    for name, expected in {**receipt["selected_file_sha256"], **receipt["artifacts"]}.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert actual == expected, f"Review model-survey evidence after changing {name}"


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
