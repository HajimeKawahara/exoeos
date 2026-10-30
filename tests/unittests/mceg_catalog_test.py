"""Keep packaged evidence synchronized with its canonical repository records."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PACKAGED = ROOT / "src/exoeos/data/mceg"


def test_catalog_rebuild_needs_only_the_standard_library(tmp_path):
    script = """
import runpy
import sys
from pathlib import Path
builder = runpy.run_path(sys.argv[1])
builder['build_catalog'](Path(sys.argv[2]))
assert not {'exoeos', 'jax', 'mceg'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-S", "-c", script, str(ROOT / "mceg/build_catalog.py"), str(tmp_path)],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    rebuilt = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    committed = {p.relative_to(PACKAGED): p.read_bytes() for p in PACKAGED.rglob("*") if p.is_file()}
    assert rebuilt == committed, "Review and regenerate the packaged MCEG catalog after changing its sources."


def test_catalog_covers_canonical_contracts_with_matching_hashes():
    catalog = json.loads((PACKAGED / "index.json").read_text())
    assert catalog["schema_version"] == "1"
    assert catalog["provider"] == "exoeos"
    models = catalog["models"]
    assert len({model["id"] for model in models}) == len(models)
    paths = []
    for model in models:
        assert model["species"] == []  # Model-level records declare no species-specific coverage.
        for record in model["records"]:
            path = Path(record["path"])
            assert not path.is_absolute() and ".." not in path.parts
            content = (PACKAGED / path).read_bytes()
            assert content == (ROOT / "mceg" / path).read_bytes()
            assert hashlib.sha256(content).hexdigest() == record["sha256"]
            assert record["species"] == ["*"]
            paths.append(path)
    expected = {p.relative_to(ROOT / "mceg") for p in (ROOT / "mceg/contracts").glob("*.yaml")}
    assert len(paths) == len(expected) and set(paths) == expected
