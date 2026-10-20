"""The repository owns HTML compilation after MCEG source generation."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("generation_fails", [False, True])
def test_update_doc_generates_before_html_and_stops_on_invalid_inputs(tmp_path, generation_fails):
    script = tmp_path / "update_doc.sh"
    shutil.copyfile(ROOT / "update_doc.sh", script)
    (tmp_path / "mceg").mkdir()
    shutil.copyfile(ROOT / "mceg/docs.yaml", tmp_path / "mceg/docs.yaml")
    binary = tmp_path / "bin"
    binary.mkdir()
    log = tmp_path / "calls.jsonl"
    python = binary / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with open(os.environ['MCEG_TEST_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:4] == ['-m', 'mceg', 'docs']:\n"
        "    assert Path(sys.argv[-1]).is_file()\n"
        "    sys.exit(int(os.environ['MCEG_TEST_FAIL']))\n"
    )
    python.chmod(0o755)
    env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ.get("PATH", ""),
           "MCEG_TEST_LOG": str(log), "MCEG_TEST_FAIL": str(int(generation_fails))}
    result = subprocess.run(["bash", str(script), "-W", "--keep-going"],
                            cwd=tmp_path.parent, env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    generate = ["-m", "mceg", "docs", "--config", "mceg/docs.yaml"]
    position = calls.index(generate)
    if generation_fails:
        assert result.returncode != 0
        assert calls[position:] == [generate]
    else:
        assert result.returncode == 0, result.stderr
        assert calls[position + 1][:4] == ["-m", "sphinx", "-M", "clean"]
        assert calls[position + 2][:4] == ["-m", "sphinx", "-M", "html"]
        assert calls[position + 2][-2:] == ["-W", "--keep-going"]
