"""Verify this archive or replay its preserved scripts without native calls.

Only absolute scratch-path string literals are relocated. The original script
bytes and saved results remain unchanged. Numerical replay is finite evidence,
not a global minimum, native uniform-error, or stability certificate.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_archive():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for record in manifest["files"]:
        path = ROOT / record["path"]
        assert path.stat().st_size == record["bytes"], path
        assert digest(path) == record["sha256"], path
    summaries = []
    for group, phase, index in [("rhm", "rhm-oxide", "26"),
                                ("plagioclase", "plagioclase", "14")]:
        receipt = json.loads((ROOT / group / f"stage2-{group}-validation.json").read_text())
        assert receipt["parameter_sha256"] == digest(ROOT / group / f"stage2-{group}-parameters.json")
        assert receipt["archive_sha256"] == digest(ROOT / "inputs/native_trials" / f"phase_{index}_{phase}.json")
        records = receipt["records"]
        count = receipt.get("comparison_count", receipt.get("finite_comparison_count"))
        assert len(records) == count
        maximum = max(abs(row["difference_J_mol"]) for row in records)
        assert maximum == receipt["maximum_absolute_difference_J_mol"]
        summaries.append({"phase": phase, "comparison_count": count,
                          "maximum_absolute_difference_J_mol": maximum})
    receipt = json.loads((ROOT / "pyx-spinel/stage2-pyx-spinel-validation.json").read_text())
    assert receipt["parameter_sha256"] == digest(ROOT / "pyx-spinel/stage2-pyx-spinel-models.json")
    for row in receipt["phases"]:
        phase = row["phase"]
        index = {"clinopyroxene": "05", "orthopyroxene": "04", "spinel": "25"}[phase]
        assert row["archive_sha256"] == digest(ROOT / "inputs/native_trials" / f"phase_{index}_{phase}.json")
        records = [x for x in row["records"] if "difference_J_mol" in x]
        assert len(records) == row["comparison_count"]
        maximum = max(abs(x["difference_J_mol"]) for x in records)
        assert maximum == row["maximum_absolute_difference_J_mol"]
        assert sum(not x["success"] for x in records) == row["optimizer_failure_count"]
        summaries.append({"phase": phase, "comparison_count": len(records),
                          "maximum_absolute_difference_J_mol": maximum})
    return {"status": "archive_integrity_and_receipt_links_verified",
            "files_checked": len(manifest["files"]), "finite_comparisons": summaries,
            "new_native_calls": 0, "new_optimizations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--group", choices=["rhm", "plagioclase", "pyx-spinel"])
    parser.add_argument("--mode", choices=["validate", "transcribe"], default="validate")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    summary = check_archive()
    if args.check_only:
        print(json.dumps(summary, indent=2))
        return
    if not args.group or args.output_directory is None:
        parser.error("Replay requires --group and --output-directory.")
    output = args.output_directory.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("Replay outputs must be outside the immutable archive.")
    if args.mode == "transcribe":
        if args.group == "plagioclase" and args.binary is None:
            parser.error("Plagioclase transcription requires --binary.")
        if args.group != "plagioclase" and args.source_directory is None:
            parser.error("Source transcription requires --source-directory.")
    output.mkdir(parents=True, exist_ok=True)
    mapping = {
        "/tmp/stage2-liquid-global-fixed2-h1e24/host_properties.json": str(ROOT / "inputs/host_properties.json"),
        "/tmp/stage2-solid-standards": str(ROOT / "inputs/native_standards"),
        "/tmp/stage2-stability-supported-20260927/results/m2_stability_search/20260927/explicit_coordinates": str(ROOT / "inputs/native_trials"),
        "/tmp/stage2-solid-eos-20260927/examples/m2_solid_mixing": str(ROOT / "pyx-spinel"),
        "/tmp/stage2-solid-eos-20260927/examples": str(ROOT / "pyx-spinel"),
    }
    for group, suffix in [("rhm", "parameters"), ("plagioclase", "parameters"),
                          ("pyx-spinel", "models")]:
        name = f"stage2-{group}-{suffix}.json"
        mapping["/tmp/" + name] = str((ROOT / group if args.mode == "validate" else output) / name)
        mapping[f"/tmp/stage2-{group}-validation.json"] = str(output / f"stage2-{group}-validation.json")
    mapping["/tmp/stage2-plagioclase"] = str(output / "stage2-plagioclase")
    if args.source_directory is not None:
        mapping["/tmp/stage2-magma-source"] = str(args.source_directory.resolve())
    if args.binary is not None:
        mapping["/tmp/exoeos-pr3-melts/runtime/alphamelts-py-2.3.2-ubuntu_22_04-x86_64/libalphamelts.so"] = str(args.binary.resolve())

    class RelocatePaths(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                for original in sorted(mapping, key=len, reverse=True):
                    if node.value == original or node.value.startswith(original + "/"):
                        node.value = mapping[original] + node.value[len(original):]
                        break
            return node

    suffix = "validate.py" if args.mode == "validate" else {
        "rhm": "transcription.py", "plagioclase": "disassemble.py",
        "pyx-spinel": "prototype.py"}[args.group]
    script = ROOT / args.group / f"stage2-{args.group}-{suffix}"
    syntax = RelocatePaths().visit(ast.parse(script.read_text(), filename=str(script)))
    ast.fix_missing_locations(syntax)
    sys.argv = [str(script)]
    sys.dont_write_bytecode = True
    exec(compile(syntax, str(script), "exec"), {"__name__": "__main__", "__file__": str(script)})


if __name__ == "__main__":
    main()
