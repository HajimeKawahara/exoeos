#!/usr/bin/env python3
"""Convert committed tutorial notebooks to deterministic RST and assets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from typing import Mapping

try:
    import nbformat
    from nbconvert import RSTExporter
except ImportError as exc:  # pragma: no cover - depends on the docs environment
    raise SystemExit(
        "Notebook conversion requires nbformat and nbconvert. Install the "
        "documentation dependencies with `python -m pip install -e '.[docs]'`."
    ) from exc


NOTEBOOK_DIRECTORY = Path(__file__).resolve().parent
NOTEBOOK_STEMS = (
    "fixed_composition_cho_eos_comparison",
    "peng_robinson_fixed_state_reference",
)
GENERATED_HEADER = (
    ".. This file is generated from the sibling .ipynb by convert_notebooks.py.\n"
    ".. Do not edit this RST file directly.\n\n"
)


def _ensure_pandoc() -> None:
    """Expose the pinned docs-extra Pandoc binary to nbconvert."""

    try:
        import pypandoc
    except ImportError as exc:  # pragma: no cover - depends on docs environment
        raise SystemExit(
            "RST export requires Pandoc. Install the documentation dependencies "
            "with `python -m pip install -e '.[docs]'`."
        ) from exc
    pandoc_path = Path(pypandoc.get_pandoc_path()).resolve()
    # Always put the docs-extra binary first.  Using an arbitrary system
    # Pandoc makes the committed RST depend on the developer's environment.
    os.environ["PATH"] = os.pathsep.join(
        (str(pandoc_path.parent), os.environ.get("PATH", ""))
    )


def _separate_directives(body: str) -> str:
    """Insert a blank line before top-level RST directives."""

    lines = [line.rstrip() for line in body.splitlines()]
    separated = []
    for line in lines:
        if line.startswith(".. ") and separated and separated[-1].strip():
            separated.append("")
        separated.append(line)
    return "\n".join(separated)


def _export_notebook(notebook_path: Path) -> tuple[str, Mapping[str, bytes]]:
    """Return deterministic RST text and binary resources without execution."""

    _ensure_pandoc()
    notebook = nbformat.read(notebook_path, as_version=4)
    exporter = RSTExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    resources = {
        "unique_key": notebook_path.stem,
        "output_files_dir": f"{notebook_path.stem}_files",
    }
    body, exported_resources = exporter.from_notebook_node(
        notebook,
        resources=resources,
    )
    body = _separate_directives(body)
    download = (
        f":download:`Download the executable notebook <{notebook_path.name}>`\n\n"
    )
    rst = GENERATED_HEADER + download + body.rstrip() + "\n"
    return rst, exported_resources.get("outputs", {})


def _validate_resource_name(name: str, stem: str) -> Path:
    """Reject exporter output paths outside the notebook-specific asset tree."""

    relative_path = Path(name)
    expected_directory = f"{stem}_files"
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or not relative_path.parts
        or relative_path.parts[0] != expected_directory
    ):
        raise ValueError(f"Unsafe nbconvert resource path: {name!r}")
    return relative_path


def _actual_resources(stem: str) -> Mapping[Path, bytes]:
    asset_directory = NOTEBOOK_DIRECTORY / f"{stem}_files"
    if asset_directory.is_symlink():
        raise ValueError(f"Refusing to read symlinked asset path: {asset_directory}")
    if not asset_directory.is_dir():
        return {}
    return {
        path.relative_to(NOTEBOOK_DIRECTORY): path.read_bytes()
        for path in sorted(asset_directory.rglob("*"))
        if path.is_file()
    }


def _check_notebook(stem: str, rst: str, outputs: Mapping[str, bytes]) -> bool:
    rst_path = NOTEBOOK_DIRECTORY / f"{stem}.rst"
    expected_resources = {
        _validate_resource_name(name, stem): content
        for name, content in outputs.items()
    }
    return (
        rst_path.is_file()
        and rst_path.read_text(encoding="utf-8") == rst
        and _actual_resources(stem) == expected_resources
    )


def _write_notebook(stem: str, rst: str, outputs: Mapping[str, bytes]) -> None:
    rst_path = NOTEBOOK_DIRECTORY / f"{stem}.rst"
    with rst_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(rst)

    asset_directory = NOTEBOOK_DIRECTORY / f"{stem}_files"
    if asset_directory.exists():
        if asset_directory.is_symlink() or not asset_directory.is_dir():
            raise ValueError(f"Refusing to replace unsafe asset path: {asset_directory}")
        shutil.rmtree(asset_directory)

    for name, content in sorted(outputs.items()):
        output_path = NOTEBOOK_DIRECTORY / _validate_resource_name(name, stem)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stems", nargs="*", metavar="STEM")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check committed generated files without modifying them.",
    )
    args = parser.parse_args()

    selected_stems = args.stems or NOTEBOOK_STEMS
    unknown_stems = sorted(set(selected_stems) - set(NOTEBOOK_STEMS))
    if unknown_stems:
        parser.error(
            "unknown notebook stem(s): "
            + ", ".join(unknown_stems)
            + "; expected one of: "
            + ", ".join(NOTEBOOK_STEMS)
        )

    stale = []
    for stem in selected_stems:
        notebook_path = NOTEBOOK_DIRECTORY / f"{stem}.ipynb"
        if not notebook_path.is_file():
            raise FileNotFoundError(f"Missing notebook source: {notebook_path}")
        rst, outputs = _export_notebook(notebook_path)
        if args.check:
            if not _check_notebook(stem, rst, outputs):
                stale.append(stem)
        else:
            _write_notebook(stem, rst, outputs)
            print(f"converted {notebook_path.name} -> {stem}.rst")

    if stale:
        print("stale generated notebook documentation: " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
