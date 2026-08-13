"""Sphinx configuration for ExoEOS."""

import os
from pathlib import Path
import sys


os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/exoeos_matplotlib")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

import exoeos


expected_package_root = (SOURCE_ROOT / "exoeos").resolve()
imported_package_root = Path(exoeos.__file__).resolve().parent
if imported_package_root != expected_package_root:
    raise RuntimeError(
        "Imported exoeos from outside this repository: "
        f"{imported_package_root} != {expected_package_root}"
    )

project = "ExoEOS"
copyright = "2026, ExoEOS contributors"
author = "ExoEOS contributors"
release = exoeos.__version__

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
