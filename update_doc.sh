#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIRECTORY}"

python documents/tutorials/convert_notebooks.py --check
python -m sphinx -M clean documents documents/_build
python -m sphinx \
  -M html \
  documents \
  documents/_build \
  --fresh-env \
  --write-all \
  "$@"
