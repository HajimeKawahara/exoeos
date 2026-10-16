"""Pinned numerical inputs shared by the conditional native material laws."""

import json
from pathlib import Path

DATA = json.loads((Path(__file__).parent / "data/conditional_materials.json").read_text())
R = 8.31446261815324
