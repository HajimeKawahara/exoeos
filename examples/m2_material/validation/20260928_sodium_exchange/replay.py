"""Replay retained primary Na data without native or equilibrium calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


root = Path(__file__).resolve().parents[2]
path = root / "sodium_reference.py"
spec = importlib.util.spec_from_file_location("sodium_reference", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rows = module.replay()
result = dict(schema="m2_sodium_reference_replay_v1", observations=rows,
              atomic_masses_g_per_mol=module.ATOMIC_MASS,
              scope="Primary reference replay and reported censoring diagnostics; no source species, solver, native call or calibrated BSE bound.",
              within_reported_FeS_errors={
                  basis: all(abs(row["concentration_bases"][basis]["residual_dex"])
                             < row["concentration_bases"][basis]["published_two_standard_errors"]
                             for row in rows if row["run"] in ("GGK4", "GGK5b", "GGK6"))
                  for basis in ("one_cation", "oxide_molecules")},
              experimental_pressure_GPa=1.,
              declared_source_pressure_GPa=None,
              detection_limit_confidence_definition=None,
              absolute_common_standard_calibrated=False,
              supports_material_admission=False,
              file_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (path, module.DATA, Path(__file__))})
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
