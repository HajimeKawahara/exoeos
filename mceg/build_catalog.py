"""Package the reviewed contracts without importing ExoEOS or MCEG.

Run ``python mceg/build_catalog.py`` after reviewing a new evidence snapshot.
Generated copies are committed so wheel builds need no evidence tooling.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src/exoeos/data/mceg"
RECORD_VERSION = "2026-10-30"

# Explicit model-level selections preserve the complete recorded history.
# They do not choose a current or most favorable scientific assessment.
MODELS = [
    ("ma_fe_si_o", "exoeos.ma_fe_si_o.v1", "Ma Fe-Si-O liquid with printed Young coefficients", [
        "numerical_review", "physical_review", "physical_review_20261021",
        "physical_review_20261029", "physical_primary_review_20261029", "physical_badro_review_20261029",
    ]),
    ("ma_fe_si_o_h", "exoeos.ma_fe_si_o_h.v1", "Formal Fe-Si-O-H liquid with ideal H dilution", [
        "numerical_review", "physical_review", "physical_source_review_20261029",
    ]),
    ("magma_carbonate_liquid", "exoeos.magma_carbonate_liquid.v1",
     "Native rhyolite-MELTS mode-4 carbon-bearing liquid", [
         "numerical_review", "physical_review", "physical_review_20261021",
         "physical_full_text_review_20261021", "physical_supplement_review_20261021",
     ]),
    ("magma_dry_liquid", "exoeos.magma_dry_liquid.v1", "Native MELTS dry silicate liquid", [
        "numerical_review", "physical_review", "physical_review_20261021",
    ]),
    ("magma_fluid", "exoeos.magma_fluid.v1", "Magma-reference H2O and H2O-CO2 fluid Gibbs energy", [
        "numerical_review", "physical_review", "physical_review_20261021",
        "physical_supplement_review_20261021", "physical_zd2009_review_20261021",
    ]),
    ("magma_hydrous_liquid", "exoeos.magma_hydrous_liquid.v1", "Native MELTS mode-1 hydrous silicate liquid", [
        "numerical_review", "physical_review", "physical_review_20261021",
    ]),
    ("magma_volatiles", "exoeos.magma_volatiles.v1", "Conditional dry11 liquid with H2O, H2 and trace He", [
        "numerical_review", "physical_review", "physical_review_20261029",
        "physical_remaining_review_20261029", "physical_jambon_review_20261029",
    ]),
    ("mass_fraction_solute", "exoeos.mass_fraction_solute.v1",
     "Extensive solute on a total-liquid mass-fraction basis", [
         "numerical_review", "physical_review",
     ]),
    ("zhang_duan_h2", "exoeos.zhang_duan_h2", "Zhang-Duan 2009 EOS for pure molecular hydrogen", [
        "imported:ideal_density_limit", "imported:published_parameter_loading", "imported:physical_accuracy",
        "numerical_review_20261021", "physical_review_20261021",
    ]),
]


def build_catalog(output: Path = OUTPUT) -> None:
    """Copy canonical contract bytes and index the explicitly selected assessments."""
    models = []
    for name, model_id, label, assessments in MODELS:
        relative = f"contracts/{name}.yaml"
        content = (ROOT / "mceg" / relative).read_bytes()
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        models.append({
            "id": model_id,
            "label": label,
            "species": [],
            "records": [{
                "id": name,
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "species": ["*"],
                "assessments": assessments,
            }],
        })
    catalog = {"schema_version": "1", "provider": "exoeos",
               "record_version": RECORD_VERSION, "models": models}
    (output / "index.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_catalog()
