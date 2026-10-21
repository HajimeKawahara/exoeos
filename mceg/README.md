# Experimental MCEG contracts

[MCEG](https://github.com/HajimeKawahara/mceg) records model assumptions,
applicability, claims and scoped evidence. This checkout pilots its frozen
`schema_version: "0.1"` with
[pure-H2 Zhang-Duan](contracts/zhang_duan_h2.yaml). MCEG is not an ExoEOS
runtime or test dependency; the model implementation is unchanged.

The [2026-10-20 model inventory](model_inventory.md) adds v0.2 contracts for the
solubility/magma survey, with separate numerical and physical Assessments and
fresh receipts. The new records use the companion MCEG volatile-quantity
extension. Existing H2 and source-study histories retain their original scope.

The [Ma Fe-Si-O contract](contracts/ma_fe_si_o.yaml) adds a 2026-10-21 source
review of Young's printed equations, the Vogel/Buchan reproductions and pinned
author/GCE code. The printed/code oxygen coefficient difference is recorded;
original Ma and Badro supplementary calibration data remain pending. Inspect
the new assessment separately from the retained survey history:

```bash
mceg explain mceg/contracts/ma_fe_si_o.yaml --assessment physical_review_20261021 --root .
```

Source inspection does not establish physical accuracy. Remote artifact hashes
identify downloaded bytes; offline MCEG checks do not verify those URLs.

## Ownership and references

This provider owns the model contract. The fixed 1700 K, 7500 bar H2 connection,
its manual applicability assessments and its combined execution record belong
to the MCEG consumer example. This contract does not validate that complete
calculation or assign it a physical error bound.

Local `sources` paths are relative to the **ExoEOS repository root**, not this
directory, the YAML's directory or the caller's working directory. Current
implementation and tutorial references are local. Historical tests and the
original MCEG audit retain immutable commit URLs. External papers and unresolved
physical questions keep their previous scope and inspection status.

The implementation revision is
`d7681db4393fd85904846b083013230d81d45d4e`, the source snapshot audited before
this metadata-only addition. It is not the commit publishing this contract.
[The new execution receipt](../results/mceg/h2_tests_d7681db.json) and
[JUnit report](../results/mceg/h2_tests_d7681db.xml) record the actual one-test
run, environment and selected source hashes. The older run remains historical
evidence; the numerical claim now links to `provider_test_run`.

## Validation and review

From an environment with an independent MCEG checkout:

```bash
PYTHONPATH=/path/to/mceg/src python -m mceg \
  /path/to/exoeos/mceg/contracts/zhang_duan_h2.yaml
```

This checks structure only. It does not retrieve URLs, inspect referenced
passages, verify numerical claims or judge scientific applicability.

The ordinary provider test suite checks the selected-file hashes without
importing MCEG. Run that check and the scoped numerical test from this root:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q \
  tests/unittests/mceg_contract_test.py \
  tests/unittests/zhang_duan_test.py::test_published_species_factory_inserts_table_parameters
```

A documentation or contract commit may change HEAD while the selected audited
files remain identical. A selected-file change requires review; the hash check
does not infer which claims changed. It covers the recorded files, not every
transitive dependency. Review assumptions and source comparisons, rerun affected
tests and save a new receipt for the new audited revision. Preserve older
receipts; do not relabel a historical pass as a new execution. Current tests
without a run are `not_run`, and current numerical claims without support are
`unknown`. Physical validation remains a separate question.

The JSON receipt is a pilot artifact, not an extension to the v0.1 contract
schema. These files are checkout metadata; wheel-level contract discovery and
automatic scientific decisions are outside this pilot.

## Inventory coverage

The [machine-readable inventory](models.yaml) maps code and model choices to
contracts, including known uncontracted families. `mceg coverage .` checks these
links and reports new/changed or unreviewed source files without importing this
package. The current partial inventory intentionally returns exit code 1.
See the [inventory notes](model_inventory.md#machine-readable-inventory-and-coverage)
for baseline counts, mutation checks and the limits of implicit-model detection.

## Models and evidence HTML

Run `python -m mceg docs --config mceg/docs.yaml` to generate RST only, or
`./update_doc.sh` from the repository root to regenerate and build HTML.
See [the documentation build guide](../documents/mceg/build.rst) for installation,
input selection and output locations. Generated files are ignored by Git.
