# Experimental MCEG contracts

[MCEG](https://github.com/HajimeKawahara/mceg) records model assumptions,
applicability, claims and scoped evidence. This checkout pilots its frozen
`schema_version: "0.1"` with
[pure-H2 Zhang-Duan](contracts/zhang_duan_h2.yaml). MCEG is not an ExoEOS
runtime or test dependency; the model implementation is unchanged.

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
