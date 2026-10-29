# Experimental MCEG contracts

[MCEG](https://github.com/HajimeKawahara/mceg) records model assumptions,
applicability, claims and scoped evidence. The
[pure-H2 Zhang-Duan](contracts/zhang_duan_h2.yaml) contract now uses v0.2,
with a lossless import of its original v0.1 declarations. MCEG is not an ExoEOS
runtime or test dependency; the model implementation is unchanged.

The [2026-10-20 model inventory](model_inventory.md) adds v0.2 contracts for the
solubility/magma survey, with separate numerical and physical Assessments and
fresh receipts. The new records use the companion MCEG volatile-quantity
extension. Existing H2 and source-study histories retain their original scope.

The [Ma Fe-Si-O contract](contracts/ma_fe_si_o.yaml) adds a 2026-10-21 source
review of Young's printed equations, the Vogel/Buchan reproductions and pinned
author/GCE code. The printed/code oxygen coefficient difference is recorded;
original Ma and Badro supplementary data were pending in that review. Inspect
the new assessment separately from the retained survey history:

```bash
mceg explain mceg/contracts/ma_fe_si_o.yaml --assessment physical_review_20261021 --root .
```

Source inspection does not establish physical accuracy. Remote artifact hashes
identify downloaded bytes; offline MCEG checks do not verify those URLs.

## Source review, 2026-10-29

The [volatile-liquid contract](contracts/magma_volatiles.yaml) now records
inspection of the supplied Hirschmann (2012) PDF and freshly retrieved Thompson
(2025), Sossi (2023), and Guillot-Sator (2012) papers. Hirschmann's Fig. 4 directly
supplies the adopted H2 constants. The water coefficients and all nine adopted
He table entries also match their original sources at the recorded scope.

Hirschmann uses oxide-component mole fractions including water; the provider
counts dry MELTS endmembers plus water. The new comparison retains that
discrepancy. Water's conversion from the printed OH response to H2O-equivalent
mass is an explicit dimensional interpretation. Physical accuracy remains
`unknown`; the historical numerical receipts retain their original scope.

The [initial review record](../results/mceg/20261029_source_review/review.json) lists
inspection locations, hashes and retrieval outcomes before the Ma PDF was supplied.
At that point Hirschmann's supplementary compositions and a fresh Thompson
notebook download were unavailable. Those historical outcomes are preserved;
the follow-ups below record their subsequent recovery, including the later
supplied Badro (2015) supplement.

```bash
mceg explain mceg/contracts/magma_volatiles.yaml --assessment physical_review_20261029 --root .
python -m mceg --trace coefficient_comparison_20261029 mceg/contracts/magma_volatiles.yaml
mceg explain mceg/contracts/ma_fe_si_o.yaml --assessment physical_review_20261029 --root .
```

The [supplied-paper review](../results/mceg/20261029_ma_primary_review/review.json)
completes `inspect_ma_original`. Ma Eqs. 21-26, Tables I-II and the nomenclature
agree with the implemented scalar and completed activities after the explicit
standard-state shift. The original confirms the selected minus reciprocal sign;
the conflicting sign in Vogel A2 remains recorded. The seven-state source-algebra
comparison, reference regeneration and 39 provider tests passed.

The first supplied Badro PDF was the five-page main paper, with no SI Appendix
or embedded files. Its Methods points to supplementary Table S1. The later SI
inspection is recorded below; this earlier assessment retains its historical
scope:

```bash
mceg explain mceg/contracts/ma_fe_si_o.yaml --assessment physical_primary_review_20261029 --root .
python -m mceg --trace implementation_ma_primary_20261029 mceg/contracts/ma_fe_si_o.yaml
```

## Follow-up source review, 2026-10-29

The [follow-up record](../results/mceg/20261029_remaining_source_review/review.json)
adds four source inspections. The supplied Hirschmann DOCX contains the starting
compositions and six selected glass analyses. Five run identifiers match the
main paper; the supplement's `B808` is not silently substituted for `A808`.
An explicit oxide/water/H2 reconstruction gives ln K differences from -0.091864
to +0.018066. The composition-recovery task is closed at this scope, with exact
run reconstruction and the provider's endmember/oxide basis difference left open.

The freshly retrieved Thompson notebook and combined CSV match the historical
hashes. Cell-level checks confirm the response scaling, 74-of-81 fit selection,
descaling and paper-rounded coefficients. The twelve pooled Sossi rows preserve
their supplied oxide fractions and one OH/H2O molar-mass conversion. This closes
fresh notebook inspection without resolving physical species normalization.

Jiang (2025) full text and its 22-row Table S1 are now inspected: these are
DFT/ML results at 135 GPa and 5000 K, with partition coefficients on an atomic
concentration basis. They do not calibrate Ma H interactions at low pressure.
Nozaki (1966) Section 4.4 and Table 6 confirm e_H_P=0.015 for P<6 wt%, with a
1427-1610 degC scope; the historical abstract-only ledger used the whole-paper
temperature range. The dated record preserves this distinction.

Seven requested sources or datasets were unavailable at that stage. Their titles,
bibliography and access outcomes remain in the historical review and its
[BibTeX list](../results/mceg/20261029_remaining_source_review/missing_sources.bib).
The later Badro/Garberoglio review below reduces this list to five.
Jiang's published Table S1 is available, but its raw simulation data require an
author request. The withdrawn PDF mislabeled as Hirschmann supplies no evidence.

Original documents and author files are retained locally under `refs/`, excluded
through this checkout's `.git/info/exclude`. They are not repository artifacts.
The contracts retain URI hashes and a verifiable review record. Runtime code,
coefficients, historical ledgers and test receipts are unchanged. Physical
accuracy remains `unknown` in the new scoped assessments:

```bash
mceg explain mceg/contracts/magma_volatiles.yaml --assessment physical_remaining_review_20261029 --root .
mceg explain mceg/contracts/ma_fe_si_o_h.yaml --assessment physical_source_review_20261029 --root .
```

## Badro and Garberoglio source review, 2026-10-29

The [supplied-source review](../results/mceg/20261029_badro_garberoglio_review/review.json)
records Badro's 12-page SI and Garberoglio's 15-page paper. Badro Table S1 (p.3)
and its temperature scaling give epsilon_OO=-1873/T. The separate
ln(gamma_O^0)=4.29-16500/T term is unchanged. This supports the oxygen self term
in the pinned author/GCE code, whereas the current default follows Young's
printed -16500/T interaction. The original-source question is closed; a separate
follow-up retains the implementation-version decision. The existing explicit
Badro parameter option agrees with SI Eqs.3-4 at seven numerical fixture states
after standard conversion. These states are not calibration observations.

Garberoglio Table 1 supplies 67 He-H2 cross second virial coefficients at
8-2000 K and expanded uncertainties with coverage factor k=2. For example,
B12(2000 K)=11.69 cm3/mol with U=0.01 cm3/mol. The local CSV preserves the
fully flexible and both rigid-model columns separately. The saved 2173.15 K
gas diagnostic is beyond this table; no interpolation or extrapolation is added.
The [gas diagnostic guide](../examples/m2_material/gas_virial.md) links this review.

The [missing-source bibliography at that stage](../results/mceg/20261029_badro_garberoglio_review/missing_sources.bib)
contains five items: Ghiorso-Gualda calibration observations,
Zhang-Duan's 111 H2 observations, and the Schenck, Distin and Jambon full texts.
Original files and complete table transcriptions remain local in `refs/`.
Earlier ledgers, receipts and bibliographies retain their historical scope.
Physical accuracy remains `unknown`; inspect the new source comparison with:

```bash
mceg explain mceg/contracts/ma_fe_si_o.yaml --assessment physical_badro_review_20261029 --root .
python -m mceg --trace badro_parameter_comparison_20261029 mceg/contracts/ma_fe_si_o.yaml
```

## Jambon source review, 2026-10-29

The [supplied-scan review](../results/mceg/20261029_jambon_review/review.json)
completes inspection of Jambon (1986), printed pp.401-408. The image-only PDF
was read visually, including experimental controls, composition and gas tables.
Table 5 gives seven He summaries at 1300-1500 degC, representing 16 analyses.
Its 1400 degC mean is `(56.5 +/- 11) * 1e-5 cm3 STP/(g bar)`; the body rounds
the central value to 56, confirming the historical abstract-only record.
The quoted uncertainty is one-sigma regression scatter. The table does not
supply all individual analyses or their weights for an exact regression replay.

Guillot-Sator explicitly calibrated its potential using Jambon and Lux MORB
data. Its Appendix C He/MORB value is 5.49% above the Jambon table mean at
nominally 1400 degC, but this agreement is not independent validation. The
[He guide](../examples/m2_material/na_helium_transfer.md) records this ancestry.
The experimental He interval and partial-pressure basis do not validate the
2173.15 K wet-BSE continuation. The provider's coefficients remain unchanged.

The [current missing-source bibliography](../results/mceg/20261029_jambon_review/missing_sources.bib)
has four items: Ghiorso-Gualda calibration observations, Zhang-Duan's 111 H2
observations, and the Schenck and Distin full texts. The earlier five- and
seven-item lists retain their historical scope. Original scans and complete
table transcriptions remain local under `refs/`, excluded from Git.

```bash
mceg explain mceg/contracts/magma_volatiles.yaml --assessment physical_jambon_review_20261029 --root .
python -m mceg --trace jambon_helium_comparison_20261029 mceg/contracts/magma_volatiles.yaml
```

## Ownership and references

This provider owns the model contract. The fixed 1700 K, 7500 bar H2 connection,
its manual applicability assessments and its combined execution record belong
to the MCEG consumer example. This contract does not validate that complete
calculation or assign it a physical error bound.

Local `sources` paths are relative to the **ExoEOS repository root**, not this
directory, the YAML's directory or the caller's working directory. Current
implementation and tutorial references are local. Historical tests and the
original MCEG audit retain immutable commit URLs. Historical source-review
statuses are preserved; later inspections have separate dated records.

The implementation revision is
`d7681db4393fd85904846b083013230d81d45d4e`, the source snapshot audited before
this metadata-only addition. It is not the commit publishing this contract.
[The preserved execution receipt](../results/mceg/h2_tests_d7681db.json) and
[JUnit report](../results/mceg/h2_tests_d7681db.xml) record the actual one-test
run, environment and selected source hashes. The older run remains historical
evidence; the imported numerical declaration retains `provider_test_run`.

## H2 source review, 2026-10-21

The retrieved Zhang-Duan (2009) PDF now supplies inspected Eq. 8 and Tables
3-6, with its URL and SHA-256 recorded in `zd2009_20261021`. The H2 parameters
match Table 4. Table 5 reports 111 points at 250-423 K and 2-700 MPa; its
1.43% average deviation concerns volume, not fugacity. The two new range
conditions are marginal bounds, not a joint coverage rectangle.

`physical_review_20261021` remains `unknown`: individual H2 observations were
not recovered, and the documented physical-Z interpretation differs from
the printed reduced-variable left side of Eq. 8. Water volume controls do
not resolve pure-H2 physical accuracy. Separate follow-ups retain these gaps.

```bash
mceg explain mceg/contracts/zhang_duan_h2.yaml \
  --assessment physical_review_20261021 --root .
mceg explain mceg/contracts/zhang_duan_h2.yaml \
  --assessment numerical_review_20261021 --root .
```

The numerical review verifies the preserved receipt and selected-file hashes;
it does not relabel that run as new. MCEG does not fetch URI artifacts. The
consumer's frozen `provider_h2` example requires a separate manual review
before adopting this changed scientific record; its earlier applicability
decision is not automatically transferred.

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
