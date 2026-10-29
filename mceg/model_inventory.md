# Model inventory: solubility and magma priority

Survey date: 2026-10-20. Audited implementation:
`37b9173f37866495949fd7c90340a02b7ceaa78d`.
This inventory distinguishes registered model families from families located
for later source-specific review. It is not a claim of complete scientific
coverage of the package.

The 2026-10-29 [provisional Badro default](contracts/ma_fe_si_o_badro.yaml)
adds a separate identity and fresh numerical evidence for the dry Ma model and
its ideal-H wrapper. The survey contracts below preserve the former printed-Young
parameterization. The current default uses `epsilon_OO=-1873/T`; its separate
infinite-dilution term still uses `4.29-16500/T`. Author clarification remains open.

## Contracts added in this survey

| Contract | Identity and main boundary | Numerical checks selected |
| --- | --- | --- |
| [Ma Fe-Si-O](contracts/ma_fe_si_o.yaml) | Printed default coefficients, completed Fe activity, formal symmetric standards; no pressure response | `ma_fe_si_o_test.py` |
| [Ma Fe-Si-O-H](contracts/ma_fe_si_o_h.yaml) | Atomic H ideal dilution; no H interaction fit or absolute H standard | `ma_fe_si_o_h_test.py` |
| [Mass-fraction solute](contracts/mass_fraction_solute.yaml) | Exact total-liquid mass fraction and host response; needs consistent supplied host G/mu | `mass_fraction_solute_test.py` |
| [Dry MELTS liquid](contracts/magma_dry_liquid.yaml) | Full extensive G, declared endmember amounts, dry11 and its supported faces | `magma_dry11_test.py` |
| [Mode-1 hydrous liquid](contracts/magma_hydrous_liquid.yaml) | Native H2O standard, eleven interactions and extra water entropy | `magma_hydrous_test.py` |
| [Mode-4 carbonate liquid](contracts/magma_carbonate_liquid.yaml) | Internal CaCO3 reaction; zero-carbon water model differs from mode 1 | `magma_carbonate_test.py` |
| [Conditional volatile liquid](contracts/magma_volatiles.yaml) | Explicit water model, H2 dilution, selected dry-host He capacity and gas standards | `conditional_water_test.py`, `two_power_water_test.py` |
| [Magma-reference fluid](contracts/magma_fluid.yaml) | Native gas standards plus selected pure-H2O or binary H2O/CO2 EOS | `magma_fluid_test.py`, `magma_mixed_fluid_test.py` |

Physical accuracy remains unknown in these records. Tests compare formulas,
derivatives and stored numerical references; they do not create new empirical
calibration. In particular, a generic two-power isotherm with synthetic capacities
does not validate the coupled Sun-Yao/ExoEOS model.

## Other located model families

| Family | Current coverage and next boundary to review |
| --- | --- |
| Pure-H2 Zhang-Duan | Existing [provider contract](contracts/zhang_duan_h2.yaml); its historical receipt is preserved |
| IdealEOS, SecondVirialEOS, PengRobinsonEOS | Baseline contracts exist in MCEG's consumer repository; provider publication requires a current audit and receipt |
| Other Zhang-Duan species/mixtures | `zhang_duan.py`; H2 evidence cannot establish water/CO2 or mixed-fluid accuracy |
| IdealGas, total Helmholtz derivatives, density composites | `ideal_gas.py`, `thermodynamics.py`, `density.py`; caloric closure, ordered masses and additive-volume assumptions need separate identities |
| Chabrier-Debras, Marcum silicate-H, HelmholtzTable | Table loaders and interpolation models; dataset variants, missing cells and potential reconstruction need separate evidence |
| Native dry/hydrous solids | `magma_solids/` and the solid functions in `magma.py`; each phase's amount basis, ordering and site constraints need their own audit |
| Associated metal, phosphorus association/MQM/phosphides | `associated_metal.py`, `phosphorus_*.py`; composition maps, temperature policies and reaction-standard alignment need separate assessments |
| MELTS runtime and M2 checkout examples | `examples/`; distinguish saved binary references, conditional models and equilibrium consumers from packaged scalar providers |

## Connection conventions found during review

ExoGibbs supplies empirical concentrations or capacities. ExoEOS supplies an
extensive thermodynamic continuation only for explicitly selected constructions.
A connection must state the gas driving quantity, the concentration denominator,
the component order and standards. A successful individual contract does not
supply a connection's physical validation.

`water_model` and `helium_host` must be recorded even when using the same Python
entry point. The current `magma_volatiles.MODEL_ID` alone does not distinguish
all water alternatives. Do not freeze composition-dependent Sun-Yao capacities
before differentiating the liquid scalar. Keep H2's extra host dilution, He's
dry-host mass, and the binary water mass basis distinct.

## Reading and maintaining these records

The contracts use v0.2 with separate numerical and physical Assessments. The
shared [receipt](../results/mceg/20261020_model_survey/receipt.json),
[JUnit report](../results/mceg/20261020_model_survey/tests.xml) and
[execution log](../results/mceg/20261020_model_survey/tests.log) record fresh
CPU/x64 runs against explicit local source paths. Each contract pins its selected
source/test/fixture files and the shared artifacts. The receipt's commands and
environment describe how to repeat each module; use a new output directory.
No external reference generator, proprietary runtime, or experiment was rerun.
The selected manifest is not a complete transitive dependency lock.

Use the companion MCEG checkout containing the volatile-solubility quantity
extension (documented under Unreleased); older v0.2 readers do not recognize
all new pressure names. From this provider root with that environment active:

```bash
mceg mceg/contracts/*.yaml
mceg check . --assessment numerical_review --root .
mceg explain . --assessment physical_review
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/unittests/mceg_model_survey_test.py
```

The common numerical selector applies only to these new contracts. A numerical
PASS reports the archived assertions and matching selected files. The physical
selector intentionally remains UNKNOWN. Default `mceg check .` also shows source
review gaps and the existing historical contracts.

Local paths resolve from the provider root. A publication-only commit need not
match the audited source commit, but the selected hashes must match. After a
selected file changes, review the affected scientific declarations, run the
relevant tests, save new artifacts and add a new Subject/Context/Assessment.
Preserve earlier records; do not just replace hashes or rename an old pass.
No MCEG import or runtime dependency is added to the provider.

For a repeatable authoring procedure, see `docs/en/provider_onboarding.md` in
the companion MCEG repository. This survey's pattern is: define one model and
its conventions, inspect code and source scope, execute targeted tests, record
pins, write separate Claims/Assessments, then verify structure and local files.

## Machine-readable inventory and coverage

[models.yaml](models.yaml) is the machine-readable boundary inventory. Model
entries have code bindings, explicit choices, and zero or more contract links.
File classifications and SHA-256 pins record the reviewed inventory snapshot;
they do not renew numerical or physical evidence. Run from this provider root
with the companion MCEG environment active:

```bash
mceg coverage .
mceg coverage . --json
```

The 2026-10-20 baseline has **45 inventory entries**: 17 registered,
28 uncontracted, and 0 invalid contract mappings. These entries link all
9 provider contracts; counts differ because a family contract may serve multiple
choices. Of 50 scanned files, 43 match reviewed classifications and 7
remain unreviewed. Exit code **1** is expected. Remaining families and files are
work items, not evidence of physical failure; this is not complete model coverage.

See the [baseline report](../results/mceg/20261020_coverage/baseline.json) and
[mutation receipt](../results/mceg/20261020_coverage/mutations.json). Eight mutations
on a temporary copy produced the expected new diagnostics: a missing model link,
a missing contract, a wrong contract ID, a new formula branch, a renamed function,
a new source file, changed coefficient data, and an unlisted contract. A ninth
case records the detection limit: deleting a whole variant row while other rows
still map its file and family contract introduces no diagnostic. Initial model
classification and inventory diffs therefore require human review.

Source scans include all packaged files, not just public functions or contract
manifests. Known uncontracted families remain explicit; unfamiliar files remain
unreviewed. Review any code/data change before refreshing the inventory digest,
and update affected contracts/evidence separately. The companion MCEG guide
`docs/en/coverage.md` specifies the format, diagnostics and remaining limits.
