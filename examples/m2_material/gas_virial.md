# Major-gas pressure diagnostic

`gas_virial.py` replays five primary-source pair correlations for H2, He, and
H2O using the existing `SecondVirialEOS` and Helmholtz derivative provider.
It reports missing interactions explicitly. It does not change the source
equilibrium, archived closure, or material admission.

## Source and convention checks

All coefficients below are density virials in cm3/mol. The implementation
converts them to m3/mol before constructing the EOS. Correlation coefficients,
source PDF hashes, domains, and uncertainty qualifications are recorded in
[gas_virial_sources.json](gas_virial_sources.json).

| Pair | Source | B at 2173.15 K (cm3/mol) | Qualification |
| --- | --- | ---: | --- |
| H2-H2 | [Hodges et al. 2004](https://doi.org/10.1063/1.1630960), Eq. 24/Table IV | 13.822665 | High-temperature potential-based correlation |
| He-He | [Hurly and Mehl 2007](https://doi.org/10.6028/jres.112.006), Table 5 | 7.764906 | Cubic Hermite interpolation of B and its published derivative |
| H2O-H2O | [Harvey and Lemmon 2004](https://doi.org/10.1063/1.1587731), Eq. 6/Table 4 | 8.410740 | Potential-guided extension beyond 1170 K measurements |
| H2-H2O | Hodges et al. 2004, Table VI | 15.101916 | Extrapolated beyond the recommended 2000 K maximum |
| He-H2O | [Hodges et al. 2002](https://doi.org/10.1063/1.1421065), Eq. 15/Table VII | 14.912020 | Extrapolated beyond the recommended 2000 K maximum |
| H2-He | [Garberoglio et al. 2014](https://doi.org/10.1007/s10765-014-1729-7) | Missing | Abstract located; numerical table not transcribed |

The H2-H2O coefficient `c3` is **+285.42**, as verified from the printed
Table VI. PDF text extraction alone can corrupt this entry. Tests replay
independent printed water and water-hydrogen values at four temperatures.

The older pure-H2 correlation gives 15.97882 and 14.08944 cm3/mol at 1000
and 2000 K. The independent [Garberoglio et al. 2012](https://doi.org/10.1063/1.4757565)
Table II gives 15.96 and 14.22, with expanded uncertainties 0.05 and 0.03.
The 2000 K discrepancy exceeds that uncertainty. Thus the old fit is useful
for a pressure sensitivity calculation, but does not inherit the newer
potential's uncertainty.

## Saved OH basal state

[gas_virial_oh_diagnostic.json](gas_virial_oh_diagnostic.json) binds the saved
OH closure by SHA-256, together with this implementation and the source
manifest. Its state is 2173.15 K and 269.65451525 bar. H2, He, and H2O account
for 0.9992020293 of the original gas; the remaining fraction is retained in
the report before conditioning the three-component composition.

For this conditioned gas, the second virial is

```text
B_mix = 10.17092314 + 0.26068602 * B_H2,He   [cm3/mol].
```

Without an explicit H2-He input, no mixture state is returned. The archived
0, 10, and 20 cm3/mol inputs are **sensitivity choices, not empirical limits**:

| Supplied H2-He B (cm3/mol) | Z | ln(phi_H2) | ln(phi_He) | ln(phi_H2O) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.01495537 | 0.01912558 | -0.00814167 | 0.02820899 |
| 10 | 1.01871910 | 0.02028723 | 0.01056225 | 0.02434852 |
| 20 | 1.02245571 | 0.02145440 | 0.02914275 | 0.02053052 |

These are fixed-composition, truncated-density-virial calculations. They
quantify percent-scale departures in these scenarios; they are not bounds
on the 76-species atmosphere or its reclosed pressure. Unknown trace pairs
occupy 0.0015953046 of the original pair weight. A small mole fraction alone
does not bound a trace species' chemical potential or phase response.
Higher density virials and the two temperature extrapolations also need
separate assessment.

## Independent He-H2 equation replay

[Beckmueller et al. 2024](https://doi.org/10.1016/j.cryogenics.2024.103817)
provide a complete binary Helmholtz EOS and machine-readable supplementary
coefficients. [h2_he_reference_sources.json](h2_he_reference_sources.json)
preserves the He-H2 entries and the original PDF, ZIP, and extracted-file
hashes. The optional [replay_h2_he_reference.py](replay_h2_he_reference.py)
uses `teqp==0.23.1` to evaluate those coefficients with its pinned normal-H2
and helium pure-fluid equations. It does not replace the five pair laws above.

The [preserved replay](h2_he_oh_reference_replay.json) retains a discrepancy
in the first Table 6 pressure check, even with the author's original JSON:

| T (K) | Density (mol/m3) | Calculated / printed pressure - 1 |
| ---: | ---: | ---: |
| 25 | 1000 | -4.86139e-3 |
| 35 | 40000 | +3.91650e-8 |
| 75 | 10000 | -1.94225e-8 |
| 100 | 60000 | -4.06438e-8 |

Thus `all_pressure_reference_checks_passed` is false at the paper's stated
1e-6 implementation-comparison scale. The source of the low-temperature
discrepancy remains unresolved; no coefficient was adjusted to match it.

At the OH source T/P, conditioning on He/H2 alone gives Z=1.01969530.
The second density virial at that same density gives Z=1.01940858. These
are binary-model calculations, not the wet 76-species atmosphere. The
extracted apparent cross virial at 2173.15 K varies from 11.2612 to
11.5594 cm3/mol over the sampled helium fractions 0.1--0.9, due to the
composition-dependent reducing function. It cannot silently fill the
missing unique pair coefficient in `SecondVirialEOS`. This temperature also
exceeds the model's 2000 K cross-virial reference data. The replay supplies
additional model evidence, without a full-gas or empirical error bound.

## Reproduction

Use the unchanged OH closure archived in ExoInventory PR #35:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
python examples/m2_material/gas_virial.py \
  --closure /path/to/closure.json \
  --output /tmp/gas_virial_oh_diagnostic.json \
  --h2-he-cm3-mol 0 10 20

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
pytest -q tests/unittests/m2_gas_virial_test.py \
  tests/unittests/second_virial_test.py
```

The combined targeted checks passed: 27 tests. The diagnostic command exited
successfully. Full coupled equilibrium was not rerun for this diagnostic.

The independent binary replay uses an optional reference environment; teqp
is not a package dependency:

```sh
python examples/m2_material/replay_h2_he_reference.py \
  --closure /path/to/closure.json --output /tmp/h2_he_reference.json
```
