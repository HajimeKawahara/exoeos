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
