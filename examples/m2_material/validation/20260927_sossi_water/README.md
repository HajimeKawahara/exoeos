# Complete-table peridotite water reference

This is a new, native-free reconstruction of the Sossi et al. (2023)
Table 1 data and Eq.11 reference. It changes no MELTS/alloy standard or
archived equilibrium. The implementation commit is
`dc8b6f32b3eb37ec950d5341215e5987149f545e`, based on ExoEOS main
`4bee4e1`.

The primary table contains fourteen samples. Eleven enter the authors'
Eq.11 selection; Per-1, Per-2 and Per-5 are excluded from that fit. Per-5
still contributes its shared background and uncertainty to every selected
measurement. The later twelve-row author CSV did not include Per-4 or
Per-5. Both datasets are preserved separately.

| Quantity | IR epsilon 6.3 | IR epsilon 5.1 |
| --- | ---: | ---: |
| Reconstructed H2O coefficient, ppm / sqrt(bar) | 526.411312 | 649.955945 |
| Reconstructed H2 coefficient, ppm / sqrt(bar) | 182.991425 | 226.290858 |
| Fitting RMSE, ppm | 2.754296 | 3.409492 |
| Group-out RMSE, ppm | 4.234195 | 5.237440 |
| Maximum absolute group-out residual, ppm | 9.221129 | 11.396423 |
| H2O coefficient SD upper bound, ppm / sqrt(bar) | 98.746278 | 143.937774 |
| H2 coefficient SD upper bound, ppm / sqrt(bar) | 66.830119 | 98.763323 |

There are nine distinct fitted gas-fugacity pairs. The three time-series
replicates always leave the training set together. Cross-validation is
conditional on the selected Per-5 background and spectroscopy calibration;
it does not provide independent external validation or a future error
bound. The IR branches observe the same samples and are not independent.

The SD bounds cover any covariance consistent with the reported raw
concentration marginal SDs, after applying the shared-background operator.
They do not cover predictor uncertainty, background-selection bias,
composition/temperature/pressure transfer or missing physics. The saved
covariance under independent raw errors is explicitly conditional on an
unverified assumption. Raw and corrected columns differ by up to 0.1 ppm
through published rounding; the report retains their coefficient offset.

The reference applies to the reported peridotite-glass experiment at
nominal 2173 K and 1 bar, with a flowing Ar-CO2-H2/O2 gas mixture.
Geometric support in its joint square-root-fugacity predictors is not
physical acceptance between samples. The actual-state API explicitly
labels changed T, P, host and unsupported joint predictors. Current BSE
closures at hundreds of bar are extrapolations. A common supported
silicate-H2/Fe-Si-O-H material domain remains unestablished.

`receipt.json` hashes all six executed evidence/source files and the
fresh report/logs. The full native-free unit suite passed 463 tests.
After a provenance-only report-field addition, all thirteen new tests
were rerun and passed. HTML built without warnings or errors; rendered
sources did not change afterwards. The table transcription was visually
checked against the primary PDF's printed page 3. The PDF is linked and
hashed in the [source provenance](../../water_data/sossi2023_provenance.json)
and is not redistributed here.

To reproduce the numerical report, check out the implementation commit
and write to a new path:

```console
PYTHONPATH=src python -m examples.m2_material.sossi_water_calibration --output /tmp/new-sossi-reference.json
```

The command requires NumPy and the source checkout, without alphaMELTS,
a network download or sibling packages. Source/data hashes remain stable;
floating-point reconstruction can differ at roundoff across NumPy builds.
