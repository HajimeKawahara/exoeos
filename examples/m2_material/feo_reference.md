# High-temperature Fe-O reference and finite water scenarios

[Distin, Whiteway and Masson (1971)](https://doi.org/10.1179/cmq.1971.10.1.13)
measured the oxygen content of levitated liquid iron in equilibrium with
liquid iron oxide. This directly reaches the present temperature range;
it is not exclusively a low-temperature steelmaking extrapolation.

| Temperature (K) | Measured O (wt%) | Published saturation fit (wt%) | Frost two-species model (wt%) |
|---|---:|---:|---:|
| 2058.15 | 0.45 | 0.46252 | 0.48704 |
| 2153.15 | 0.62 | 0.63372 | 0.72065 |
| 2233.15 | 0.83 | 0.80919 | 0.98504 |

The abstract reports ±0.02 wt% without an individual covariance or confidence
level. Its fit is `log10[O wt%] = -6380/T + 2.765`, with the rounded stated
range 1800–2230 K; the last Celsius-converted observation is 3.15 K above
that rounded endpoint. At 2173.15 K the fit gives 0.67479 wt% O. This binary
boundary does not certify a multicomponent Fe-Si-O-H alloy.

[Frost et al. (2010)](https://doi.org/10.1029/2009JB006302), Eqs. 6–9,
provide one associated Fe/FeO Gibbs scalar with asymmetric Margules mixing.
The new helper solves equality of **both** component potentials between the
two liquids, retaining the visible discrepancy from the measurements above.
It does not refit those observations or declare the two-species approximation
exact. The basis conversion is `n_Fe_atoms = n_Fe_species + n_FeO` and
`n_O_atoms = n_FeO`, hence `mu_O_atomic = mu_FeO - mu_Fe`.

After aligning the infinite-dilution reference, the Frost and current Ma
composition-dependent oxygen potentials differ by 0.00454 RT for binary
iron with 0.0853056 wt% O at 2173.15 K and 269.655 bar. The corresponding
dilute atomic O-O coefficients are -6.0600 and -7.5927. This comparison
supports a limited binary composition sensitivity; it does not compare
absolute oxygen standards, constrain H/Si cross interactions, or establish
a whole-alloy stability domain. The reported Frost pressure slope alone
changes the dilute oxygen excess term by -0.000877 RT from 1 bar; pure Fe/FeO
pressure volumes are separate and are not bounded by that number.

[Replay and source conventions](feo_reference.py) and
[primary provenance](steel_data/feo_reference_sources.json) preserve these
distinctions. [Saved numerical comparison](validation/20260928_feo_reference/comparison.json)
contains the actual coexistence residuals and the unchanged null common-domain admission.

The separate [water capacity replay](water_capacity_scenarios.py) constructs
a finite `[1/2.265, 2.265]` capacity interval covering all 74 nonblank original
fit observations and 14 independent KFT central measurements. The limiting
factor is 2.264858 from `HT_1550 sample 1`. Covering the reported concentration
1-sigma intervals alone instead requires 2.548961; fugacity and shared
absorption errors have not been propagated. The
[manifest](validation/20260928_water_reconstruction/capacity_scenario_manifest.json)
retains every observation and ready-to-use `h2o_melts` standard scenarios
with shifts ±1.6351495 RT. These are finite model comparisons for robustness,
not statistical confidence intervals or universal BSE extrapolation bounds.
