# Na reference limits and exchange-standard replay

The official tables of [Steenstra et al. (2018)](https://doi.org/10.1038/s41598-018-25505-6)
provide numerical S-free Na censoring limits, in addition to their FeS exchange
fit. [The source record](sodium_reference_sources.json) preserves the original
table cells, selected analytical method, concentration bases and source hashes.
This improves the earlier Na evidence inventory without adding a Na species,
changing a source energy or claiming a low-pressure omission bound.

## Reported S-free limits

At 1 GPa, the following paired LA-ICP-MS observations give mass-partition
limits `D = Na mass fraction in metal / Na mass fraction in silicate`:

| Run | T (K) | Metal Si (wt%) | Calculated alloy xC | Metal Na (mass ppm) | Silicate Na (mass ppm, 2SE) | D upper at central silicate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GGK1 | 1683 | 0.01 | 0.19 | <29 | 974 (66) | 0.0297741 |
| GGK2 | 1783 | 0.02 | 0.20 | <35 | 1034 (84) | 0.0338491 |
| GGK7 | 1683 | 14.16 | 0.07 | <44 | 1768 (200) | 0.0248869 |
| GGK8 | 1783 | 15.31 | 0.08 | <53 | 2882 (884) | 0.0183900 |
| GGK9 | 1883 | 11.76 | 0.11 | <30 | 2053 (104) | 0.0146128 |

All capsules were graphite-saturated. Table S4 calculates `xC` using the cited
metal-activity model; it is not a measured C fraction. The separate C mass
estimate inferred from EPMA totals is also retained and is not substituted.
The paper does not specify the noise multiplier or confidence definition of
the `<` values in the retained text. They are censoring thresholds, not new
probabilistic confidence bounds. A second diagnostic divides by the reported
silicate central value minus its 2SE; it does not assign a new joint confidence
level to the result.

GGK3 is preserved separately: Table S4 lists `48(17)` mass ppm, while main
Table1 calls Na below detection. It is neither promoted to a detected fit point
nor assigned an invented `<` threshold. Gessmann and Wood (2002)'s unquantified
Na detection limit remains a distinct older reference; it does not describe
these newly retained 2018 thresholds.

## Independent exchange-constant replay

The printed reaction and Equation4 use one-Na `NaO0.5`:

```text
0.5 Fe(m) + NaO0.5(sil) = Na(m) + 0.5 FeO(sil)
K_D = xNa * sqrt(xFeO) / (xNaO0.5 * sqrt(xFe))
K = K_D * gammaNa / sqrt(gammaFe) * sqrt(gammaFeO) / gammaNaO0.5
```

The diagnostic reconstructs major compositions from EPMA, replaces Na/K with
LA-ICP-MS as the authors specify, and uses EPMA for GGK6 because its laser
measurements encountered silicate inclusions. It uses the reported calculated
carbon mole fraction to close the metal denominator. Unquantified minor
denominator entries are omitted, not asserted to be experimentally zero.
The printed, rounded TableS2 activities are used directly. No free coefficient
is fitted to reduce a residual.

| FeS run | Published log10 KNa (2SE) | One-cation oxide replay | Oxide-molecule replay |
| --- | ---: | ---: | ---: |
| GGK4 | -3.12 (0.09) | -3.427611 | -3.154769 |
| GGK5b | -2.51 (0.15) | -2.773631 | -2.494585 |
| GGK6 | -2.22 (0.18) | -2.500931 | -2.220736 |

The oxide-molecule convention counts Na2O formula moles in the denominator
term and normalizes all oxides as formula units. It reproduces the three
published values within their reported errors. The literal one-cation basis
counts two Na per Na2O and all other oxide cations consistently; it differs by
0.264–0.308 dex. Both outputs remain visible. The numerical match is evidence
about the published concentration recipe, not proof of its hypothetical oxide
standard or permission to mix the two conventions. Table rounding and the
declared host-denominator reconstruction also contribute to the residuals.

## Henry rebasing and the next physical step

For the same activity convention used to construct `K`, an infinite-dilution
reference can be removed algebraically:

```text
ln K_H = ln K - ln gammaNa_infinite
muNa_H/(RT) = muNaO0.5_standard/(RT) + 0.5 muFe_standard/(RT)
              - 0.5 muFeO_standard/(RT) - ln K_H
```

`henry_standard_rt` performs only this balanced conversion on explicitly
supplied, consistent one-Na standards. The arbitrary infinite-dilution activity
normalization cancels when both inputs are rebased together. It is not by
itself a reason to stop reconstruction. A JANAF atomic Na standard, however,
cannot be substituted for the unknown oxide reference without the matching
standard transformation. The helper does not select a native/MELTS anchor or
silently fit one to a single BSE contact point.

Remaining physical assumptions include the authors' ideal NaO0.5 activity,
Na-S taken from K-S, unmeasured Na-Si and C effects, and the pressure/temperature
continuation from the experimental hosts. No measured zero is assigned to
these interactions. The FeS `5.44 - 14340/T` regression and its coefficient
errors do not bound the S-free 2173.15 K, approximately 270 bar extrapolation.
The five limits can constrain declared continuations and finite-inventory
tests, but their direct transplantation is not an empirical BSE error bound.
The separate [fixed-host Na box](sodium_fixed_box.md) declares constant D
explicitly and evaluates the finite 1%-of-global-Na engineering threshold.

The [saved replay](validation/20260928_sodium_exchange/replay.json) records
all nine basalt cases, including the excluded GGK3 observation. The
[runner](validation/20260928_sodium_exchange/replay.py) and
[execution receipt](validation/20260928_sodium_exchange/execution.json)
need neither a native provider nor an equilibrium solve.
