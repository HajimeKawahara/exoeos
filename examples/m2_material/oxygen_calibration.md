# Measured oxygen reaction and explicit standard scenarios

The inherited metal O standard with the adopted Ma activities fails a low-pressure experimental reaction
check, even within the measured temperature interval. Replaying all 23
[Sakao and Sano (1959)](https://doi.org/10.2320/jinstmet1952.23.11_671)
observations gives the following log10 gas-ratio residuals:

| Calculation | RMSE | Maximum absolute residual |
| --- | ---: | ---: |
| Pinned Young gas + inherited O standard + adopted Ma | 0.929446 | 0.975860 |
| Sakao O standard + unchanged Ma activities | 0.007021 | 0.018167 |
| Sakao published standard and finite-O activity fit | 0.006296 | 0.019868 |
| Matoba O standard + unchanged Ma activities | 0.041405 | 0.054890 |
| Matoba published standard and finite-O activity fit | 0.021314 | 0.044796 |

The [actual common-gas replay](validation/20260927_oxygen_common_gas.json)
uses ExoGibbs `ab527ff0e23fd857bbafa934b27464fecb2a9ab6`, ExoEOS
`44b977847ea7ed1e16377907722277a3e75bc9d7` and the 35-species
`m1_expanded` gas standards. Its uncorrected RMSE is 0.928636 and maximum
absolute residual 0.974914; the corrected rows above are unchanged. The
[actual-standard receipt](validation/20260927_oxygen_common_gas_receipt.json)
records both gas pairs, the alloy standard and infinite-dilution coefficient.
No native liquid or source/planet equilibrium calculation is required for this
reference replay. A caller can supply its own standard callback and provenance
to `calibration_report`; the callback's O coefficient must match this Ma model.

No coefficients are fitted to these points or to a BSE state. The new data,
source hashes, pinned inherited thermochemistry and calculation are retained
in [steel_data](steel_data/oxygen_provenance.json) and the
[replay archive](validation/20260927_oxygen_standard.json). The last two rows
use the independent published fit of
[Matoba and Kuwana (1965)](https://doi.org/10.2355/tetsutohagane1955.51.2_163)
on Sakao's observations; Matoba's raw observations have not been digitized.
These residuals and inter-study differences are not confidence intervals.

## Standard conversion

For `O(metal) + H2(g) = H2O(g)`, the measured constant is
`K = (f_H2O/f_H2)/a_O`, with `a_O = f_O * wt%O` and `f_O -> 1` in dilute
pure liquid Fe. The same gas standard pressure cancels in the ratio.
Sakao gives `log10 K = 7040/T - 3.224`; Matoba gives
`log10 K = 7480/T - 3.421`, with T in K. Their finite-O expressions are
`log10 f_O = (-1750/T + 0.76) wt%O` and
`(-10130/T + 4.94) wt%O`, respectively.

For the caller's mole-fraction standard, the conversion is

```text
C = 100 M_O / M_Fe
mu_O,Henry/(RT) = mu_O,formal/(RT) + ln(gamma_O,infinity) - ln(C)
mu_O,Henry,target/(RT) = (mu_H2O,gas^0 - mu_H2,gas^0)/(RT) + ln(K)
offset_O = mu_O,Henry,target/(RT) - mu_O,Henry,adopted/(RT)
```

The explicit masses used for this reference conversion are Fe=55.845 and
O=15.999 g/mol. A caller must supply its actual O standard and its matching
infinite-dilution activity coefficient; applying the convention shift twice
is incorrect. Consistent elemental gauge shifts cancel. The archived replay
also verifies the finite-O calculation against the direct binary Ma formula.

Call `oxygen_standard_scenario(T_K, P_Pa, oxygen_standard_rt=...,
oxygen_lngamma_infinite_dilution=..., hydrogen_gas_standard_rt=...,
water_gas_standard_rt=..., reference="sakao1959")`. Potentials are reduced by
the same RT. The returned `standard_offsets_rt={"O_metal": offset}` can be
passed to a fixed-temperature source scenario; recompute it at a changed T.
Only a linear O term is added to extensive G and its derivative. No excess
Gibbs parameter, pressure law or archived baseline is replaced.

At 1873.15 K the inherited log10 K is -0.394484, against Sakao +0.534375 and
Matoba +0.572273. At 2173.15 K the **extrapolated** O shifts are respectively
+1.63183861647 RT and +1.64443621458 RT. For the actual 35-gas standards these become +1.63208171962 RT and
+1.64467931773 RT. Holding either value fixed at this T over all source
pressures is an explicit sensitivity assumption, with no claimed pressure
upper bound. This is a concrete constitutive-model
alternative for new conditional source/global solves; its effect on those
solves must be calculated separately.

## Applicability and remaining errors

Sakao Table 1 contains 1824.15, 1873.15 and 1924.15 K and 0.0118–0.1975 wt% O
in liquid Fe at one atmosphere. The table's corrected H2O/H2 ratios already
account for the authors' water-vapor correction. Preserve its 1551/1651 C
labels instead of substituting the rounded nominal temperatures in the text.
Matoba measured 1823.15, 1880.15 and 1936.15 K with dilute O; their treatment
assumes the dissolved-H effect on O activity is negligible. Neither paper
calibrates the target Fe-Si-O-H alloy at 2173.15 K and 60–776 bar.

Temperature extrapolation requires `allow_temperature_extrapolation=True`.
Total-pressure distance, temperature distance and unknown composition/pressure
transfer errors remain separate. The historical BSE states' approximately
0.37–0.58 wt% O also exceed this experiment's O range. The O standard repair
substantially reduces the measured reaction residual; it does not establish a
joint BSE material domain, quantify high-pressure nonideal-gas errors, or
resolve alloy/silicate phase stability. The report therefore records
`material_admission="adopted_oxygen_reaction_reference_failed"` for the original
model and leaves `accepted_coupled_material_domain=null`.

The inherited sign is intentional in
[Schlichting and Young (2022)](https://doi.org/10.3847/PSJ/ac68e6), Appendix
p18, reaction R4. The measured alternative is not described as a coding sign
fix. Joint treatment of the remaining H/water references and omitted transfer
paths remains required.

## Reproduction

```console
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
  python -m examples.m2_material.oxygen_calibration --output /tmp/new-oxygen-replay.json
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
  python -m pytest -q tests/unittests/m2_oxygen_calibration_test.py
```

Output creation is exclusive; an existing archive is never overwritten.
The pure conversion helper uses no sibling provider, native MELTS or JAX.
The replay uses the local Ma provider, requires JAX float64, and hashes it.
