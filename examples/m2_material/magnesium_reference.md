# Measured Mg deoxidation reference

`magnesium_reference.py` reconstructs an omitted alloy-transfer standard
from [Itoh, Hino and Ban-ya (1997)](https://doi.org/10.2355/tetsutohagane1955.83.10_623).
The 37 Table 2 observations and all published second-order coefficients are
retained with primary-source hashes. No BSE data are fitted and no new
five-component alloy Gibbs model is installed.

For `MgO(cr) = Mg(metal) + O(metal)`, both solute activities use Henry mass
percent in dilute pure liquid Fe, and crystalline MgO has unit activity:

```text
log10 K = -4.28 - 4700/T
mu0_Mg,Henry/(RT) = mu0_MgO,cr/(RT) - mu0_O,Henry/(RT) - ln(K)
mu0_Mg,infinite-mole-fraction/(RT) = mu0_Mg,Henry/(RT) + ln(100 M_Mg/M_Fe)
```

Supply the O and MgO standards on the same RT and elemental gauge to
`magnesium_henry_standard`. The O input can be the target Henry standard
from the [measured oxygen scenario](oxygen_calibration.md). The optional
building block `mgo_crystal_standard_rt` reconstructs 1-bar MgO(cr) from the
[NIST-JANAF table](https://janaf.nist.gov/tables/Mg-008.txt), using fixed
298.15-K elemental enthalpy zeros, absolute entropy and enthalpy increments.
It does not use the formation-G column, whose elemental phases change with T.
Cubic Hermite interpolation of G and endpoint -S is restricted to 1700–2400 K.
The caller must establish the common gauge before mixing this standard with
other thermochemical sources.

The original finite-solute reference is replayed separately:

```text
log10 f_Mg = (958 - 2.59e6/T) wO + (-1.90e6 + 4.22e9/T) wO^2
             + (2.14e5 - 5.16e8/T) wMg wO
log10 f_O = (-1750/T + 0.76) wO + (630 - 1.71e6/T) wMg
            + (70500 - 1.70e8/T) wMg^2 + (-2.51e6 + 5.57e9/T) wMg wO
```

Here `wMg` and `wO` are numerical mass percentages, not mass fractions or
ppm. The O self coefficient follows the Sakao/JSPS convention used by Itoh.
Only the printed rounded coefficients are replayed; no residual is removed
by fitting or excluding a table row.

| Temperature | Observations | RMSE of log10(activity product / K) | Maximum absolute residual |
| --- | ---: | ---: | ---: |
| 1873 K | 28 | 0.423949 | 0.859702 |
| 2023 K | 9 | 0.837816 | 2.079482 |

This substantial raw-point scatter remains visible in the
[replay archive](validation/20260927_magnesium_reference.json). It is not a
statistical confidence interval for K, and cannot establish an error bound
on omitted BSE transfer. The independent O gas reference is much more tightly
reproduced than this finite-Mg deoxidation data set.

The reported Mg and O extrema are 0.00002–0.00307 and 0.00034–0.00428 wt%,
respectively, at laboratory pressure. They do not form an independently
validated composition rectangle. The 2173.15 K source is 150.15 K above the
Mg experiment. Its historical 0.37–0.58 wt% alloy O is about 86–136 times the
largest measured O concentration. Polynomial continuation to that state
can generate enormous activity corrections and is not an empirical bound.
Extrapolation requires explicit opt-in and remains marked unsupported.
Si/H cross interactions, high-pressure response, a common high-O domain and
an integrable finite Mg extension remain unestablished. An alternative O
standard may change the equilibrium O concentration; assess its new state
before deciding whether this reference becomes applicable.

```console
PYTHONPATH=src python -m examples.m2_material.magnesium_reference \
  --output /tmp/new-magnesium-reference.json
PYTHONPATH=src python -m pytest -q tests/unittests/m2_magnesium_reference_test.py
```

Neither operation needs native MELTS, a sibling provider or a pressure solve.
