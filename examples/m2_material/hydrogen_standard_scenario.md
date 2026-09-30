# Measured atomic-H standard scenarios

The [existing Kato reference](hydrogen_reference.json) also permits a
concrete repair of the inherited atomic-H reaction standard. The adopted
source predicts only 12.6047 ppm H in dilute pure Fe at 1873.15 K and one
atmosphere H2, compared with the measured 25.0 ppm reported by
[Kato et al. (1970)](https://doi.org/10.2355/tetsutohagane1955.56.5_521).
This difference precedes any application to a high-pressure Fe-Si-O-H alloy.

For `0.5 H2(g) = H(metal)`, the published law is
`wt%H = 10^(-1874/T - 1.601) sqrt(p_H2/101325 Pa)`. The existing measured
pure-Fe temperature interval is 1843.15–2013.15 K. With `w_H=wt%H/100` at
one atmosphere H2, convert to the caller's atomic mole-fraction convention:

```text
mu0_H,Henry,target/(RT) = 0.5 [mu0_H2,gas/(RT) + ln(101325/p0)] - ln(w_H)
mu0_H,Henry,adopted/(RT) = mu0_H,formal/(RT) + ln(gamma_H,infinity) - ln(M_H/M_Fe)
offset_H = target_Henry_standard - adopted_Henry_standard
```

`atomic_h_standard_scenario` receives the actual H/H2 standards, matching
infinite-dilution H coefficient, T, total P and gas standard pressure `p0`
in Pa. M_H=1.00794 and M_Fe=55.845 g/mol are explicit. The returned
`standard_offsets_rt={"H_metal": offset_H}` replaces the old reaction
standard through its **difference**, rather than adding the complete new
standard. It leaves the Ma excess Gibbs model unchanged. Consistent H
elemental-gauge and 1-bar/1-atm gas-standard changes cancel.

The [actual-source receipt](validation/20260927_atomic_h_standard.json)
uses Gibbs `590ddd278c792006dc85cc1e14f0855b2b38c5fb`, EOS
`f6967cbc5782cc6b230e07ea5953082ead8b54fb`, and the 76-gas/26-condensate
catalog's 13-element gauge at 2173.15 K. The common 35-gas H2/H2O standards
are identical there. At 1873.15 K the six added JANAF atomic references are
outside their tabulated excerpt, so the measured H comparison evaluates
only the shared H/O reference. It does not extrapolate those six atoms.

| 1873.15 K observation | Observed H ppm | Inherited Ma H ppm | Corrected Ma H ppm | Published Henry fit ppm |
| --- | ---: | ---: | ---: | ---: |
| Pure H2, one atmosphere | 25.0 | 12.604707 | 25.069065 | 25.034921 |
| H2:Ar = 1:10, one-atmosphere total P | 7.34 ± 0.18 | 3.798643 | 7.551414 | 7.548313 |

The second row retains all five raw sampling replicates. Its reported
standard deviation is not a universal model tolerance. The small difference
between the last two columns is the exact atomic-fraction-to-mass conversion:
the adopted Ma extension has zero H excess coefficient, whereas the
measurement's Henry law is ideal in mass fraction. This difference is
reported rather than absorbed into a finite-concentration fit.

At 2173.15 K the **extrapolated** H offset is -0.4445191104224264 RT.
Combining it with the independent oxygen references gives:

| Conditional source scenario | O_metal offset / RT | H_metal offset / RT |
| --- | ---: | ---: |
| Kato H only | 0 | -0.4445191104224264 |
| Sakao O + Kato H | +1.6320817196241961 | -0.4445191104224264 |
| Matoba O + Kato H | +1.6446793177347274 | -0.4445191104224264 |

These are linear, integrable standard changes suitable for new conditional
source/global solves. Temperature extrapolation requires explicit opt-in;
applying a constant RT offset across all total pressures at this T is an
explicit sensitivity assumption. It does not establish high-pressure
fugacity, high-O/Si/H interactions, the silicate-water standard, or a common
BSE material domain. No archived source or closure is changed here.
