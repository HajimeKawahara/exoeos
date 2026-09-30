# Conditional Na partition in the archived OH boxes

The [primary Na replay](sodium_reference.md) supplies five S-free censoring
limits at 1 GPa and 1683–1883 K. This separate diagnostic **holds each derived
mass-partition limit constant in temperature, pressure and composition** and
asks how much Na would enter the existing metal box. It does not establish
that continuation experimentally or supply a new Na Gibbs model.

The [saved assessment](validation/20260928_sodium_fixed_box/assessment.json)
binds the original accepted OH closure at 2173.15 K and 269.6545152505794 bar.
Its primitive species amounts and formula matrix independently give

| Quantity | Original value |
| --- | ---: |
| Global Na inventory | 1.163488354656573e22 mol |
| Primitive silicate Na | 1.1606884137433066e22 mol |
| Primitive gas Na, held fixed | 2.799940913266546e19 mol |
| Primitive metal/cloud Na | 0 mol |
| Full primitive silicate mass | 9.678349204498963e22 kg |
| Silicate non-Na mass | 9.651665245661035e22 kg |
| Metal non-Na mass | 2.626564986262626e21 kg |

The original primitive Na sum differs from the global inventory by 1.80e-16
relative; that discrepancy is retained. The active box uses global Na minus
the fixed primitive gas/cloud Na. It does not renormalize the archived source.
Every non-Na host amount remains fixed, including silicate water and molecular
H2. Removing only Na mass from the initial silicate does not remove its oxygen
or other companion atoms.

For redistributed Na amounts `ns` and `nm`, Na molar mass `m`, and non-Na
host masses `Ms` and `Mm`, the equations are

```text
ns + nm = Nactive
ws = m ns / (Ms + m ns)
wm = m nm / (Mm + m nm)
D = wm / ws
```

Both phase denominators include their redistributed Na; there is no dry-host
or trace-denominator replacement. [`sodium_fixed_box.py`](sodium_fixed_box.py)
solves this monotone scalar partition to adjacent binary64 endpoints. It also
evaluates `Dcrit` directly at a requested absolute metal Na amount.

| Primary run | Continued D limit | Metal Na / global Na | Total metal Na (mass ppm) | Dcrit / D |
| --- | ---: | ---: | ---: | ---: |
| GGK1 | 0.0297741 | 0.0805500% | 82.0235 | 12.5182 |
| GGK2 | 0.0338491 | 0.0915653% | 93.2393 | 11.0112 |
| GGK7 | 0.0248869 | 0.0673361% | 68.5689 | 14.9765 |
| GGK8 | 0.0183900 | 0.0497655% | 50.6775 | 20.2674 |
| GGK9 | 0.0146128 | 0.0395475% | 40.2726 | 25.5064 |

The existing M2 engineering target of **1% of global Na in metal**, rather than
1% of only the active inventory, requires `Dcrit = 0.37271842604561445`.
This is **11.0111672 times** the largest limit obtained with the central
silicate concentrations. It quantifies the extrapolation required to reach
the target under the fixed-host assumptions; it gives no probability or
empirical bound on that extrapolation.

A separate set divides each reported metal limit by its silicate central
concentration minus the reported 2SE. Its largest metal fraction is
0.0996543% of global Na, and the corresponding Dcrit ratio is 10.116643.
This is not a new joint confidence interval. All ten box budgets close within
1e-16 relative. The primary observational limits, their analytical errors,
the constant-D continuation and the engineering target stay separate.

No component speciation, chemical potential, redox, phase stability, density,
atmospheric column or bottom-pressure response is solved. The fixed gas/cloud
amounts refer to the primitive source ledger. These small conditional transfers
do not certify a planetary pressure error or physical omission bound. In
particular, the FeS temperature regression cannot automatically be applied
as a multiplier of S-free D when oxide activities and host compositions change.

## A separate declared thermal continuation

The [thermal record](validation/20260928_sodium_thermal_box/assessment.json)
tests a different scenario. It transfers the central FeS/basalt exchange
temperature slope to the S-free limits while holding every other activity,
oxide-ratio and molar-to-mass conversion term fixed:

```text
K(T)/K(Tref) = exp[ln(10) * (-14340) * (1/T - 1/Tref)]
D(T)/D(Tref) = K(T)/K(Tref)  # only under the stated frozen-term assumptions
```

| Run | Actual reference T (K), 1 GPa | K multiplier to 2173.15 K | Assumed D at 2173.15 K | Metal Na / global Na |
| --- | ---: | ---: | ---: | ---: |
| GGK1 | 1683 | 83.5186 | 2.486692 | 6.345884% |
| GGK2 | 1783 | 27.7913 | 0.940711 | 2.489717% |
| GGK7 | 1683 | 83.5186 | 2.078516 | 5.354988% |
| GGK8 | 1783 | 27.7913 | 0.511082 | 1.366661% |
| GGK9 | 1883 | 10.3942 | 0.151888 | 0.409697% |

Four of the five central-silicate cases exceed the 1% engineering target.
The constant-D screen therefore does not remain small under this specified
alternative continuation. This is not a prediction of the fully coupled BSE
equilibrium, nor does a failed screen calibrate an absolute Na standard.

The printed `ln gammaNa = ln gammaFe + ln gammaNa_infinite
- epsilonNaS ln(1-xS)` relation has a zero S term for these S-free hosts even
when `epsilon(T)` scales as `1/T`. The authors did not measure Na-Si or Na-C
temperature effects, so those cannot be filled with the K-Si coefficient.
Their contribution and the remaining oxide/metal activity terms are frozen
assumptions here, not measured zeros. If the infinite-dilution reference is
changed, `K` must be rebased with it; a change of notation alone cannot create
this thermal effect. The fit intercept cancels in the ratio, and the reported
parameter errors do not supply a joint extrapolation uncertainty. Pressure
invariance is an additional declared continuation from 1 GPa to the old root.
The exact Na-containing mass denominators are still used inside each box.

This separate [runner](validation/20260928_sodium_thermal_box/replay.py) and
[execution receipt](validation/20260928_sodium_thermal_box/execution.json)
preserve the earlier temperature-invariant assessment without modification.

The [replay](validation/20260928_sodium_fixed_box/replay.py) requires the exact
source SHA and rejects an existing output path. The
[execution receipt](validation/20260928_sodium_fixed_box/execution.json)
retains the native-free run and focused tests. Earlier frozen OH and primary
reference records remain unchanged.
