# Conditional sodium standards from projected reference hosts

The [2018 raw replay](sodium_reference.md) reports five S-free alloy Na
censoring limits, not five detected equilibrium concentrations. All nine basalt
analyses have K/Al > 1. Their exact oxide-to-MELTS transformation requires negative
`al2o3`; GGK2/GGK3 also require negative `sio2`. The three detected FeS liquids
add unsupported SO2. Consequently, their **actual measured host potentials are
not available within the declared nonnegative 19-component liquid domain**.
No amount was clipped and no actual-host calibration was claimed.

The [conditional evaluator](sodium_host_calibration.py) preserves measured Na
and Fe absolute oxide amounts, the selected analytical methods, and the original
alloy denominator including its calculated C fraction. It separately evaluates:

- Removal of K2O to atomic K/Al = 0.25, 0.5, or 0.9.
- Addition of Al2O3 to K/Al = 0.9, plus SiO2 when necessary to retain 0.05 mol
  free silica per original 100 g analysis.

Unsupported SO2 removal is explicit when requested. No total is renormalized.
Changed oxide masses, the original negative component amounts, full mass totals,
and the preserved Na/Fe moles are saved. These alternate compositions are model
choices, **not experimental error bars**. The five S-free analyses used in the
calibration have no positive measured SO2.

For a one-Na Henry standard, define the balanced virtual reference

```text
R_Na = 0.5 mu0_Na2SiO3 - 0.25 mu0_SiO2 - 0.25 mu0_Fe2SiO4
       + 0.5 mu0_Fe(metal)
mu0_Na,H = R_Na + delta(T)
```

All potentials are on common RT. This combines
`NaO0.5 = 0.5(Na2SiO3-SiO2)` and `FeO = 0.5(Fe2SiO4-SiO2)`;
its net formula is Na and it transforms correctly under an elemental gauge.
Actual projected-host potentials supply the mixing correction `b`, with the
same three coefficients. Under the explicitly selected S-free continuation
`gamma_Na,H = gamma_Fe`, the measured upper mole fraction implies

```text
delta >= b + 0.5 ln(x_Fe) - 0.5 ln(gamma_Fe) - ln(x_Na,upper).
```

This construction uses the original metal mass detection limit converted to
its recorded atomic denominator. It does not substitute the paper's ambiguous
molecular-oxide mole fraction for a one-Na chemical potential. Na–Si/C effects
remain unmeasured and are omitted in this continuation, not asserted zero.

The [saved exact T/P calculations](validation/20260928_sodium_host_calibration/assessment.json)
contain 20 proxy states evaluated by both native MELTS and the declared
published dry mixing expression at the experimental 1 GPa and 1683–1883 K.
All 40 evaluations succeeded; the largest native/published difference in a
Na constraint is 6.44e-11 RT. Every projection has GGK9 as its active constraint.

| Projection | Minimum constant delta | Delta at 2173.15 K with declared exchange slope |
| --- | ---: | ---: |
| Remove K, K/Al = 0.25 | 6.759749 | 4.418501 |
| Remove K, K/Al = 0.5 | 6.921915 | 4.580668 |
| Remove K, K/Al = 0.9 | 7.194308 | 4.853060 |
| Add Al/Si, K/Al = 0.9 | 7.546549 | 5.205301 |

The second family selects `delta=A+B/T`, with the **declared**, unfitted
`B=14340 ln(10) K` from the FeS exchange regression. The minimum A satisfying
all five censoring constraints is retained. There is no measured upper A:
raising A suppresses Na without violating a nondetection. Neither family is a
BSE calibration; low pressure, changed silicate composition, removal of C, and
thermal continuation remain explicit choices. Native virtual standards carry
their own T/P dependence, distinct from constant partition D or constant K.

The [frozen OH diagnostic](validation/20260928_sodium_proxy_trace/assessment.json)
reuses the original source SHA and its fresh native standard receipt at
2173.15 K and 269.654515 bar. Its actual Fe-metal standard is -8.083468321 RT,
`R_Na=-17.395169883 RT`, and `ln gamma_Fe=3.64445e-5`.
The thermal candidates give trace `x_Na=0.000380109–0.000834854`:
old metal particle moles times this fraction are 0.15823–0.34752% of the
**global Na inventory**. Constant-delta candidates give 0.01522–0.03343%.
Element potentials and all reservoirs remain frozen. These are neither finite
partition solutions nor pressure roots or empirical inventory bounds.

## Potassium diagnostic on the same saved hosts

The same state files allow a separate inverse calculation for the current
ideal-K component, without another native call. Use
`KO0.5=KAlSiO4-0.5 Al2O3-SiO2` and its balanced Fe exchange reference.
Original measured metal K is retained; projected silicate K may have changed.
Thus this is a **conditional proxy diagnostic**, not a new K fit to the actual
silicate compositions.

At the old OH state, `R_K=-22.473533192 RT` and the actual retained K gas
standard is -17.168085959 RT. The corresponding constant-delta continuation
of all twenty reference inversions gives gas-to-metal offsets
3.77177–9.50587 RT. For only the K/Al=0.9 removal projection, the five offsets
are 8.10651, 8.93682, 6.49613, 7.34139, and 7.95581 RT. This distinguishes
these offsets from `delta` on the virtual oxide reference.

The saved table also diagnoses a proposed pseudo-Fe/linear-Si K scalar using
`epsilon_KSi=10.42*1783/T`. This is kept separate from the Table S2 coefficients,
which give approximately `gamma_K/gamma_Fe=e` even in the Si-rich experiments.
For the 0.9 removal projection, the largest residual of an unweighted A+B/T
fit increases from 1.074 to 2.372 RT when this explicit Si term is used.
The projection and source activity conventions therefore do not establish that
this alternate scalar improves calibration. No K scalar or standard is replaced.

A [separate K thermal continuation](validation/20260928_potassium_proxy_thermal/assessment.json)
borrows only the central published `log10 K_K=4.41-12530/T` slope. At each proxy
reference, its inverse standard is retained; native virtual standards continue
at the target T/P. The strongest declared case is GGK7 with the K/Al=0.25 removal
projection: gas offset -0.0947612 RT, frozen `x_K=0.000537696`, or 4.22283% of
old global K inventory. The other projection maxima are 1.47054%, 0.276968%, and
0.144428%. Thus the constant-delta trace screen (maximum 0.088387%) does not
remain below 1% across this thermal continuation. This is a model sensitivity
endpoint, not an empirical upper bound or a new calibrated K scalar. The
positive Si interaction sign agrees with the primary trend; its poorer proxy
fit cannot distinguish host-activity changes from activity-convention choices.
