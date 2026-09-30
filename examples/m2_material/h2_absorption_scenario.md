# Molecular-H2 absorption recalibration scenario

[Chaudhari et al. (2025)](https://doi.org/10.1007/s00410-025-02272-y)
calibrated the H2 infrared coefficient using two methods on haplogranitic
glass: oxidation to water and total-H Karl Fischer titration minus infrared
water. Tables 3/4 contain six/four samples. Their published slope is
2.12 ± 0.05 L mol⁻¹ cm⁻¹, compared with the 0.26 coefficient used by
Hirschmann et al. (2012). The authors report agreement between recalculated
old basalt/andesite data and their new data in Fig. 8.

The [source record](water_data/Chaudhari2025_absorption_sources.json)
retains all ten calibration observations, including published differences
that differ slightly from subtraction of rounded water columns. It does
not infer unreported KFT errors or independently refit the slope without
the density and regression recipe. The receipt converts each measured
water-equivalent difference to molecular-H2 mass with one H2 molecule per
H2O molecule; no additional factor of two is introduced.

At fixed absorbance, the inferred concentration scales as the inverse
extinction coefficient. This scales a **dilute** Henry coefficient by
`0.26/2.12 = 0.1226415094`. For the existing dissolved-H2 model, the
corresponding standard scenario is

```text
delta(mu0_H2/RT) = ln(2.12/0.26) = +2.09848973665053
delta(G/RT) = n_H2_dissolved * delta(mu0_H2/RT)
```

The positive shift lowers dissolved H2 capacity. It is independent of the
water-capacity factor 2.265: the H2 factor comes from a measured spectral
recalibration, whereas the water interval covers retained central residuals.
Varying only the new coefficient by its reported ±0.05 gives standard shifts
[2.0746222552, 2.1218008155] RT; this is not a joint or extrapolation error bar.

The [saved scenario](validation/20260928_h2_absorption/scenario.json)
preserves the previously declared Sakao-O and Kato-H offsets and adds only
`H2_dissolved`. [Receipt](validation/20260928_h2_absorption/receipt.json)
and [execution record](validation/20260928_h2_absorption/execution.json)
bind the original scenario bytes, source values and helper. Generate another
record with:

```sh
python -m examples.m2_material.h2_absorption_scenario \
  --base-scenario original_scenario.json --output-dir new_output
```

This is an explicit continuation of the inherited mole-fraction scalar,
not a default change or a claim that a finite glass composition rescales
exactly like a dilute Henry coefficient. At fixed dry-host moles, scaling
solute moles by `q` instead gives `x_new = q*x/(1-x+q*x)`; the receipt displays
that difference. The experimental mole denominator, peridotite matrix
transfer and the uncalibrated temperature dependence of the inherited
`-11.403 - 0.76 P/GPa` law remain separate. No original result, gas standard,
host mixing term, or common material admission is changed. Inventory owns
the subsequent finite response and pressure closure.
