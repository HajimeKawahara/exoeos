# Fixed-box helium partition threshold

This supplements the [finite-He metal diagnostic](helium_metal_box.md) without
changing its fifteen scenarios or original receipts. It asks what imposed
mass-fraction partition coefficient would put **1% of the total He inventory
in metal**, with the same frozen hosts, temperature, total pressure and other
gas amounts. The 1% is an engineering partition criterion, not a measurement
error or a confidence bound on low-pressure material properties.

For a target `eta`, first set `nm = eta NHe`. The remaining `(1-eta) NHe`
obeys the original gas/silicate balance with no further He allocated to metal.
The required coefficient then follows directly:

```text
Dcrit = [m nm / (Mm + m nm)] / [m ns / (Ms + m ns)]
```

Both fractions retain He-inclusive total phase mass denominators; `Ms`
includes the original host water and molecular H2. The gas/silicate solution
still uses dry mass in its Henry capacity and the complete changing gas
denominator. This inverse calculation can return `Dcrit > 1` without extending
the original forward diagnostic's `D < 1` API: the computed finite threshold
state has both mass fractions strictly between zero and one.

| Simulated dry-host Henry capacity | Dcrit at metal He / total He = 0.01 | Dcrit / maximum Table S2 central D (0.017) |
| --- | ---: | ---: |
| Olivine | 340.8498893 | 20049.99349 |
| MORB | 179.0325669 | 10531.32746 |
| Rhyolite | 73.44262194 | 4320.154232 |

The [three saved threshold states](validation/20260928_helium_metal_box/threshold.json)
close He conservation within 1.17e-16 relative and preserve the exact hash of
the original box assessment. Their ratios quantify how far an **imposed**
constant D would need to move from the largest retained Table S2 central
value to cross this particular fixed-box criterion. They establish neither
an allowed range for D nor an empirical exclusion of such a value at 270 bar.
`critical_D_is_adopted_material_parameter`, `supports_material_admission` and
`is_BSE_error_bound` remain false. Atmospheric-mass, bottom-pressure and
metal-appearance-boundary targets are unassessed.

[`helium_metal_threshold.py`](helium_metal_threshold.py) uses the unchanged
forward gas/silicate balance. Tests replay a sub-unity threshold through the
full three-box forward equation and independently verify a super-unity D
against finite mass-fraction and Henry identities. No source Gibbs model or
native calculation is invoked.

```bash
python examples/m2_material/validation/20260928_helium_metal_box/threshold.py \
  --output /path/to/new-threshold.json
```

The runner checks all original diagnostic recipe hashes and refuses to
overwrite an output.
