# Conditional finite helium transfer to metal

[`helium_metal_box.py`](helium_metal_box.py) closes a finite He inventory in
three fixed boxes. It evaluates the size of a declared metal-partition
continuation without adding an equilibrium species or changing the archived
source. The [replay and original primitive ledger](validation/20260928_helium_metal_box/assessment.json)
use the accepted historical OH root at 2173.15 K and 269.6545152505794 bar.
They are not a new coupled source or bottom-pressure closure.

[Bouhifd et al. (2013)](https://doi.org/10.1038/ngeo1959), Supplementary Table S2,
reports simultaneous Fe80Ni20/CI-chondrite phase pairs. The five central
partition coefficients span 0.00067--0.017 at 1.9--13.1 GPa and 2400--2600 K.
Their [existing transcription and PDF hash](na_helium_transfer_sources.json)
are unchanged. Each central value is a separate scenario here: it is held
constant in pressure, temperature and phase composition. Reported analytical
uncertainties are retained in the receipt but are not a low-pressure uncertainty
interval. Table S3 uses separately interpolated silicate concentrations and is
not mixed into the same-run scenarios.

The silicate Henry coefficient `a` comes from the existing
[`helium_reference.py`](helium_reference.py) dry-host simulation interpolation.
With `N0` other gas moles, prescribed total pressure `P` in bar, He molar mass
`m` in kg/mol and fixed *non-He* host masses `Ms` and `Mm`, the equations are

```text
pHe = P ng / (N0 + ng)
ns  = Mdry a pHe
ws  = m ns / (Ms + m ns)
wm  = D ws
nm  = Mm wm / [m (1 - wm)]
ng + ns + nm = NHe
```

The scalar balance is monotone for the supported `0 <= D < 1`; bisection
retains its final binary64 bracket and budget residual. The dry mass `Mdry`
excludes host H2O and molecular H2, as required by the adopted Henry model.
The partition coefficient instead uses **total phase mass fractions**:
`Ms` includes the original silicate water/H2 and each denominator includes
its redistributed He. Using `D Mm/Mdry` as the ratio of metal/silicate He
amounts would mix incompatible conventions. No dilute replacement of these
mass denominators is made, although the continued constitutive laws themselves
remain dilute assumptions. The original gas denominator is also updated as
He moves; the other gas amounts, host masses, temperature and total pressure
remain fixed.

The archived source contains 1e23 mol global He; its primitive He gas amount
differs by 1.68e-16 relative from binary64 rounding. The new box uses the global
inventory and preserves that original discrepancy separately. It reconstructs
the original dry, full silicate and metal masses independently from the saved
species formula matrix, atom masses and amounts, and checks the masses against
the previous frozen Henry illustration.

| Dry-host capacity | Silicate He / total He at D=0 | Metal He / total He, five D scenarios | Largest metal He, total metal mass ppm | Largest change in ideal pHe from D=0, bar |
| --- | ---: | ---: | ---: | ---: |
| Olivine | 0.0010884720 | 1.97914e-8--5.02170e-7 | 0.0765253 | -1.91077e-5 |
| MORB | 0.0020722939 | 3.76799e-8--9.56056e-7 | 0.145693 | -3.63606e-5 |
| Rhyolite | 0.0050517771 | 9.18537e-8--2.33061e-6 | 0.355160 | -8.85073e-5 |

All 15 scenario budgets close within 1.30e-16 relative. These small conditional
transfers do not bound redox, phase selection, atmospheric structure or bottom
pressure. In particular, no Gibbs function or host chemical potentials for
finite He in this alloy have been supplied. The three simulated hosts do not
bracket wet BSE empirically. Material admission and BSE omission-error flags
remain false.

Reproduce without running native properties or chemical equilibrium:

```bash
python examples/m2_material/validation/20260928_helium_metal_box/replay.py \
  --source /path/to/original/OH/closure.json --output /path/to/new-assessment.json
pytest tests/unittests/m2_helium_metal_box_test.py tests/unittests/m2_helium_reference_test.py
```

The replay requires the exact original source SHA256 and never overwrites an
output. Its receipt preserves source, primary-data, evaluator and replay hashes.
