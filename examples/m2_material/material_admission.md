# Actual-state material evidence

`material_admission.py` evaluates a supplied liquid/alloy state against the
source-specific material evidence. It reports temperature, pressure,
composition and concentration differences instead of copying the selected
2173.15 K, 1 bar control's status to a changed state. It does not alter any
energy, activity, standard state or archived result.

Call `assess_material_state(temperature_K, pressure_Pa, ...)` from a source
checkout. The keyword arguments are:

| Argument | Required basis |
| --- | --- |
| `silicate_oxide_mass_fractions` | Normalized native-host oxide masses, including native H2O and excluding the added molecular H2; chemical-case or lowercase MELTS names |
| `molecular_h2_mass_ppm` | Molecular-H2 mass divided by complete liquid mass, including native water and H2 |
| `water_mass_percent` | Native-H2O mass divided by that same complete liquid mass |
| `alloy_atomic_fractions` | Fe, Si, O, H atomic fractions as a mapping or vector; `None` if absent/unsupplied |
| `hydrogen_partial_pressure_Pa` | H2 fraction of the complete gas species set times total pressure |
| `buffer` | Actual imposed experimental buffer, or `None`; it is never inferred from the presence of Fe and water |

The caller owns amount-to-mass conversion and species identification. The
evaluator checks normalization, names, nonnegative finite values, H2 partial
pressure versus total pressure, and the shared concentration denominator.
For a dry experimental-host comparison only, it removes native H2O and
normalizes the remaining oxide masses. For the Sossi glass comparison it
reports Fe as FeO-equivalent before normalization. These reporting conversions
do not modify the supplied physical state.

The output distinguishes the following evidence:

| Evidence | Evaluation |
| --- | --- |
| Nominal MELTS limits | Coordinate inclusion; never a liquid-stability certificate |
| Sossi KLB-1-derived glass | Reported T/P and oxide differences, with FeO-total reporting |
| Chaudhari named hosts | Measured T/P pairs, the existing reference evaluator's permitted interpolation, buffer identity, reported host composition and concentration envelopes |
| Kato Fe/Fe-Si reference | Atomic-to-mass composition conversion, temperature, 1-atm total/partial pressure, absence of O, and the published Si mass-percent limit |
| Common implemented H references | Necessary-condition temperature/pressure intersection, calculated from the source data |

Composition differences have no fitted acceptance radius. An unreported oxide
remains unreported; no analytical detection limit is invented. Exact equality
to a reported host permits that reference's condition check but does not
validate a nearby composition. Likewise, observed H2/H2O ranges across runs
and the dilute alloy fit's concentration range do not define independent
concentration domains. An absent alloy does not validate its incipient state.

The two implemented hydrogen-reference families have a 170 K temperature gap
and a 499898675 Pa total-pressure gap. This excludes a joint *measured
reference domain for those families*, without proving that no physical common
domain exists. Every report keeps `material_admission="not_established"` and
`accepted_coupled_material_domain=null`. Competing phases, reaction standards
and omitted-transfer errors require independent assessments.

## A published extrapolation comparison

[Chaudhari et al. (2025)](https://doi.org/10.1007/s00410-025-02272-y), printed
pages 14–16, give an illustrative pure-H2 coefficient of 515 mass ppm/GPa,
derived from their buffered basalt regression. They identify its transfer to
hotter peridotite as uncertain. `published_basalt_h2_extrapolation` reproduces
that illustration, additionally setting fugacity equal to H2 partial pressure
for an ideal mixture:

```text
c_H2_mass_ppm = 515 * p_H2_Pa / 1e9
```

At one bar of pure H2 this gives 0.0515 ppm. The report compares the supplied
concentration with this value and records the assumptions. It fixes
`supports_material_admission=false` and `is_error_bound=false`. No calibration,
upper bound or constitutive replacement follows from the comparison. A zero
reference concentration produces a null ratio and an explicit zero flag.
This uses the authors' stated illustrative coefficient, not guessed units
for their separate Equation 5 fugacity regression.

The [Hirschmann et al. (2012) abstract](https://doi.org/10.1016/j.epsl.2012.06.031)
supports experiments at 0.7–3 GPa on basalt/andesite and identifies peridotite
solubility as an ionic-porosity extrapolation. The original concentration mole
denominator remains unverified in this audit; no correction is silently
applied to the inherited model. The 170 K gap above concerns the implemented
Chaudhari/Kato references only, not every earlier experiment or extrapolation.

## Reproduction

Save the function arguments as a JSON object and use a new output path:

```console
python examples/m2_material/material_admission.py \
  --input /tmp/material-state.json --output /tmp/material-evidence.json
```

The result hashes both source evidence files, the evaluator and the input
file. A report applied to an archived state is a new evidence assessment of
that state, not a new equilibrium result. Ordinary tests require neither
alphaMELTS nor a sibling provider; they cover unsupported T/P combinations,
host mismatch, atomic/mass conversion, water denominators, exact absence and
the distinction between a reference illustration and material admission.
