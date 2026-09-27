# Constitutive evidence at the actual source state

[The evidence ledger](constitutive_evidence.json) separates reported conditions,
declared temperature/pressure continuations, ranges covering reference
discrepancies, and unidentified coefficients for six material subsystems. Source
records have fixed commit links and SHA256 values. The ledger preserves the
distinction between a reference measurement, an integrable constitutive model,
and uncertainty in applying that model to a different state.

`assess_material_state` accepts optional `liquid_model`, `metal_model` and
`hydrogen_oxygen_model` selectors. Its `constitutive_evidence` output retains
these declarations and the actual temperature, pressure, complete-liquid
concentrations, dry oxide comparison and complete-alloy atomic/mass fractions.
Omitted selectors mean unspecified; neither a standard shift nor a model is
inferred from its resulting concentrations. Source receipts retain the actual
scenario offsets separately.

The existing Chaudhari/Kato condition checks and numerical concentration values
are unchanged. Their `reference_conditions_supported=false` means that the
specific experiment does not match, not that another declared constitutive
model is rejected. Historical `native_water` keys remain aliases for the same
equivalent-water concentration comparison. The diagnostic still makes no
coupled acceptance decision; `not_established` is not a numerical solver gate.

In particular, peridotite water and binary Fe/O have direct evidence near the
source temperature. Pressure transfer, finite-H interactions, the hybrid
associated-alloy host and K standard have different remaining uncertainties.
A missing simultaneous measurement of every component is not an automatic
reason to reject all constitutive models. Numerical closure and global phase
support must be established independently, then actual finite responses can
test robustness over stated model choices. A range covering reference data
does not become an empirical bound at an unmeasured source state.

This addition changes no Gibbs energy, phase-selection tolerance, stored source
result or experimental observation.
