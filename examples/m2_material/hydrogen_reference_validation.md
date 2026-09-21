# Hydrogen reference receipt, 2026-09-22

`hydrogen_reference_validation.json` was generated from clean source commit
`fd06abb0c7484aa88a19931300f3c1df106a2012` with:

```bash
python examples/m2_material/hydrogen_reference.py \
  --output examples/m2_material/hydrogen_reference_validation.json
```

The receipt includes the source-data and evaluator SHA-256 hashes, all 19
retained Chaudhari observations and regression residuals, and two Kato
pure-Fe reference checks. The two author-excluded observations remain in
`hydrogen_reference.json`. These are published-regression reproductions;
the residuals are not a new fit or acceptance of every experimental point.

The mass-fraction scalar and source-restricted evaluators passed 23 focused
tests. The complete ExoEOS suite passed 395 tests, and Sphinx built the
rendered documentation with warnings treated as errors.

The silicate and metal reference temperature ranges do not overlap. No
coupled BSE material domain, O-containing alloy calibration, or M2-A/B
scientific acceptance is established by this receipt.
