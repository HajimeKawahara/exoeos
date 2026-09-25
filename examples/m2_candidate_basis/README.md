# Native candidate amount-basis validation

The saved property evaluations use the unchanged high-H pilot host at
2173.15 K and 432.298471237978 bar. They reevaluate properties and do not
resolve the source or change its earlier archived result.

| Candidate | Original oxide property call | Direct native endmember evaluation |
| --- | --- | --- |
| Hornblende | Changes the requested oxide amounts | Maximum oxide error 1.07e-14 g on the requested 100 g basis |
| Orthoamphibole | Changes the requested oxide amounts | Maximum oxide error 7.11e-15 g on the requested 100 g basis |
| Kalsilite | No finite incipient saturation estimate | Two finite interior compositions; incipient estimate remains unavailable |

`native_supplied_host.json` preserves the complete provider receipt, original
oxide-conversion diagnostics, native endmember bases, and unchanged host.
`native_kalsilite_interiors.json` preserves separate evaluations with native
endmember amounts `(0.25, 0.25, 0.25, 0.25)` and
`(0.499, 0.499, 0.001, 0.001)` mol. Their energies are finite, but neither
point proves phase stability or a global minimum. Runtime and executed
evaluator hashes are embedded in both receipts.

The upstream [hornblende conversion](https://github.com/magmasource/MAGMA/blob/main/sources/hornblende.c)
assigns elemental Mg directly to the pargasite amount, despite four Mg
atoms per pargasite and Mg in the ferric endmember. The
[amphibole conversion](https://github.com/magmasource/MAGMA/blob/main/sources/amphibole.c)
can reset slightly negative inferred ferric Fe to 1% of total Fe, changing
the requested ferrous amount. The direct molar API bypasses these inverse
conversions. Its returned molar energy and mass are scaled by the reconstructed
endmember total; no tolerance, binary, model parameter or oxide amount changes.

The [kalsilite scalar](https://github.com/magmasource/MAGMA/blob/main/sources/kalsilite.c)
contains vacancy-dependent logarithms and an endpoint penalty. Finite interior
properties do not justify replacing unavailable endpoint or saturation
values with zero. Composition searches must retain these failed evaluations
and may report a negative feasible witness; nonnegative samples remain
insufficient to certify absence.
