# Supplied regular-liquid mixing expression

`../melts_liquid_mixing.py` exposes the published rhyolite-MELTS 1.0.2
mixing expression independently of the native optimizer. It supplies a
physical-property expression; phase-stability decisions remain in ExoGibbs.

For native fractions `x`, native water fraction `w`, and `alpha = R_native/R`,

```text
g_mix/(RT) = 0.5 x' W x
             + alpha [sum_i x_i log(x_i) + w log(w) + (1-w) log(1-w)]
W_ij = [H_ij - T S_ij + (P_bar - 1) V_ij]/(RT), i != j
W_ii = 0
```

The boundary convention is `0 log(0) = 0`. The pure-water energy has a finite
zero mixing limit even though a direct native endpoint call may return NaN.
Absent-component potentials are unavailable; they are never filled with zero.
The four unsupported CO2/S/Cl/F components remain excluded in mode 1.
Component element counts identify absent-element exclusions separately from
an arbitrary zero amount. Standard states, oxygen buffering and equilibrium
compositions are not changed.

`parameters.json` transcribes the 171 interaction records from the official
[MAGMA parameter source](https://github.com/magmasource/MAGMA/blob/705a0fb315e5054d18275a580562f6121c8e458c/includes/param_struct_data_v34.h).
The equation and extra water entropy are in
[`gmixLiq_v34` and `actLiq_v34`](https://github.com/magmasource/MAGMA/blob/705a0fb315e5054d18275a580562f6121c8e458c/sources/liquid_v34.c).
The source files and revision are SHA256-pinned in the dataset. Numeric source
coefficients are data; the external C implementation is not redistributed.

`liquid_mixing_parameters(T_K, P_Pa, common_R)` returns the dimensionless
matrix, entropy coefficient, element ledger and provenance.
`liquid_mixing_state(parameters, component_moles)` returns extensive mixing
energy and present-component potentials. `compare_native_mixing(properties)`
compares the published expression with the independently obtained native
`mu - mu0` at the supplied state.

All three existing native reference states test this comparison. This is
finite compatibility evidence: the release binary's thermodynamic C build
commit is not independently established, and finite comparisons do not prove
a uniform native error bound. A certificate for the explicit expression must
not be reported as empirical calibration or a certified error bound for an
unidentified native build. Provider consumers retain both records separately.
