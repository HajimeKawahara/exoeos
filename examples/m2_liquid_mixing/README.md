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

## Explicit constitutive-model selection

`make_published_liquid_evaluator(native_evaluator, runtime=...,
python_executable=...)` returns a liquid-only property callback with model ID
`melts_v102_published_mixing_native_standard_states_v1`. Its
`evaluate_liquid(T_K, P_Pa, n, ...)` gets native pure-liquid standards once per
exact T/P/R, using a positive probe of all 15 supported components, then
computes subsequent total G and potentials from the published expression.
`standard_state_receipts` retains every original probe, compatibility check,
standard-state policy and SHA256. No standard is fitted or interpolated.

This explicit model lets a consumer use precisely the same coefficients for
equilibrium and global phase-stability bounds. Native composition evaluation
remains available separately as a control. Newly computed states contain only
G, potentials, activities, mass, basis and composition data; they do not reuse
probe density, volume, enthalpy, entropy or heat capacity. Candidate saturation
and other calculation modes are not supplied by this callback. Exact-zero
amounts retain continuous energies and unavailable absent potentials.
# Independent derivative audit

The published evaluator exposes `energy_value_and_grad_rt(T_K, P_Pa, n, ...)`
for an independent scalar audit. JAX differentiates a separately written
extensive standard/regular-solution/entropy expression on the positive
component face. It shares native pure standards and the published coefficients
as physical inputs, and does not reuse the analytical mixture potentials or
mixture energy. Exact-zero components have unavailable face derivatives;
present trace components remain auditable even when total-energy finite
differences lose significance. This numerical audit does not extend the
empirical material domain or certify native/published global equivalence.
