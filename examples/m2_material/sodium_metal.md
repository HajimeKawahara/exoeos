# Conditional finite sodium in associated metal

[sodium_metal.py](sodium_metal.py) adds a separately conserved Na1 component to
[K19](potassium_reference.md). For the excess term only, it evaluates the parent
at `x'_Fe=x_Fe+x_Na`, with every other species unchanged. The external
`total_solution_gibbs_RT` still uses all **twenty** physical ideal-mixing terms
and their own standards. No Fe atom is substituted into the physical Na ledger.

Equivalently, merge the Fe/Na amounts only when evaluating the parent G, then
add the Fe/Na ideal subdivision and `n_Na*(mu0_Na-mu0_Fe)`. Thus all potentials
come from one extensive scalar and `ln gamma_Na,H = ln gamma_Fe` exactly. This
is an explicit S-free continuation of the reference convention, not evidence
that Na-Si/C/O/H interactions vanish. Setting Na to zero recovers K19. At a
supported positive host, absent Na or Fe has its ordinary `-infinity` ideal
potential. Both absent retain the parent's Fe-host domain restriction; an
entirely absent phase has zero G and undefined potentials.

`make_associated_model(T, temperature_policy=..., hydrogen_oxygen_model=...)`
keeps the existing parent options. `COMPONENTS`/`FORMULAS` append Na1;
`associated_excess` exposes the same pure-arithmetic expression for independent
interval evaluation. The full20 Gershgorin helper keeps the supplied outer box,
drops the positive Fe ideal rank-one Hessian, and preserves a negative lower
bound when it occurs. It does **not** certify a stable equilibrium. A separate
consumer proof can analytically minimize the ideal Fe/Na split on the projected
parent domain while retaining original 20-species constraints for witnesses.

The consumer appends the result of
`sodium_standard_rt(T_K, P_Pa, actual_Fe_standard_rt, native_properties,
projection=..., temperature_policy=...)` to its existing 19-standard vector.
Both selectors are required. Projections are `remove_potassium_0.25`,
`remove_potassium_0.5`, `remove_potassium_0.9`, and
`add_aluminum_silica_0.9`; temperature policies are `constant_delta` and
`published_exchange_slope`. See the [reference derivation](sodium_host_calibration.md).

```text
mu0_Na(T,P) = 0.5 mu0_Na2SiO3 - 0.25 mu0_SiO2 - 0.25 mu0_Fe2SiO4
              + 0.5 actual_parent_Fe_standard + A + B/T
```

The three native standards must be supplied at the exact T/P and recorded
common R. The recipe retains their values, source provenance, actual Fe standard,
selected A/B and primary calibration hash. It does not replace an independently
anchored Na gas standard or infer a gas offset. **Na's virtual standard follows
P**. In contrast, a fixed K gas-to-metal offset, including the separate declared
thermal endpoint -0.09476121345 RT, keeps that dimensionless offset fixed during
a pressure search. These are intentionally different declared pressure laws.

The [saved provider check](validation/20260928_sodium_metal/assessment_after_interval_review.json)
uses the old OH source's exact standard receipt and confirms the Na candidate
standard -12.976668910 RT at 2173.15 K and 269.654515 bar (delta=4.418500973 RT).
Independent scalar/AD/Euler checks cover both H-O choices. Its other artificial
standards test algebra only; this is not a new source equilibrium. The
[execution receipt](validation/20260928_sodium_metal/execution.json) records
focused checks and preparation failures without changing prior results.

The material diagnostic accepts `metal_model='associated_k_na'` and counts
all twelve alloy elements, including Na, in phase mass. Its ledger marks the
conditional scalar as available, with finite response and robustness assessment
still separate. No empirical BSE domain or phase acceptance is inferred.
