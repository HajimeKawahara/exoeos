# Finite Fe-P liquid association

`exoeos.phosphorus_association` exposes the Fe/P/Fe2P liquid scalar from
[Oikawa and Ueshima (2026)](https://doi.org/10.2355/tetsutohagane.TETSU-2025-090),
plus an explicit conditional extension to the existing associated alloy.
The primary PDF (Table 1, p. 355) and original supplementary Table S1 agree
on all five binary parameters. The [source ledger](phosphorus_association_sources.json)
records their hashes.
Fe2P is a chemical species within the liquid, not a solid precipitate.

For chemical amounts `(n_Fe, n_P, n_A)`, `A=Fe2P`, use
`N=sum(n)` and `y=n/N`. The nonideal extensive energy is

```text
G_nonideal = n_A*dG_A
           + N*y_Fe*y_A*(L0 + L1*(y_Fe-y_A))
           + N*y_A*y_P*(M0 + M1*(y_A-y_P)).
```

All coefficients are J/mol. The scalar adds one ideal species-mixing term
and the supplied elemental standards. The atom ledger is
`Fe=n_Fe+2*n_A`, `P=n_P+n_A`; chemical and atomic mole fractions differ.
ExoGibbs owns the constrained equilibrium of these three species.

## Common reference and conditional extension

The binary paper uses pure-liquid standards. The existing consumer instead
supplies a P Henry standard anchored to its actual gas. In the Fe-rich
infinite-dilution limit, `K=exp(-(dG_A+L0+L1)/(R*T))` is the ratio of bound
to free P. Adding `ln(1+K)` to the input Henry standard gives the required
free-P standard. The Fe2P standard is then `2*mu_Fe+mu_P_free+dG_A/(R*T)`.
This preserves the original **equilibrated binary** dilute P standard
without fitting a new offset to an inventory or phase outcome. It does not
reconstruct the paper's absolute SGTE/SSUB unary/gas database.

The 21-species extension keeps the existing host law, pseudo-Fe Na term,
and independently supplied standards. It collapses Fe2P into two Fe and
one P when evaluating the retained host and cross terms. The old quadratic
P-P self term is removed; the five binary association parameters replace it.
The new binary terms use actual free Fe/P/Fe2P. Global species entropy is
counted once. Both the scalar and its host derivatives change consistently.
No new multicomponent cross coefficient is inferred from the binary fit.

The `quadratic` control reproduces the parent scalar and supplies its
continuous pure-P endpoint. It allows callers to separate the effects of
changing a numerical composition domain from replacing the P-P law. Neither
provider choice installs a P upper bound. Consumers must state their actual
host constraints and report contacts with every remaining artificial face.

## Evidence and limits

The paper assesses binary Fe-P phase and thermodynamic data, with liquid
activity comparisons at 1673 and 1873 K and finite-P calorimetry. It also
reports sparse high-P liquid evidence. The present use at 2173.15 K and in
a Si/O/H/K/Na-bearing host is a conditional continuation. There is no
calibrated pressure term or empirical multicomponent error certificate.
Solid phosphides, the full binary phase diagram, and the Fe-C-P extension
are outside this implementation. A binary liquid calculation below the Fe
melting temperature may describe a metastable liquid.

Alternatives include the modified-quasichemical assessments by
[Bernhard et al. (2020)](https://doi.org/10.1016/j.calphad.2020.101795) and
[You and Jung (2020)](https://doi.org/10.1007/s11663-020-01939-0).
The publicly available association parameters permit the smaller explicit
species implementation here. No claim of comparative superiority is made.

Tests check the analytic reaction derivative independently, exact atom
stoichiometry, the dilute reference conversion, parent and binary limits,
extensivity, elemental gauge invariance, reciprocity and T derivatives.
The frozen Inventory experiment performs separate source and insertion
calculations; numerical acceptance does not establish material admission.

The preserved Henry **standard** must not be confused with the dilute
P-P derivative. Writing `k=K/(1+K)`, the binary model gives

```text
epsilon_P_P = 4*k + 2*(-k*L0 - k*(2+k)*L1 + M0*k*(1-k))/(R*T).
```

At 1873.15 K this is **3.57770**, versus the independently reported
**7.3 +/- 0.1** in [Yamada and Kato (1979)](https://doi.org/10.2355/tetsutohagane1955.65.2_264)
(0.7--3.2 mass percent P, extrapolated dilute derivative). At 2173.15 K it
is 1.51298; the old conditional quadratic uses 7.3. A 39-point binary
numerical replay confirms the analytical limit and smooth finite-P response,
but does not resolve this disagreement between material descriptions.
No coefficient is adjusted to conceal it. The candidate is therefore a
**sensitivity model, not an admitted replacement** for the existing alloy.
