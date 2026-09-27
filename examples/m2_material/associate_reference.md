# Finite oxygen associates in the metal phase

`associate_reference.py` supplies one extensive species-mole Gibbs model for
Fe, Si, O, H, P, Mg, Ca, Al, Cr, Ti, MgO, CaO, AlO, CrO, TiO, Al2O, Cr2O
and Ti2O. The associated species remain inside the metal reservoir. Their
formulas, rather than their number of particles, determine elemental budgets.
No equilibrium solver or planetary inventory is implemented here.

The retained Fe-Si-O-H/P scalar is diluted by the fraction `h` of its five
atomic species. Additional symmetric interactions form `x.T E x / 2`:

```
g_ex(x) = h g_ex,MaP(x[:5]/h) + x.T E x / 2
G/(RT) = n.mu0 + sum_i n_i log(n_i / sum_j n_j) + sum_j n_j g_ex(x)
```

The derivative of this single scalar determines every chemical potential,
including reciprocal cross effects. At zero added amounts it recovers the
existing finite-P host. Zero interaction entries specify omitted terms in
this continuation; they are not assertions of measured zero interactions.

## Primary references and common standards

[Jung, Decterov and Pelton (2004)](https://doi.org/10.1007/s11663-004-0050-4),
Tables I/II, Section V and Appendix II provide the free-metal Henry references,
associate energies, self interactions and Ca-Al/Ca-Si/Si-Al interactions.
The [same-author open model description](https://doi.org/10.2355/isijinternational.44.527)
gives the species configurational entropy and atom balances. Numerical facts
and source locations are pinned in [associate_sources.json](associate_sources.json).

| Associate | Formation G from free metal and free O (J/mol species) |
| --- | ---: |
| MgO | -218272 |
| CaO | -305828 |
| AlO | -108614 |
| CrO | -41840 |
| TiO | -96145 |
| Al2O | -179912 |
| Cr2O | -62760 |
| Ti2O | -142256 |

The actual gas standard supplied by the consumer anchors each elemental gauge.
Mg/Ca Henry standards refer to their gases; the primary paper's Figs. IIb/IIc
explicitly use bar. Al, Cr and Ti standards use the JANAF pure-minus-atomic-gas
Gibbs difference, then add the source Henry coefficient. The selected pure
phases are Al liquid, Cr crystal and Ti liquid, including the metastable Cr
crystal continuation above its melting point. The tabulated 2100--2400 K
JANAF values are interpolated in G with entropy derivatives; extrapolation is
rejected. The original five host standards are retained exactly.

The associate standards use the independently specified Jung free-O reference.
Its difference from the host free-O reference is saved explicitly. Named O/H
host-standard scenarios change only the free host species unless a different
model is explicitly constructed. This hybrid retains the Ma/P host; it is not
a claim to reproduce the complete original Jung oxygen model. No pressure
volume correction is supplied for the new condensed standards.

P-Al/P-Cr/P-Ti coefficients come from
[Yamada and Kato (1983)](https://doi.org/10.2355/isijinternational1966.23.51).
The H-Cr coefficient `e_H^Cr=-0.0056` comes from
[Ban-ya, Fuwa and Ono (1967)](https://doi.org/10.2355/tetsutohagane1955.53.2_101)
(Cr below 20 mass %, 1550--1670 C, 1 atm H2). Its conversion includes both
mass-to-mole and normalization terms. Existing P-reference choices and the
constant/enthalpic continuation remain explicit alternative models.

## Mathematical domain and physical limits

`associated_excess` is a pure arithmetic expression suitable for independent
interval evaluation. `associated_curvature_lower_bound` evaluates the
17-dimensional reduced Hessian with outward interval arithmetic. The ideal
Fe rank-one Hessian is positive semidefinite and may be omitted from a lower
bound. Positive numerical weights only propose a diagonal similarity scaling;
every Gershgorin row is then evaluated outward. No computed eigenvalue itself
is accepted as a certificate.

The source adapter declares a new species box with Fe at least 0.75, Si at
most 0.02, free O at most 0.01, and the remaining bounds in its receipt. At
2173.15 K its matched curvature lower bound is approximately 1.4579 RT/mol
species. This is a mathematical restriction of the new scalar, not a revision
of the previous model's box or an empirical calibration interval. Boundary
contact must remain visible in the source audit.

Mg/Ca binary Henry data were available at 1873 K; Jung explicitly adopted an
enthalpic temperature continuation. The current 2173.15 K use and additional
H/P cross terms therefore need finite-response assessment. A frozen-potential
trace demand is not a finite equilibrium or a bound on coupled errors. K metal
transfer remains outside this model. Primary K non-detection at different
pressure and host composition is not silently imported as a BSE error bound.
