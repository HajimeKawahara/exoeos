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

## Optional hydrogen-oxygen interaction

`make_associated_model(..., hydrogen_oxygen_model="schenck1961_abstract")`
adds an explicit symmetric O/H entry to the same additional matrix. The default
`"omitted"` retains the original scalar. A contemporary Japanese abstract of
[Schenck and Wuensch (1961)](https://doi.org/10.1002/srin.196103272),
[Tetsu-to-Hagane49(1),1963,p96](https://tetsutohagane.net/articles/search/files/49/1/KJ00002706857.pdf),
states `d log10(gamma_H) / d x_O = 52.4` at1610C and independently illustrates
`x_O=.00343, gamma_H=1.52`. The reconstructed value is1.513, establishing the
base10 convention; using52.4 directly with natural logarithms gives1.197.
The natural-log interaction is therefore `epsilon_HO=52.4*ln(10)`.

The primary full concentration table and uncertainty remain unavailable.
[The source record](hydrogen_oxygen_sources.json) preserves that distinction,
the inspected PDF and page-image hashes, the explicit equation and example.
The inconsistent1967 retelling of52.4 and mass-percent2.71 is not silently
used as a modern natural-log conversion. This is a historical-reference
continuation, not an independently refitted H/O data set.

The additional extensive term is `N_species*epsilon_HO*x_H*x_O`, on the
complete18-species denominator. It changes both potentials reciprocally.
The existing temperature policy selects either a constant coefficient or
`epsilon_HO(T)=epsilon_HO(1883.15K)*1883.15/T`; the distinct P reference remains
1873.15K. Pressure response, H self interaction and empirical BSE errors
are not supplied.

The finite-K factory accepts the same option and retains this eighteen-species
host normalization. Its extensive perspective contributes
`N_19*epsilon_HO*x_H*x_O/(1-x_K)` in nineteen-species coordinates. It reduces
exactly to the eighteen-species term at zero K; its curvature audit evaluates
the actual perspective, including the nonconvex host when selected.

At2173.15K the old declared box has curvature bounds -78.7129 (constant) and
-63.2693 (enthalpic), so the old positive-curvature proof cannot be reused.
A separate outward point calculation also proves negative O/H-direction
curvature at one feasible point: approximately -112.8146 and -80.6125.
This establishes nonconvexity of these declared continuations, not an unstable
computed equilibrium, a negative insertion value, or experimental phase
separation. Global insertion requires a proof that admits nonconvexity.
[Saved box comparison](validation/20260928_hydrogen_oxygen/comparison.json)
and [separate point calculation](validation/20260928_hydrogen_oxygen_curvature_v2/comparison.json)
preserve both results. No source equilibrium or pressure closure was run here.
