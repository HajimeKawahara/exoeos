# Finite phosphorus in an iron-rich alloy

`phosphorus_reference.py` supplies a five-component Fe-Si-O-H-P scalar for
the M2 example. ExoGibbs owns finite equilibrium; ExoInventory owns the
elemental inventory and pressure closure. No frozen-reservoir demand is
interpreted as a finite equilibrium or a coupled error bound.

The original four-component Ma host is recovered at zero P. All five atomic
fractions contribute to ideal mixing. The excess term is the extensive host
excess plus `epsilon_PP*x_P**2/2 + x_P*sum(epsilon_Pj*x_j)` for Si, O and H.
Differentiating this single scalar preserves reciprocity and extensivity.
This quadratic continuation reproduces dilute derivatives; it is not a unique
finite-concentration reconstruction of the measured activity curves.

| Term | Primary source and adopted result | Measured scope |
| --- | --- | --- |
| P gas to 1 wt% Henry P | [Yamamoto et al. (1980)](https://doi.org/10.2355/tetsutohagane1955.66.14_2032), Table 3: `-95.3 + 0.0155*T` kcal/mol | 1863.15–1923.15 K, 1–3 wt% P |
| Half P2 gas to 1 wt% Henry P | Same table: `-37.7 + 0.0013*T` kcal/mol | Alternative reference, not averaged with P gas |
| P-P | [Yamada and Kato (1979)](https://doi.org/10.2355/tetsutohagane1955.65.2_264), epsilon 7.3 | 1873.15 K |
| P-Si | [Yamada and Kato (1983)](https://doi.org/10.2355/isijinternational1966.23.51), epsilon 11.9 | 1873.15 K |
| O-P | [Sanbongi and Koizumi (1962)](https://doi.org/10.2355/tetsutohagane1955.48.14_1729), mass-percent coefficient 0.06 | 1813.15–1898.15 K; reported values 0.05–0.07 |
| H-P | [Nozaki et al. (1966)](https://doi.org/10.2355/tetsutohagane1955.52.13_1823), mass-percent coefficient 0.015 | 1723.15–1943.15 K, P below 6 wt% |

The caller supplies its actual P or P2 gas standard. The original one-atmosphere
gas convention and 1 wt% Henry convention are converted separately to the
one-bar, mole-fraction convention. No absolute elemental gauge is fitted.
Mass-percent interaction conversion includes the changing mean molar mass.
[phosphorus_sources.json](phosphorus_sources.json) records the numerical
transcription, source locations and retrieved PDF hashes.

At 2173.15 K these data require explicit temperature continuation. Constant
and inverse-temperature interaction continuations, P/P2 standards, and the
reported standard-fit shifts are separate conditional calculations. Neither
their spread nor the original fit's two-standard-deviation errors certify a
coupled extrapolation error. No calibrated pressure term is supplied.

The interval Hessian uses the same pure arithmetic excess expression as the
scalar. Weighted Gershgorin rows are rounded outward; numerical eigenvectors
only propose positive weights. The Fe ideal Hessian term is positive
semidefinite and can be discarded for a lower bound. This certifies curvature
of the declared scalar within the supplied box, not physical applicability.
New host interactions require a matched proof rather than inheriting this
certificate. The material assessment accepts complete five-component mass
fractions but does not relabel the four-component Kato reference as P-bearing.
