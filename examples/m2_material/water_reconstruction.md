# Reconstructed water basis and independent pressure comparison

`water_basis_pressure.py` converts the Thompson2025 response to H2O-equivalent
mass using `response * M_H2O / M_OH`. [Sossi2023, Eq1](https://doi.org/10.1016/j.epsl.2022.117894)
defines the inherited 3550 cm-1 absorptivity per H2O-equivalent mole;
[Thompson2025, Eq1](https://doi.org/10.1016/j.chemgeo.2025.123048) multiplies
that signal by M_OH. Undoing this factor does not require a second factor
of two. The reconstruction returns the original twelve Sossi water columns
within 4.4e-8 mass ppm. Absorptivity calibration errors remain separate.
The earlier literal-OH completion and archived results are preserved.

The complete 14 observations at 500/1000 bar from
[Shishkina2010, Table2](https://doi.org/10.1016/j.chemgeo.2010.07.014)
are an independent comparison at 1523.15 K, without coefficient fitting.
All ten starting oxides remain in the predictor denominator. Missing analyses,
measured zeros, fluid-fraction errors, Fe loss, and Fe redox are retained.
KFT and NIR analyses of one capsule are not independent experiments.

| Gas treatment | KFT count | Prediction / KFT | KFT RMSE (wt%) | Largest multiplicative discrepancy |
|---|---:|---:|---:|---:|
| Ideal partial pressure | 14 | 0.6204–1.1029 | 0.4196 | 1.6119 |
| Zhang–Duan2009 H2O–CO2 | 14 | 0.6192–1.1011 | 0.4199 | 1.6151 |

For Zhang–Duan, the 12 NIR total-water comparisons have maximum discrepancy
1.5274; OH alone is overpredicted by factors 1.167–1.642. Quenched-glass
speciation is not an in-situ melt equilibrium constraint. Neither gas treatment
reproduces the original Pitzer–Sterner/Aranovich–Newton recipe. These observed
discrepancies are not confidence bounds for BSE at 2173.15 K and about270 bar.

`examples/melts_water_reconstruction.py` supplies the opt-in model
`dry_melts_thompson2025_water_equivalent_v1`. It evaluates published MELTS
with zero native water, then adds the integrated mass-fraction water law:

```
mu_water / RT = mu_H2O_gas_standard / RT + 2 ln(w_H2O / C_H2O)
G_added / RT = n_water (mu_H2O_gas_standard / RT - 2 ln C_H2O)
               + 2 G_mass_fraction_mixing / RT
```

The gas standard uses the consumer's exact element gauge at1 bar. Every
host derivative of C is included. Ferric iron is recast as2FeO only in the
empirical predictor; actual masses and atom ledgers retain their native basis.
No native water interaction or standard remains in the dry-host term.
The model assumes zero water pressure-volume correction, as in the published
capacity law; this assumption is not a measured error bound. Explicit OH/molecular
H2O speciation, molecular H2 calibration, and alloy calibration are separate.

Against the old literal-OH completion, this basis change alone is
`-2 n_water ln(2)` in G/RT. It is **not** that constant correction to native
MELTS. Selecting the full dry-host replacement changes host composition
responses and requires fresh equilibrium and phase-stability calculations.
The old second-liquid bound is not inherited. Native and published models
remain available, and the coupled material domain remains unestablished.

[Saved validation](validation/20260928_water_reconstruction/execution.json)
includes both unfitted replays and one evaluation at the original OH source
composition. The latter has zero reported active derivative discrepancy and
Euler residual 1.68e-16; it is not a newly solved source or pressure closure.
The source table and publication locators are recorded in
[provenance](water_data/water_basis_pressure_sources.json).

Saved results include the exact binary64 coefficient arrays in
`water_reconstruction.expression` and the unmodified dry-host result in
`water_reconstruction.dry_properties`. Independent stability audits can
reconstruct the scalar using those recorded values. This metadata does not
reuse the hydrated MELTS supporting-plane result or establish a new bound.
