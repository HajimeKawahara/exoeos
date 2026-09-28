# Sodium and helium transfer evidence

The earlier omitted-metal catalog covered Mg, Al, Ca, K, Ti, Cr and P.
Its completion by `associated_k` does not assess Na in metal, He in silicate,
or He in metal. These three paths remain explicit in the
[constitutive ledger](constitutive_evidence.json) for every selected model.
The finite source currently puts He only in gas. This is a model restriction:
chemical inertness does not establish zero solubility. No species or old
equilibrium result changes in this evidence update.

[The source record](na_helium_transfer_sources.json) retains primary locations,
PDF hashes, actual observation bases and unresolved quantities.

| Path | Retained primary evidence | Limit at 2173.15 K and about 270 bar |
| --- | --- | --- |
| Na to metal | [Gessmann and Wood (2002)](https://doi.org/10.1016/S0012-821X(02)00593-9), S-free Na-bearing runs at 1873–1973 K and 2.5 GPa are below detection | No numerical Na detection limit was identified in the retained paper. The 2173 K/24 GPa run is Na-free. Neither gives a quantitative low-pressure omission bound. |
| He to silicate | [Jambon et al. (1986)](https://doi.org/10.1016/0016-7037(86)90193-6) measured basalt solubility; [Guillot and Sator (2012)](https://arxiv.org/abs/1112.2055) simulated dry hosts up to 2273 K; [Iacono-Marziano et al. (2010)](https://doi.org/10.1016/j.chemgeo.2010.10.017) calibrated a composition-dependent law | Dry-host simulations can support a declared continuation; they do not bracket wet BSE empirically. The pooled 22% reference fit error does not bound the hot BSE extrapolation. |
| He to metal | [Bouhifd et al. (2013)](https://doi.org/10.1038/ngeo1959), Supplementary Table S2 reports simultaneous Fe80Ni20/silicate partitioning at 1.9–13.1 GPa and 2400–2600 K | Nonzero measured partitioning rules out an inference of guaranteed insolubility; its low-pressure Fe-Si-O-H continuation remains uncalibrated. Table S3 pure-Fe ratios use separately interpolated silicate concentrations. |

[Steenstra et al. (2018)](https://doi.org/10.1038/s41598-018-25505-6) supplies a
further quantitative Na candidate: its 1 GPa Fe-S/basalt exchange fit is
`log10 K_Na = 5.44 - 14340/T`. It assumes ideal silicate NaO0.5 activity
and adopts the K-S interaction for Na-S. The [Na reference replay](sodium_reference.md)
now retains five numerical S-free censoring limits with matching silicate
concentrations and compares two concentration conventions against three FeS
exchange constants. Converting to a selected shared source standard remains
work in progress; these are not low-pressure Na omission bounds.

## Small provider diagnostic

`helium_reference.helium_henry_coefficient(host, temperature_K)` returns the
mass-specific He capacity in mol/(kg dry host bar fugacity) for the specified
simulated host. It interpolates `ln(S)` against `1/T` between Appendix C points;
it rejects unknown hosts and temperatures outside that host's simulated
interval. It does not silently substitute a host for BSE. The STP conversion
explicitly assumes 273.15 K and 1 atm. The original `S` unit is cm3 STP/(g bar):

```text
a = 1000 S / V_STP           [mol / (kg dry host bar)]
n_He = M_dry a (f_He / bar)  [mol]
```

The second expression is a fixed-fugacity capacity. The consumer must use
the complete gas denominator, retain the finite total He inventory, and solve
the coupled pressure again before calling the response an omission error.
For a declared fixed dry-host coefficient, a possible extensive trace scalar
is `G_add/(RT) = n_He [mu_gas_standard/(RT) + ln(n_He/(M_dry a)) - 1]`,
where fugacity is measured relative to one bar. Its dry-host derivative must
be included as well as the He chemical potential. A composition-dependent
coefficient additionally requires its composition derivatives. This scalar
is implemented by `helium_dissolution.make_helium_dissolution` as a separate
provider scalar. ExoGibbs owns its opt-in finite-source connection and
ExoInventory owns the finite budget and pressure response.

Select `guillot2012_olivine`, `guillot2012_morb` or `guillot2012_rhyolite` and
provide the actual shared He gas standard plus the dry mass per host component.
Assign zero dry mass to native water and dissolved H2. `state` returns analytic
G/RT and all host/He potentials; `energy_value_and_grad_rt` independently
differentiates the scalar. Positive He requires positive dry-host mass. At
zero He the energy and host shifts vanish; its insertion potential is negative
infinity at positive dry mass and positive infinity at fixed zero dry mass.
The latter is a one-sided limit, not a joint smooth derivative. Thus a He-free
water-only host keeps its original energy and host potentials. An entirely
absent phase has zero energy and undefined
potentials. He contributes neither to the existing H2 mixing denominator nor
to the dry-host mass. This trace-law completion supplies no new pressure-volume
term or empirical finite-concentration bound.

The [frozen OH illustration](validation/20260928_na_helium/frozen_oh.json)
uses the archived 76-gas source, without changing it. For pHe = 45.8751 bar,
dry silicate mass 9.66026e22 kg and total He = 1e23 mol, the olivine/MORB/rhyolite
coefficients give 0.10895%, 0.20759% and 0.50731% of total He, respectively.
These are three declared dry-host illustrations, not a BSE uncertainty range
or a finite pressure response. The Na inventory is 1.16349e22 mol
(2.67483e20 kg), or 10.18% of the existing metal mass if all Na entered it;
small total abundance alone therefore does not certify negligible metal impact.

The saved replay verifies the input source SHA and reconstructs its gas
denominator and host mass. It needs no native provider or equilibrium solve.
He metal partition and Na metal standard/activities remain separate tasks;
no bounded error or automatic physical acceptance is assigned to them.
