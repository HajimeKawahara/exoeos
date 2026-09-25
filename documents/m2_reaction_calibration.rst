Water and hydrogen reaction calibration references
==================================================

``examples/m2_material/water_calibration.py`` independently evaluates the
published Thompson et al. (2025) water law and replays its experimental
data. It also replays the available original Sossi et al. (2023) water
columns. ``reaction_calibration.py`` distinguishes these references from
the hydrogen standards currently adopted by the M2 consumer. Nothing in
this example changes native MELTS, the alloy model, or material admission.

Pinned observations and concentration conventions
-------------------------------------------------

The ``water_data`` directory preserves two original CSV files from
`Zenodo version 16418810 <https://doi.org/10.5281/zenodo.16418810>`_, attributed
to Maggie Thompson under CC-BY-4.0. SHA256 checks precede every replay.
``provenance.json`` records the paper and notebook hashes, coefficient
table, selection rule, and source locations. The large notebook and PDFs
are not redistributed. There is no download or refitting at runtime.

`Thompson et al. (2025) <https://doi.org/10.1016/j.chemgeo.2025.123048>`_
Eq.18 and Table4 give

.. math::

   q = A\exp\left(\frac{\sum_i b_i X_i}{T}\right)
       \sqrt{f_{\mathrm{H_2O}}/(1\ \mathrm{bar})}.

The implementation uses the rounded table values
:math:`A=7.19\times10^{-4}` and, in CaO, SiO2, MgO, Al2O3, FeO order,
:math:`b=(-1511.1,886.5,-1015.2,890.6,-1755.9)` K. The predictors are
dry oxide mole fractions; other oxides must not be renormalized away.
The pure algebra is exposed as ``thompson_oh_response`` and
``log_oh_capacity``, with JAX derivatives in their positive interiors.

The paper describes its response as a mole fraction, but the archived
fitting notebook defines ``XOH = OH_Conc_ppm / 1e6`` and converts its
prediction back to ppm by multiplication by one million. We reproduce
that numerical response, rather than silently treating it as a mole
fraction. ``water_reference`` returns values and derivatives with respect
to temperature, fugacity in Pa, and the five oxide fractions. The
composition derivatives hold the other fractions fixed; a normalized
composition path must apply its own chain rule.

There is a second convention to resolve before using this response for
atom budgets. If the original ``Sossi23_Data.csv`` H2O columns are read as
literal H2O mass ppm, the combined dataset's OH columns follow
:math:`M_{\mathrm{OH}}/M_{\mathrm{H_2O}}`, whereas counting two OH groups
per H2O gives :math:`2M_{\mathrm{OH}}/M_{\mathrm{H_2O}}`.
``sossi_basis_audit`` records both mappings for all twelve shared samples.
Sossi's Eq.1 uses :math:`M(\mathrm{H_2O})=0.018015` kg/mol for the
Beer--Lambert conversion. Clarification of the pooled spectroscopy and
equivalent-species conventions remains necessary; this audit neither
corrects the authors' data nor asserts an author error.

The independent ``sossi2023_water_response`` avoids that pooled conversion
and reproduces Sossi Eq.11 directly as water-equivalent mass fraction.
Its two named IR calibration branches remain separate. The H2-fugacity
term describes incorporation as OH, not molecular H2 dissolution. The
available CSV contains twelve of the original fourteen samples, so its
replay does not claim to reproduce the complete original experiment.
See `Sossi et al. (2023) <https://doi.org/10.1016/j.epsl.2022.117894>`_.

Reproduction and uncertainty
----------------------------

The combined CSV has 81 rows. Following the author's notebook, the replay
uses 74 positive-water-fugacity rows and preserves seven blanks separately.
It keeps observed scatter, including residuals beyond individual reported
measurement standard deviations. This is an independent implementation
on the fitting data, not held-out experimental validation or reproduction
of posterior samples. Rounded Table4 means give :math:`R^2=0.928696`,
RMSE 27.53 ppm in the author's OH response, and a maximum residual of
7.95 reported observation standard deviations. These agree with the
paper's approximately 0.93 coefficient of determination without claiming
that every observation is accepted.

Published coefficient marginal SDs are retained. A covariance matrix or
numerical joint posterior is not supplied by the downloaded dataset.
No independent-normal approximation is made. For
:math:`\ln C=\ln A+b\cdot X/T`, Cauchy--Schwarz gives the parameter-only
bound :math:`\sigma_{\ln C}\leq\sum_i|X_i|\sigma_{b_i}/T` for any covariance
with those marginals. This bounds a standard deviation; it is not a
confidence interval, a limit on realization errors, or a bound on
systematic or extrapolation error. Measurement correlations, absorption
calibration, the pooled atom conversion, and transfers in total pressure
and composition remain distinct unresolved contributions.

``water_reference`` defaults to a matching published predictor point at
1 bar total pressure. Any other evaluation requires
``allow_extrapolation=True`` and is labeled accordingly. A matching point
does not establish the accuracy of the fit or reproduce all gas-buffer
coordinates. The paper's approximate recommended T/P ranges are not
converted into a joint experimental rectangular domain. Total pressure
is not interchangeable with water fugacity and has no fitted correction.

An explicitly conditional Gibbs completion
------------------------------------------

``water_dissolution_state`` returns an extensive addition to a *dry* host
G and all its host/water derivatives. Its mandatory keyword is
``response_interpretation="literal_OH_group_mass"``. This explicitly
assumes the response represents literal OH-group mass; it does not settle
the pooled-data convention above. The function is never called by the
current M2 native-liquid provider.

The dry component order is CaO, SiO2, MgO, Al2O3, FeO, Na2O. The amount
of the added component is moles of H2O, with two H atoms and one added O
atom. If its amount is :math:`h`, then

.. math::

   w_{\mathrm{OH}}=\frac{2M_{\mathrm{OH}}h}{M_d+M_w h},\qquad
   \frac{\mu_w}{RT}=\frac{\mu^0_{w,g}}{RT}
          +2\ln(w_{\mathrm{OH}}/C)+\ln((1\ \mathrm{bar})/p^0).

One oxygen already in the dry host supplies the second OH group; no
additional OH reservoir is created. The amount cannot exceed the dry
host oxygen inventory. At fixed host, equality with the gas chemical
potential recovers the reference square-root fugacity response. The
scalar is twice the existing integrated mass-fraction mixing term plus
the composition-dependent water standard. Host derivatives include
:math:`-2h\partial\ln C/\partial n_i`; omitting these terms would violate
the Euler identity and derivative symmetry. Exactly zero water gives
zero energy/host shifts and a negative-infinite insertion potential.

This construction is an integrable mathematical completion, not a
calibrated full melt EOS or phase-stability result. It must not be added
to a wet native MELTS energy, which would duplicate the water physics.
Na2O participates in the dry denominator without a separately fitted
coefficient. Other oxide bases require an explicit new mapping and
experimental assessment, not dropping unmatched elements.

H2 and metal-H reaction standards
---------------------------------

``source_h2_standard_audit`` preserves the consumer's adopted
Hirschmann2012/Seo2024 molecular-mole-fraction law and checks
:math:`\Delta G^0/(RT)=-\ln K`. It is distinct from the existing
Chaudhari2025 buffered-pressure measurements. The latter's total-pressure
slope cannot be substituted for an arbitrary-fugacity Henry coefficient.
The separate 515 ppm/GPa pure-H2 illustration is converted into a
mass-basis standard solely as a descriptive comparison; it supplies no
experimental BSE error bound.

``kato_reaction_audit`` uses the existing measured-domain Kato1970
pure-Fe reference, retaining atomic H and the reaction
:math:`\tfrac12\mathrm{H_2(g)}=\mathrm{H(metal)}`. It reports a dimensionless
equilibrium constant, reaction free energy, temperature derivative, and
the equilibrium residual. The inherited M2 metal-H standard instead
comes from the Young/Okuchi exchange fit. ``audit_metal_water_exchange``
checks supplied, consistently transformed standard potentials against
that fit. It detects host-standard changes without inventing a MELTS
pure-oxide basis or a cross-phase calibration. Sources and the inspected
consumer revision are pinned in ``reaction_calibration_sources.json``.

Run and checks
--------------

From the checkout, with ExoEOS importable:

.. code-block:: console

   JAX_ENABLE_X64=1 python -m examples.m2_material.reaction_calibration \
       --output /tmp/reaction-calibration.json

The CLI refuses to overwrite an existing result. The report retains
source hashes, all experimental residuals, both original Sossi IR
branches, convention discrepancies, and unbounded uncertainties.
``reaction_calibration_validation.json`` is the archived replay. Tests
check raw-data hashes, source selection, derivative finite differences,
amount scaling, Euler consistency, reciprocal derivatives, the
water/H atom conversion under the explicit interpretation, equilibrium
recovery, zero-water behavior, and standard-pressure/element-gauge
invariance. No native MELTS run or new global closure is needed.

Common BSE material admission and both M2 scientific gates remain pending.
