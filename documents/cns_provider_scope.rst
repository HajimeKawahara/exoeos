C/N/S provider responsibilities
================================

This implementation follows the 2026-09-12 ExoInventory assessment. ExoEOS
owns phase properties, activities, fugacities, their thermodynamic potential,
and the associated component and standard-state conventions. ExoGibbs owns
local chemical reactions, dissolution, partition, and phase selection.
ExoInventory owns absolute reservoirs and atmospheric pressure closure.

Implemented provider steps
--------------------------

The independent `CNS gas-property PR
<https://github.com/HajimeKawahara/exoeos/pull/22>`_ extends the existing
critical-property table with CH4, N2, NH3, H2S, and SO2. The existing
Peng--Robinson constructor accepts these records without a new adapter or
species registry. Pure-fluid critical properties do not calibrate mixture
interactions; the default zero binary coefficients remain an explicit
approximation. HCN is not included in the curated table.

The supplied-liquid :doc:`melts_liquid_evaluator` now offers an explicit
``calculation_mode=4`` carbon path (rhyolite-MELTS 1.2.0). Its 19 independent
input components retain the previous order. The internal carbonate species
does not introduce another independent reservoir. Positive CO2 is rejected
in the default 1.0.2 mode; SO3 and formal halogen placeholders are rejected in
both supported modes. Nitrogen is absent from the component basis.

The optional ``tests/reference/generate_melts_carbon.py`` command saves a
separate ``melts_carbon_v1.json`` fixture. It checks a zero-carbon control and
three positive-carbon liquids at 1473.15/1673.15 K and 50/500 MPa, using the
first saved MORB liquid as a fixed composition seed. Only the independent
CO2 amount changes. This adds one C and two O atoms per CO2; it does not
implicitly prescribe an oxygen fugacity. Every present independent chemical
potential is checked against a fresh finite difference of the extensive
Gibbs energy. Scaling by 0.1 and 2.5 checks extensivity and invariant chemical
potentials. The original zero-C reference archive remains unchanged.

These are equation-consistency controls, not calibrated equilibria. In
particular, they do not validate the 2350 K finite-MELTS experiment, its
gas/alloy standards, or the stability of its chosen phases. The original
``calculation_mode=1`` and new ``calculation_mode=4`` are different models;
the exact-zero carbon control belongs to mode 4, without requiring numerical
equality to mode 1.

Remaining physical inputs and ownership
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 50 35

   * - Connection
     - Required physical input
     - Responsible layer
   * - Melt C exchange
     - Align full MELTS component potentials with gas/alloy standards;
       supply reduced CO/CH4 dissolution with a defined amount basis if used.
     - ExoGibbs reaction model; ExoEOS supplies the declared liquid potentials.
   * - Metal C
     - Establish intrinsic activity versus partition definitions, standards,
       reciprocal derivatives, and graphite/carbide alternatives.
     - ExoEOS for a justified scalar alloy model; ExoGibbs for phase chemistry.
   * - Melt/metal S
     - Calibrate unsaturated exchange for the selected host; distinguish
       dissolved silicate S, metal S, and a separate sulfide phase.
     - ExoGibbs exchange and saturation; ExoEOS only for intrinsic phase properties.
   * - Melt/metal N
     - Define total-N speciation, redox/host exchange, metal N and nitride
       treatment or an omission bound.
     - ExoGibbs chemistry; a new alloy potential requires independent support.
   * - Joint C/N/S alloy
     - Supported cross interactions or explicit uncertainty controls derived
       from a common scalar potential.
     - ExoEOS after the separate limits and thermodynamic definitions are established.
   * - Planetary amounts
     - Exact-zero support, all-element conservation, atmospheric column mass,
       and a freshly evaluated pressure root.
     - ExoInventory using ExoGibbs local chemistry.

The pinned GCE source controls provide empirical reaction corrections, not a
common multicomponent alloy Gibbs energy. The Carbon and Sulfur/Nitrogen
prescriptions differ; the sulfur correction also depends on the silicate
host. They cannot be appended as independent activity coefficients to
``MaFeSiOHLiquid``. Likewise, a total elemental-N solubility law does not
identify a molecular N2 component or its full potential. No C/N/S alloy
scalar, saturation law, or new generic reservoir API is introduced here.

The audited source implementations are retained in the
`ExoGibbs source control
<https://github.com/HajimeKawahara/exogibbs/blob/5d95bd54b1880f7eb01792f104d94e16f0538be9/examples/metal_silicate/sulfur_source.py>`_
and its
`reference audit
<https://github.com/HajimeKawahara/exogibbs/blob/5d95bd54b1880f7eb01792f104d94e16f0538be9/examples/metal_silicate/sulfur_reference.json>`_.
Their numerical reproduction does not establish experimental calibration of
the MELTS branch. Full MELTS potentials already contain all mixing and
pressure terms supplied by that model; adding a separate empirical reaction
offset requires a declared ExoGibbs reaction model, not a silent change to the
full-potential callback.
