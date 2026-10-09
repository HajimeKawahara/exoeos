Choosing and using a model
==========================

Start with the quantity you need, then check composition, standard-state
conventions, and the model's domain. This guide covers the source baseline
in :doc:`overview`; it is not a claim that all models apply at the same state.
The :doc:`feature_plots` gallery shows representative numerical curves for
these paths and records the conditions and standards behind each plot.

Choose an entry point
---------------------

.. list-table:: Output-driven selection
   :header-rows: 1
   :widths: 28 37 35

   * - I need
     - Entry point
     - Before calling
   * - Fluid density and fugacity at T/P
     - ``state_tp(eos, T, P, x, phase=...)``
     - Select a TP-capable residual model and its allowed root policy.
   * - Fluid pressure and fugacity at known molar density
     - ``state_trho(eos, T, rho, x)``
     - Supply molar density in mol/m3 and normalized mole fractions.
   * - Excess Gibbs and activities
     - ``solution_state(model, T, P, x)``
     - Match the model's component order and symmetric standards.
   * - Full solution G and component potentials
     - ``total_solution_state(model, T, P, n, mu0_RT)``
     - Supply amounts in mol and compatible standard potentials.
   * - A mass-based solute in an existing host
     - ``mass_fraction_solute_state(...)``
     - Supply consistent host G/mu, masses, and a host-composition-independent
       solute standard; retain the host derivative correction.
   * - Native MgO--SiO2 liquid and forsterite G
     - ``exoeos.magma.liquid_gibbs`` and ``forsterite_gibbs``
     - K, Pa, mol; liquid order SiO2, Mg2SiO4. Enable x64 and read the
       restricted equilibrium scope in :doc:`native_magma`.
   * - H/He caloric table properties
     - ``table.state_tp(T, P)`` or ``table.state_trho(T, rho_mass)``
     - Select a fixed Chabrier--Debras variant and load its verified tables.
   * - Silicate--hydrogen table properties
     - ``table.state_tp(T, P, x)``
     - Use the MgSiO3, MgSiO3H4 endmember order and verified Marcum table.
   * - Ideal-gas enthalpy, entropy and heat capacities
     - ``IdealGas(...).state(T, P, x)``
     - Supply molar masses, constant component cp and the required references.
   * - Fluid caloric properties, sound speed and RCE responses
     - ``HelmholtzThermodynamics(residual, ideal).state_tp(T, P, x)``
     - Supply a compatible ideal free energy and molar masses; responses
       hold composition fixed. See :doc:`thermodynamic_derivatives`.
   * - Density for an atmospheric column
     - ``MassDensityProvider.mass_density_tp(...)``
     - Choose the appropriate provider and match the species/mass ordering.

Residual fluids
---------------

All four models share the residual potential
:math:`\alpha^r=A^r/(nRT)`. Pressure inversion belongs to the model;
the common ``state_tp`` then obtains all residual fields from the same
``state_trho`` calculation.

.. list-table:: Residual models
   :header-rows: 1
   :widths: 23 37 40

   * - Model
     - Inputs and root policy
     - Applicability and next reading
   * - ``IdealEOS``
     - Zero residual energy; ideal density inversion.
     - Useful as a control. Use ``IdealGas`` for caloric quantities.
   * - ``SecondVirialEOS``
     - Constant symmetric B_ij in m3/mol; ``phase="vapor"`` only.
     - Low-density truncation. Require a strictly positive inversion
       discriminant and negligible omitted higher virials.
   * - ``PengRobinsonEOS``
     - Critical properties, acentric factors and optional k_ij;
       largest/smallest physical root for vapor/liquid.
     - Root selection is not phase coexistence. Default zero k_ij is a
       modeling assumption. See :doc:`tutorials/peng_robinson_eos`.
   * - ``ZhangDuanEOS``
     - Published species parameters and mixing rules;
       ``phase="vapor"`` selects the first mechanically stable root
       connected to the low-density branch.
     - A homogeneous fluid model, including dense states. Source ranges
       differ by species and mixture; see :doc:`tutorials/zhang_duan_eos`
       and :doc:`tutorials/zhang_duan_high_pressure_h2o_co2`.

A minimal density/fugacity comparison uses the same state contract:

.. code-block:: python

   import jax.numpy as jnp
   from exoeos import IdealEOS, PengRobinsonEOS, get_critical_properties, state_tp

   co2 = get_critical_properties("CO2")
   eos = PengRobinsonEOS(
       [co2.critical_temperature], [co2.critical_pressure], [co2.acentric_factor]
   )
   x = jnp.array([1.0])
   ideal = state_tp(IdealEOS(), 400.0, 1.0e6, x)
   fluid = state_tp(eos, 400.0, 1.0e6, x, phase="vapor")
   density_mol_m3 = fluid.rho
   log_fugacity_coefficients = fluid.lnphi

For quantitative reference comparisons, use
:doc:`tutorials/peng_robinson_fixed_state_reference` and
:doc:`tutorials/fixed_composition_cho_eos_comparison`. The curated critical
property records cover CO, H2O, CO2, H2, CH4, N2, NH3, H2S and SO2; their
availability does not establish a calibrated multicomponent mixture.

Solutions and standards
-----------------------

``solution_state`` returns the excess part, whereas ``total_solution_state``
adds supplied standards and ideal mixing exactly once:

.. math::

   \frac{G}{RT}=\sum_i n_i\frac{\mu_i^0}{RT}
                 +\sum_i n_i\ln x_i+\frac{G^E}{RT},
   \qquad
   \frac{\mu_i}{RT}=\frac{\mu_i^0}{RT}+\ln x_i+\ln\gamma_i.

.. list-table:: Solution models and evidence
   :header-rows: 1
   :widths: 24 36 40

   * - Model
     - Composition and purpose
     - Boundary
   * - ``IdealSolution``
     - Zero excess energy for any supported component count.
     - Full ideal mixing is added by the total-solution helper.
   * - ``MaFeSiOLiquid``
     - Fe, Si, O atomic mole fractions; Fe-rich alloy activities.
     - Requires Fe > 0. No pressure response or phase selection;
       :doc:`fe_si_o_reference` specifies the source/endmember conversion.
   * - ``MaFeSiOHLiquid``
     - Fe, Si, O, H atomic mole fractions; dry Ma model plus ideal H dilution.
     - No H excess interaction. ``standard_state_shift_RT`` is a convention
       conversion, not an absolute H standard; see :doc:`fe_si_o_h_reference`.

This example isolates the amount/standard contract with **synthetic zero
standards**; it is not a measured alloy calibration:

.. code-block:: python

   import jax.numpy as jnp
   from exoeos import MaFeSiOHLiquid, total_solution_state

   model = MaFeSiOHLiquid()
   n = jnp.array([0.80, 0.08, 0.04, 0.08])  # mol of Fe, Si, O, H
   model.validate_state(1873.0, 1.0e5, n / n.sum())
   state = total_solution_state(model, 1873.0, 1.0e5, n, jnp.zeros(4))
   gibbs_over_RT_mol = state.gibbs_RT
   chemical_potentials_over_RT = state.mu_RT

An absent phase has zero G and undefined (NaN) chemical potentials. At a
supported zero-component boundary of a present phase, the ideal contribution
gives minus infinity for the absent component's potential. No trace floor is
inserted. Differentiate only on a supported positive component face.

For a solute specified in complete-liquid mass fraction, use the separate
``mass_fraction_solute_state`` construction in :doc:`m2_material_contract`.
It adds a consistent host response as well as the solute potential. Do not
duplicate host mixing or native MELTS water thermodynamics. Atomic H,
molecular H2, water-equivalent ppm and endmember mole fractions require
explicit conversions; :doc:`m2_reaction_calibration` records those conventions.

Tables, caloric quantities and density
--------------------------------------

.. list-table:: Table paths use dedicated states
   :header-rows: 1
   :widths: 24 38 38

   * - Model
     - State and loading
     - Domain behavior
   * - Chabrier--Debras
     - ``MassThermodynamicState``: mass density, specific u/s,
       tabulated logarithmic derivatives and adiabatic gradient.
       ``ChabrierDebrasTableLoader(...).load()`` verifies the table pair.
     - Y0275, Y0292, Y0297 are discrete published variants; no continuous
       Y interpolation. Outside the nominal grid the state is NaN.
       A finite value inside the rectangle is not a physical-validity mask.
   * - Marcum--Stixrude--Young
     - ``SilicateHydrogenState``: mass density, specific h/s/cp and
       responses. ``MarcumSilicateHydrogenTableLoader().load()`` verifies
       the composition-dependent CSV.
     - MgSiO3--MgSiO3H4 table coordinates span 3000--10000 K and 1--800 GPa,
       with missing low-pressure cells above 6000 K. Required missing
       corners or extrapolation give an all-NaN state.

Table loading, downloads and checksums run on the host, before JAX tracing.
Subsequent interpolation uses the loaded arrays. The tabulated fields do not
define a residual Helmholtz model or component chemical potentials. In
particular, this supercritical silicate--hydrogen table is not a molecular-H2
solubility law for a cooler BSE melt.

For fixed-composition H/He potential derivatives, explicitly convert the
original model with ``table.to_helmholtz()`` in JAX 64-bit mode. The new
``HelmholtzTable`` derives a mass-specific state from one C2 potential and
retains source residuals and stability as separate checks. It does not
replace the original backend; see :doc:`potential_tables` for construction,
the explicit TP density bracket, and measured limitations.

``IdealGas`` provides a separate analytic caloric state with molar h/s/cp/cv.
It also supplies ``molar_helmholtz`` for the
:doc:`total Helmholtz derivative engine <thermodynamic_derivatives>`, which
combines an ideal closure with residual models and generates caloric and
response quantities from one potential.
For density alone, ``mass_density_tp`` converts a residual model's molar
density using supplied molar masses. ``TPHelmholtzDensityProvider`` wraps that
path; ``FixedCompositionDensityProvider`` checks the requested composition
against a fixed table mixture. ``AdditiveVolumeCompositeDensityProvider``
groups components with explicit species metadata and applies
:math:`1/\rho_{mass}=\sum_i w_i/\rho_i`.

Checkout tools: MELTS and M2
----------------------------

These tools live under ``examples/`` rather than the top-level package API.
Use a checkout at the overview's source revision. MELTS calls additionally
require the pinned external runtime described in :doc:`melts_liquid_evaluator`.
Ordinary package imports and stored-reference tests do not require that runtime.

.. list-table:: Tools, outputs and supporting records
   :header-rows: 1
   :widths: 24 41 35

   * - Tool family
     - Available result
     - Detail and limitation
   * - Native MELTS liquid and candidates
     - Supplied-liquid G/mu, native standard states, candidate compositions
       and amount/basis checks.
     - :doc:`melts_liquid_evaluator`, `candidate basis guide`_. Mode 4 adds
       carbon; S/halogen placeholders remain unsupported and N is absent.
       See :doc:`cns_provider_scope`.
   * - Published liquid mixing
     - Explicit regular-solution/water mixing scalar; a callback combining
       that expression with native pure standards and a derivative audit.
     - `Liquid mixing guide`_. Subsequent composition evaluations do not
       acquire caloric or density fields from a different native probe.
   * - Candidate-solution declarations
     - Site entropy, interaction polynomials, ordering variables, endmember
       maps and domains for 20 native solution models, including Fe/Ni
       solid and liquid alloys distinct from the Ma Fe--Si--O--H model.
     - `Solid mixing guide`_. Signed coordinates and ordering minima need
       their declared domains. Plagioclase has binary-specific provenance.
   * - Major-gas residual potential
     - A full-catalog Gibbs/fugacity/density model using the core virial EOS;
       declared H2--He and water-cross alternatives.
     - `Major-gas guide`_. Every trace pair is explicitly zero in this model;
       trace species still have a dilution/fugacity response. Implementation
       range 1000--3000 K is distinct from source calibration ranges.
   * - Material references and assessment
     - Experimental replays, conditional water/H/O/Mg/Na/K/P/He models,
       reaction conventions and actual-state evidence reports.
     - :doc:`m2_material_contract`, :doc:`m2_reaction_calibration`,
       `material evidence guide`_. Scenario contrasts are not automatically
       empirical uncertainty bounds.
   * - Ma alloy interval bounds
     - ``exoeos.ma_interval`` supplies curvature and fixed-plane insertion
       bounds using outward interval arithmetic.
     - :doc:`m2_material_contract`. A negative curvature lower bound is
       inconclusive; it does not prove instability. This is a package
       submodule, not an examples-only dependency.

Full native MELTS G/mu already include the native mixing and pressure terms.
Adding a separate ideal or excess term would double count them. The published
mixing callback is an explicitly selected model, with its own provenance;
finite native comparisons do not prove global equivalence to the binary.
Phase selection and certificates belong to the chemical consumer.

Numerical use and reproducibility
---------------------------------

Residual, total Helmholtz and solution entry points evaluate one scalar state;
use ``jax.vmap`` for batches. ``IdealGas.state`` supports native broadcasting with the
last axis reserved for components. Keep phase strings static under ``jax.jit``.
JAX differentiation follows the model's smooth domain: root degeneracies,
table-cell boundaries and unsupported composition endpoints need separate
care. External native workers are not JAX-traceable evaluators.

For each reported calculation retain the source commit, ordered component
basis, T/P and units, selected model and root, standard-state convention,
data/runtime hashes where applicable, and the distinction between numerical
checks and physical applicability. Preserve earlier test counts and results
as historical records rather than relabeling them as checks of a newer commit.

.. _candidate basis guide: https://github.com/HajimeKawahara/exoeos/blob/f049cfcb3f42e198124c8f6c921385ff0d67f69d/examples/m2_candidate_basis/README.md
.. _liquid mixing guide: https://github.com/HajimeKawahara/exoeos/blob/f049cfcb3f42e198124c8f6c921385ff0d67f69d/examples/m2_liquid_mixing/README.md
.. _solid mixing guide: https://github.com/HajimeKawahara/exoeos/blob/f049cfcb3f42e198124c8f6c921385ff0d67f69d/examples/m2_solid_mixing/README.md
.. _major-gas guide: https://github.com/HajimeKawahara/exoeos/blob/f049cfcb3f42e198124c8f6c921385ff0d67f69d/examples/m2_material/major_gas_eos.md
.. _material evidence guide: https://github.com/HajimeKawahara/exoeos/blob/f049cfcb3f42e198124c8f6c921385ff0d67f69d/examples/m2_material/constitutive_evidence.md
