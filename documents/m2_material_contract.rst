Common solution energy and the M2 material audit
================================================

ExoEOS supplies material energies and derivatives. ExoGibbs owns constrained
minimization, common reaction standards and phase selection. ExoInventory
owns absolute inventories and planetary closure. The material audit below
supports stages 2 and 3 of the BSE milestone; **M2-A and M2-B remain pending**.
An equation-consistent callback does not supply missing material calibration.

Full energy with supplied standards
----------------------------------------

``total_solution_gibbs_RT(model, T, P, n, mu0_RT)`` evaluates

.. math::

   G/(RT)=\sum_i n_i\mu_i^0/(RT)+\sum_i n_i\ln(n_i/N)+N g^E/(RT).

The inputs use K, Pa, mol of named components, and dimensionless standard
potentials at the same T/P as the excess model. The scalar result has units
of mol. ``total_solution_state`` returns ``TotalSolutionState(gibbs_RT,
mu_RT)``; the latter is dimensionless. Both accept JAX transformations and
retain the existing excess-model domain. The functions validate static
shapes only: the caller must validate physical inputs and material domains.

.. code-block:: python

   import jax.numpy as jnp
   from exoeos import MaFeSiOHLiquid, total_solution_state

   model = MaFeSiOHLiquid()
   T, P = 2173.15, 1e5
   n = jnp.array([0.90, 0.05, 0.03, 0.02])  # mol Fe, Si, O, H atoms
   model.validate_state(T, P, n / n.sum())
   source_mu0_RT = jnp.zeros(4)  # Explicit synthetic control, not calibration.
   formal_mu0_RT = source_mu0_RT + model.standard_state_shift_RT(T)
   state = total_solution_state(model, T, P, n, formal_mu0_RT)

Standards are mandatory. The helper does not infer the formal/source shift
or reconstruct absolute thermochemistry. Its ideal term uses the same R as
the model's excess term. A consumer must supply a consistent gas constant
and standard convention; changing a standard by an elemental gauge changes
G by a conserved constant, but cannot correct a reaction energy.

The whole-phase zero vector has exactly G=0 and undefined (NaN) chemical
potentials. At supported component boundaries, absent-component ideal
potentials are minus infinity. Zero amounts use the continuous energy
limit, without artificial traces. Full derivatives and Hessians are defined
only in the material's differentiable domain, with positive amounts for
the components differentiated. In particular, differentiating the scalar
at an entirely absent phase does not define an incipient composition; the
phase-selection solver must evaluate trial compositions separately.

Tests independently verify ideal-mixture potentials, finite differences,
Euler's relation, reciprocal derivatives, extensivity, gauge changes, dry
alloy limits, and exact whole-phase absence. They also exercise JIT/VMAP.

Conservative alloy curvature on a declared domain
--------------------------------------------------

``exoeos.ma_interval.ma_alloy_curvature_lower_bound(model, temperature_k,
lower, upper)`` supplies a global lower eigenvalue bound for the molar
ideal-plus-Ma energy :math:`g/(RT)`. Bounds follow Fe, Si, O, H atomic
fractions and intersect :math:`\sum_i x_i=1`. Eliminating Fe gives independent
coordinates :math:`z=(x_{Si},x_O,x_H)`; fixed solute coordinates are removed.
The result bounds the Hessian in these coordinates throughout the declared
domain. Linear standard potentials and elemental insertion costs contribute
no curvature.

.. code-block:: python

   from exoeos import MaFeSiOHLiquid
   from exoeos.ma_interval import ma_alloy_curvature_lower_bound

   lower = [0.86, 0.0, 0.0, 0.0]
   upper = [1.0, 0.08, 0.02, 0.04]
   bound = ma_alloy_curvature_lower_bound(
       MaFeSiOHLiquid(), 2173.15, lower, upper,
   )
   # Approximately 7.95567654: positive curvature for this restricted model.

This Fe-rich box is chosen for a mathematical control, not experimental
calibration. The helper accepts the exact built-in MaFeSiOHLiquid and
MaFeSiOLiquid types; subclasses that could change the scalar are rejected.
Custom dry interaction coefficients are included in the bound but remain
uncalibrated model variants. This eager NumPy calculation is separate from
the JAX state evaluator and has no pressure argument because the current
Ma excess scalar has no pressure dependence.

Second-order interval differentiation encloses the existing Ma excess
Hessian on the enclosing Si/O/H box. The ideal Hessian is

.. math::

   H_{ideal}=\operatorname{diag}(1/x_{Si},1/x_O,1/x_H)
             +(1/x_{Fe})\mathbf{1}\mathbf{1}^{\mathsf T}.

For a lower bound, the calculation replaces each diagonal by
:math:`1/u_i`, where :math:`u_i` is its declared upper fraction, and drops
the positive-semidefinite Fe term. Gershgorin row bounds then enclose the
smallest eigenvalue of the complete Hessian. Basic binary64 operations are
widened outward. Logarithms use exact binary range reduction and a
convergent series with an explicit geometric remainder, without assuming a
platform-specific logarithm error. Composition samples only test the
implementation against independent JAX Hessians; they do not supply the
global bound.

A nonnegative bound establishes convexity for this scalar on the declared
domain, including its continuous zero-solute energy limits. It can support
a consumer's tangent-plane certificate once the minimizer and its optimality
conditions are checked. It does not itself minimize insertion cost, certify
metal absence, or address another phase's stability. Negative bounds are
inconclusive: interval overestimation can require a narrower domain even
for a convex model. Boxes crossing pure-H or dry pure-solute singularities,
nonfinite bounds, infeasible inputs, and complex inputs raise exceptions.
Narrowing a domain to obtain a bound does not justify excluding physically
possible alloy compositions. Material acceptance remains pending.

Pinned BSE liquid evaluation
----------------------------------------

``examples/m2_material/bse_inventory.json`` is the stage-1 ExoInventory
ledger exported from commit ``0c604bcb7ccd850e9e747fb9fb16448e33d22e25``.
``bse_source.json`` preserves its source hash and Table 4, page 237, column 1
of `McDonough and Sun (1995) <https://doi.org/10.1016/0009-2541(94)00140-4>`_.
The retained projection removes MnO/NiO and keeps Al/Ca/K/Ti/Cr/P active in
the liquid. Total Fe is initially FeO-equivalent with ferric fraction zero.
This is an input oxygen convention, not Earth's measured ferric fraction.

The property audit scales the existing dry ledger to 100 g by a single
factor, preserves oxide moles and every retained atom, and maps it through
the pinned MELTS component matrix. It does not reconstruct an inventory or
add the independent H/He budgets to the dry liquid. Those remain in the
exported total ledger. Conventional atomic masses define the 100 g physical
mass; MELTS' rounded masses give 100.000436337 g for those same atoms.
The backend conversion retains that distinction instead of rescaling atoms.

The evaluated point is 2173.15 K and 1 bar. The official
`MELTS guidance <https://melts.ofm-research.org/>`_ gives nominal limits of
500--2000 degrees C and 0--2 GPa, with composition-dependent accuracy.
`Sossi et al. (2020) <https://doi.org/10.1126/sciadv.abd1387>`_ report complete
fusion of KLB-1-derived Mg-Si-Fe-Al-Ca samples at 1900 +/- 50 degrees C and
1 bar. This supports a BSE-like **candidate evaluation condition**, not a
stable-phase certificate for the exact retained source composition, its
redox/water changes, or the planned Mg/Si variants. The old 2350 K MORB
condition is not relabeled as BSE validation. No mineral assemblage has
been minimized here.

The saved ``bse_liquid_validation.json`` contains 22 fresh native evaluations:
one state, its factor-2.5 amount scaling, and central energy differences for
all ten present components. The maximum derivative error was 0.003095 J/mol,
below the declared 0.02 J/mol threshold. Full G, mu, standards, element
amounts, finite-difference steps, backend hashes and code provenance are
retained. This extends equation checks to the BSE input without changing
the existing MORB/carbon reference files. The evaluator now explicitly
returns ``gibbs_RT`` using its recorded common R, alongside ``gibbs_J``.

Reproduce from a source checkout with the separately installed runtime:

.. code-block:: console

   python examples/m2_material/validate_bse.py \
     --runtime /path/to/alphamelts-py-2.3.2-ubuntu_22_04-x86_64 \
     --python /path/to/melts-env/bin/python \
     --output /tmp/bse-liquid-validation.json

The output path must not exist. See :doc:`melts_liquid_evaluator` for pinned
runtime preparation, worker isolation and the complete output contract.
No ExoInventory import or MELTS installation is required for ordinary tests.

Material gates and omissions
----------------------------------------

``examples/m2_material/material_contract.json`` records units, models,
evidence, missing inputs, ownership, and the unresolved common domain.
The following limitations prevent scientific acceptance:

* MELTS has full supplied-liquid G and mu, but cross-phase reaction cycles
  with gas and alloy standards still need independent validation.
* The Fe-Si-O-H model is the integrable no-H-excess-interaction control.
  It supplies neither H partition calibration nor a pressure response.
  Positive T/P validation is mathematical, not a calibrated domain.
* Molecular H2 augmentation must retain reciprocal host dilution and its
  declared concentration basis. Native MELTS H2O already contains its own
  thermodynamics; adding a second water-solubility correction double counts
  it. The `Chaudhari et al. (2025) experiments
  <https://doi.org/10.1007/s00410-025-02272-y>`_ do not establish calibration
  at this dry-BSE candidate temperature and composition.
* Omitted Mg uptake into alloy cannot be bounded by a zero in a model
  excluding Mg. The `Badro et al. (2018) experiments
  <https://doi.org/10.1029/2018GL080405>`_ concern much higher T/P than this
  liquid point and do not quantify its omission here. Al/Ca/background
  evaporation and metal transfer also need supported bounds.
* Column temperature rules, geometry, pressure range, and shared-species
  contact are consumer conditions. A one-temperature reference does not
  provide continuous standards through the atmospheric column.

The machine-readable audit proposes separate physical error budgets: 1%
atmospheric mass, 1% of each element's total amount for changed partition,
and 0.01 dex in the hydrogen amount of the metal boundary. These are
explicit engineering targets, not empirical uncertainty estimates. Actual
omission effects remain unevaluated and the targets unaccepted. Evaluating
them requires a supported expanded-channel or bounded comparison, including
incipient alloys on the metal-free side. Numerical residual tolerances do
not establish these physical bounds. Missing boundaries or a vanished
atmosphere require a separate decision rather than division by zero.

An initial conservative inventory bound is also recorded: all Al/Ca/K/Ti/Cr/P
atoms together represent 5.317% of dry rock mass, and Mg represents 22.830%.
K alone represents 1.712% of the baseline's initial H+He mass. These maxima
cannot establish that their transfer is small. They bound only the named
atoms' mass, excluding associated O/H and any chemical or phase-boundary
response; they are not atmospheric error estimates.

There is currently no accepted coupled material domain. The artifact keeps
that field null and scientific gates pending, while the scalar/derivative
contract and the native BSE equation audit remain usable for further work.

Source-specific domain decision
----------------------------------------

The 2026-09-21 review is recorded in
:download:`domain_evidence.json <../examples/m2_material/domain_evidence.json>`.
Each entry separates the source's conditions, supported claim and missing
evidence. Numerical evaluation, a mathematical curvature box and experimental
calibration are distinct fields in
:download:`material_contract.json <../examples/m2_material/material_contract.json>`.
The review retains **2173.15 K and 100000 Pa as a numerical control point**;
it establishes no coupled physical domain. This conclusion describes the
available evidence, not a proof that no such domain can exist.

.. list-table:: Evidence relevant to the selected point
   :header-rows: 1
   :widths: 22 35 43

   * - Evidence
     - Conditions or composition
     - Consequence for this implementation
   * - MELTS guidance
     - Nominal 773.15--2273.15 K and 0--2 GPa
     - Candidate point is inside nominal limits; exact BSE liquid stability
       still needs a competing-phase calculation and material validation.
   * - Sossi et al. (2020)
     - Synthetic KLB-1-like melt, 1900 +/- 50 degrees C, 1 bar
     - Similar T/P does not make this reduced experimental composition
       identical to the retained McDonough-Sun inventory.
   * - Chaudhari et al. (2025)
     - Buffered Table 2 hosts over 1473.15--1673.15 K and 0.5--4 GPa
     - Candidate lies outside this envelope; preliminary pure-H2 runs are
       recorded separately and do not extend the buffered fit.
   * - Marcum et al. (2026)
     - Supercritical MgSiO3--MgSiO3H4; table starts at 3000 K
     - No temperature overlap with nominal MELTS; not a replacement
       molecular-H2 dissolution standard.
   * - Implemented Ma Fe-Si-O-H control
     - Integrable scalar, ideal H dilution, no pressure dependence
     - Positive curvature supplies mathematical evidence only.
   * - Mg uptake and background vapor
     - Different high-T/P experiments and dry-BSE calculations
     - Transfer errors in this H-rich inventory remain unbounded.

The `new H2 measurements
<https://link.springer.com/article/10.1007/s00410-025-02272-y>`_
report molecular H2 in mass ppm. Their buffered concentration/total-pressure
fits cannot be inserted as arbitrary-fugacity standards. Converting mass
concentration to a MELTS component basis does not remove the temperature,
pressure and host-composition extrapolation. Native MELTS water remains a
separate contribution.

The `Marcum et al. EOS <https://arxiv.org/html/2608.27401v1>`_ describes a
supercritical binary whose hydrogen need not remain molecular. Its table's
3000--10000 K and 1--800 GPa coordinates are not a calibration rectangle
for BSE or for H2 dissolution. Replacing the current material with that
binary would change the physical model, rather than close the present gate.

For alloy H, `Jiang et al. (2025)
<https://www.eppcgs.org/en/article/doi/10.26464/epp2025055>`_ find a
composition effect on partitioning at 5000 K and 135 GPa. This cannot
calibrate the present low-pressure scalar. Likewise, the lack of resolved
pressure dependence in the Mg partition fit of Badro et al. is not evidence
for pressure-independent Fe-Si-O-H activities. The background vapor
channels in `Fegley et al. (2023) <https://arxiv.org/pdf/2305.13327>`_ use a
different BSE composition and do not bound their omission here.

A fixed-bottom-pressure column may use the selected point to check provider
connections, amount mapping and conservation, while carrying the pending
material gates. Extending bottom pressure, liquid composition or temperature
requires new material evidence. Neither a successful column solve nor
numerical closure promotes M2-A or M2-B to scientific acceptance.
