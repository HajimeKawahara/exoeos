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
