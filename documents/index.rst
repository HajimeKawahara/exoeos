ExoEOS
======

ExoEOS provides differentiable equations of state and excess free-energy
models for planetary atmospheres, fluids, and melts, powered by JAX.

Tutorials
---------

.. toctree::
   :maxdepth: 1

   tutorials/peng_robinson_eos
   tutorials/zhang_duan_eos
   tutorials/fixed_composition_cho_eos_comparison
   tutorials/peng_robinson_fixed_state_reference
   tutorials/zhang_duan_high_pressure_h2o_co2

API overview
------------

See the project `README <https://github.com/HajimeKawahara/exoeos>`_ for the
current public API and installation instructions. Exact units, shapes, and
validity behavior are recorded in the
:download:`thermodynamic-state contract <thermodynamic_state_contract.md>`.

Physical model references
-------------------------

.. toctree::
   :maxdepth: 1

   fe_si_o_reference
   melts_silicate_reference
   melts_liquid_evaluator
