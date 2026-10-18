Fe-P modified quasichemical liquids
==================================

``exoeos.phosphorus_mqm`` exposes an extensive, differentiable Gibbs scalar
for the binary liquid models I and II of `Bernhard et al. (2020)
<https://doi.org/10.1016/j.calphad.2020.101795>`_ and Table 3.2 of
`You's 2020 author thesis, Chapter 3
<https://mcgill.scholaris.ca/server/api/core/bitstreams/af508e1a-b2ee-4a73-873e-3606b8531ec6/content>`_.
The related You--Jung journal article is
`doi:10.1007/s11663-020-01939-0
<https://doi.org/10.1007/s11663-020-01939-0>`_; identity of the thesis and
final journal parameter versions is not assumed. No parameters are refitted.

Call ``gibbs_j(T, pairs, pure_standards_rt, model=...)`` with pair amounts
ordered ``(FeFe, PP, FeP)``, in moles, and pure-liquid elemental Fe/P standards
in units of RT. ``atom_matrix(model=...) @ pairs`` returns atom moles.
For model I and You, the unlike coordination numbers are 3 for Fe and 6
for P. For model II both are 6. The like coordination numbers are 6.
Therefore the conserving pair direction is (-1, -1/2, 1) for model I/You,
and (-1/2, -1/2, 1) for model II. Pair amounts are not molecular amounts.

The scalar evaluates the complete configurational entropy and the published
pair-fraction polynomial. Equilibrium belongs to the caller, such as
ExoGibbs. A constant-energy mass-action shortcut omits composition derivatives
and does not minimize this scalar. No artificial phosphorus concentration
bound is imposed. Pure and empty endpoint values are continuous; chemical
potential derivatives are intended for strictly positive pair amounts.

The exact infinite-dilution conversion is provided by ``henry_log_gamma``:

.. math::

   \mu_P^H-G_P^{\mathrm{liquid}} = RT\ln\gamma_P^\infty
   =3\Delta g_{FeP}(X_{FeFe}=1).

It follows from the same Gibbs scalar. Do not add an independent Henry shift
unless intentionally defining a different absolute model. The You parameters
recover the rounded thesis Eq. 3.24 within 0.001 in log activity over the
comparison temperatures. ``sgte_liquid_p_j`` provides the 317.3--3000 K
branch of GLIQP from `SGTE UNARY 5.0 (2009)
<https://www.sgte.net/en/free-pure-elements-database>`_, relative to white-P
SER. It returns NaN outside that branch. Its linear-T coefficient differs
by -0.013 J/mol/K from Oikawa's later Table S1. That difference is retained.
Gas and other element standards must share an explicitly reconciled gauge.

These are liquid-only binary models. They do not reproduce the original
solid phase diagrams, establish a joint empirical domain at 2173.15 K, or
supply a multicomponent alloy or pressure correction. Historical activity
data used in the original assessments are not independent validation data.
In particular, the You thesis discusses disagreement between the 1982
transportation measurements and mutually checked P/P2 Knudsen measurements;
its proposed flow/condensation explanation is not a measured correction.

Tests check the source entropy form, atom basis, extensivity, elemental
gauge shifts, finite differences, endpoints and independent dilute limits.
The ExoInventory S1 example records the joint absolute-reference and
concentration comparison without changing the existing production alloy.
