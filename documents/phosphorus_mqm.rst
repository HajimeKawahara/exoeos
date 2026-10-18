Fe-P modified quasichemical liquids
======================================

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

Conditional multicomponent connection
-------------------------------------

``phosphorus_mqm_alloy`` connects the binary scalar to the existing
twenty-component host. This construction is a declared hypothesis, not the
published multicomponent You assessment. Its 21 coordinates replace Fe and P
with FeFe and PP pairs and append FeP. Other coordinates still count chemical
species. ``collapse_matrix(model=...)`` returns their exact mapping to the
parent chemical ledger; the caller multiplies its parent atom matrix by this
mapping. Pair counts must never be treated as molecular or atomic amounts.

The scalar counts global ideal entropy on the collapsed ledger once, removes
the binary Fe/P ideal term, then adds the complete binary MQM Gibbs energy.
It retains the parent host and P-cross excess terms and removes the old
quadratic P-self interaction. Na remains pseudo-Fe in the host excess only.
This recovers the binary MQM with other components absent and the parent
host when P is absent. Host composition restrictions remain the caller's
responsibility; there is no phosphorus concentration cap.

Pass pure-liquid Fe/P standards to ``phosphorus_mqm_alloy.gibbs_j``. For a
joint absolute reference, ``pure_p_standard_rt(T, gas_p2_standard_rt)`` in
``phosphorus_mqm`` translates SGTE liquid P using the caller's one-bar P2
gas standard and the `NIST SRD69 P2 Shomate reference
<https://webbook.nist.gov/cgi/cbook.cgi?ID=C12185090&Mask=1>`_. The gas standard
must exclude its partial-pressure logarithm. To construct a separate
Henry-preserving concentration control, subtract ``henry_log_gamma`` from
the chosen Henry standard instead. These are different physical references,
not equivalent gauges.

The required keyword ``delta_v_p_m3_mol`` declares an additional constant
P partial volume relative to the supplied standards. Its extensive energy
is :math:`n_P(P-10^5\,\mathrm{Pa})\Delta V_P`. Zero explicitly continues the
one-bar P standard. A nonzero value is a caller assumption; no measured
coefficient, pressure ceiling or uncertainty interval is supplied.

Stoichiometric phosphide candidates
-----------------------------------

``phosphorus_phosphides`` transcribes HT-Fe3P, Fe2P, FeP and FeP2 from the
same You thesis Table 3.2. Energies and heat capacities count formula units.
``standard_gibbs_j`` integrates the stated heat capacities from 298.15 K
using the formation enthalpies and absolute entropies on the alpha-Fe and
white-P SER. HT-Fe3P includes its 1000 J/mol transition at 698 K and the
Hillert--Jarl magnetic term with four atoms per formula.

The explicit Fe3P heat-capacity endpoint is 2000 K. The other three formulas
have no printed endpoints. This candidate set conservatively requires
``high_temperature_continuation=True`` above 2000 K, continuing the final
heat-capacity expressions only up to 3000 K. Outside 298.15--3000 K it returns
NaN. The flag does not establish measured high-temperature applicability.
The low-temperature Fe3P polymorph and Fe/P solid solutions are not supplied.

``common_gibbs_j`` translates every solid with the same native Fe-liquid
and P2 standards that anchor the alloy. It uses SGTE GLIQFE and NIST P2 to
preserve reaction energies under elemental gauge changes. Supply four
``delta_volumes_m3_mol_formula`` explicitly; these are additional one-bar
volume increments, not tabulated measured values. Each compound can be
registered as a separate one-component ExoGibbs phase. Their presence,
absence and competition then follow from the coupled finite inventory.

Tests check atom mappings, binary and P-free limits, extensivity, gradients,
Hessians, volume derivatives, common-reference reaction invariance and
independent numerical integration of heat capacities. The ExoInventory
comparison also checks the published Fe2P and FeP congruent temperatures.
These four candidates do not constitute a complete phase-diagram proof.
