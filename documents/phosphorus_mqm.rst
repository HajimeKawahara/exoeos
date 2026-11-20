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

The optional ``host_model`` argument selects the Fe-Si-O excess model;
omitting it retains ``MaFeSiOLiquid()`` and its current defaults. To reproduce
an explicitly chosen older conditional model, construct, for example,
``MaFeSiOLiquid(interaction_K=(12.41*1873., -16500., -5.*1873.))`` and pass
that instance to ``gibbs_j``. Convert source Fe/Si/O standards with the same
instance's ``standard_state_shift_RT(T)`` before supplying the twenty
standards. Changing only the excess term would mix two conventions. The host
coefficients remain differentiable through JAX, including through
``gibbs_response``; this selection supplies no new calibration or phase proof.

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

The Fe3P magnetic normalization follows Eq. 3.4's per-atom wording. The
smaller heat-capacity peak in the thesis Fig. 3.3 suggests that its software
may instead apply the magnetic term per formula. That implementation is not
available, so reproduction of that figure is not claimed. The S1 consumer
reports the one-versus-four multiplier energy span; above 2100 K it is below
5 J/mol formula. The four-atom convention lowers the competitor's energy
more. Neither convention resolves the broader high-temperature continuation.

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

Global insertion lower bound
-----------------------------

``exoeos.phosphorus_mqm_interval.mqm_alloy_insertion_bound`` supplies a
conservative lower bound for the unchanged conditional alloy scalar. This is
an eager NumPy/interval calculation; it is not differentiable or JIT-compatible.
It can be used after a local equilibrium solve. ExoGibbs still owns phase
selection, and an accepted alloy bound does not establish stability of the
melt or of the complete phase assemblage.

The two 20-entry potential arrays follow
``associated_metal.COMPONENTS``. Standards use RT per mole of collapsed
chemical species. A supporting potential is the elemental chemical-potential
plane evaluated for that chemical species. The output ``lower_rt`` instead
uses **RT per mole of the original 21 phase coordinates**, including pair
coordinates. The other result fields retain the chemical-mole intermediate
bound and each conservative contribution.

``host_lower`` and ``host_upper`` bound chemical-species fractions after P
has been removed from the host normalization. Their P entries must be zero;
this does not restrict the total P concentration. The bound covers pure P
by continuity. Pass the same ``host_model`` and dynamic material coefficients
as the forward scalar. The supplied box must have positive lower Fe and
upper Si/Fe and O/Fe ratios below one. The implemented Ma coefficient signs
are nonnegative Si self interaction and nonpositive O self/cross interactions.
Unsupported signs, nonfinite inputs and reduced-precision provider
coefficients raise errors. This numerical domain is not an empirical domain.

A standalone ExoEOS example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example deliberately supplies a synthetic supporting plane and
standards, so it needs no equilibrium solver. It demonstrates the sign and
normalization of the result; these numbers do not describe a planetary state.
The declared host box is likewise an illustrative mathematical input.

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)
   import numpy as np
   from exoeos import associated_metal
   from exoeos.phosphorus_mqm_interval import mqm_alloy_insertion_bound

   temperature = 2173.15
   epsilon, matrix = associated_metal.interaction_parameters(temperature)
   lower = np.zeros(20)
   lower[0] = 0.75  # At least 75% Fe in the P-free chemical host.
   upper = np.full(20, 0.02)
   upper[0], upper[4] = 1.0, 0.0  # P is free outside this host normalization.
   inputs = dict(
       temperature_k=temperature, pressure_pa=6e6,
       pure_standards_rt=np.full(20, 40.0),
       supporting_potentials_rt=np.zeros(20),
       phosphorus_epsilon=epsilon, additional_matrix=matrix,
       host_lower=lower, host_upper=upper,
       model="you2020", delta_v_p_m3_mol=0.0,
   )
   bound = mqm_alloy_insertion_bound(**inputs)
   assert bound.lower_rt > 0
   print("Insertion is excluded against this plane:", bound.lower_rt)

   inputs["supporting_potentials_rt"] = np.full(20, 50.0)
   unresolved = mqm_alloy_insertion_bound(**inputs)
   assert unresolved.lower_rt < 0
   print("This lower bound is inconclusive:", unresolved.lower_rt)

A positive lower bound excludes insertion on the declared domain. A negative
lower bound is only inconclusive: it is not a feasible negative-energy witness.
For real equilibrium outputs, use a common elemental gauge and enclose the
external atom-matrix/dual product. The function accepts an outward upper bound
on each supporting potential because all chemical amounts are nonnegative.
Supplied binary64 standards and material coefficients are treated as exact
constants of the mathematical scalar being bounded; experimental uncertainty
and arbitrary floating-point errors in separate forward evaluations are not
included. The bound evaluation itself uses outward-rounded arithmetic and a
logarithm series with an explicit remainder.

Why the bound is global
~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`y=Cn` be the collapsed chemical amounts, :math:`H=\sum_{i\ne P}y_i`
the P-free host amount, and :math:`N=H+y_P`. The supporting-plane costs are
:math:`c_i=s_i-v_i`; include the declared pressure-volume term in :math:`c_P`.
After cancelling the binary ideal term that is replaced by the MQM scalar,
the mixing terms are chemical ideal entropy plus the pair correction

.. math::

   N_q\left[\sum_{ab}p_{ab}\ln p_{ab}
                -2\sum_a p_a\ln p_a\right]
   =N_q D_{\rm KL}(p_{ab}\Vert p_a p_b)\geq0.

Here :math:`p_{FeP}=p_{PFe}=X_{FeP}/2`, and the two diagonal entries are
:math:`X_{FeFe},X_{PP}`. Thus the correction is mutual information, including
the explicit unlike-pair :math:`-n_{FeP}\ln2` term. The remaining chemical
ideal term satisfies :math:`\sum_i y_i\ln(y_i/N)\geq-N\ln20`, including K
and Na even though their excess-energy treatment differs.

For the pair polynomial :math:`\Delta g=d_0+d_1a+d_2a^2+d_3b`,
:math:`0\leq a,b\leq1` gives the conservative bound

.. math::

   d_* = d_0+\sum_{i=1}^3\min(0,d_i)\leq\Delta g,\qquad
   \frac{n_{FeP}\Delta g}{2RT}\geq y_P b_q,
   \quad b_q=\frac{\min(0,d_*)}{2C_{P,FeP}RT}.

The exact stored collapse coefficient :math:`C_{P,FeP}` is used, rather than
assuming that a rounded reciprocal is exact. This covers the entire pair
simplex and the unrestricted P range. No sampled pair minimum is used.

The P-free host box gives a linear cost bound. For every real :math:`\beta`,

.. math::

   \sum_{i\ne P} c_i y_i/H\geq
   \beta+\sum_{i\ne P}
   \left[l_i(c_i-\beta)_+ + u_i(c_i-\beta)_-\right].

The implementation maximizes several such valid bounds to obtain
:math:`L_h`. Candidate choices do not require an optimizer certificate.
The remaining host excess is bounded below by :math:`Hb_h+y_Pb_P`:

* The Ma Si term uses :math:`(1-x)\ln(1-x)\geq-x`; the nonnegative O-self
  contribution can be omitted. The Si--O term uses
  :math:`-\ln(1-x)\leq x/(1-x)` on the enclosing Si/O box. Its dry host amount
  is between :math:`l_{Fe}H` and :math:`H`.
* The additional-matrix core includes Na as pseudo-Fe, excludes K, and includes
  P. Its denominator is at least both :math:`l_{Fe}H` and :math:`y_P`.
  Every negative non-P matrix term is bounded using the host upper amounts
  and the first denominator bound. For negative P-cross terms,
  :math:`y_P/\mathrm{core}\leq1` gives a host-linear bound. A negative P-self
  term contributes :math:`b_P=\min(0,M_{PP})/2`. Positive terms may be omitted.
* The separate P-cross coefficients use the same argument with their
  five-component denominator, which is at least :math:`y_P`. Na changes only
  the pseudo-Fe host coordinate and its upper bound, never the pair polynomial.

For clarity, the Ma cross-term upper function used above is

.. math::

   \Phi(s,o)=so\left[-1+\frac{1}{1-s}+\frac{1}{1-o}\right]
     +\frac{(so)^2}{2}\left[\frac{1}{1-s}+\frac{1}{1-o}-1\right].

Every factor and bracket is nonnegative and nondecreasing for
:math:`0\leq s,o<1`. Replacing them by the enclosing upper values
:math:`s\leq u_{Si}/l_{Fe}`, :math:`o\leq u_O/l_{Fe}` therefore bounds the
whole term above. Multiplying by the nonpositive Ma cross coefficient reverses
the bound, as required. Finite inputs that overflow any interval step are
rejected; an infinite or NaN intermediate cannot produce a certificate.

Combining these inequalities gives

.. math::

   q(n)/(RT)\geq N B,\qquad
   B=\min(L_h+b_h,\ c_P+b_q+b_P)-\ln20.

For :math:`\sum_j n_j=1`, the collapse columns give
:math:`r_{\min}\leq N\leq r_{\max}`. Consequently the reported bound is
:math:`r_{\min}B` if :math:`B\geq0`, and :math:`r_{\max}B` otherwise. Using
only the minimum normalization factor for negative bounds would be incorrect.
All these evaluation steps are enclosed with interval arithmetic. Fe/P pure
faces and zero guest amounts are included by continuity. The bound can remain
loose at an existing alloy's tangent plane; local sampling cannot tighten a
negative lower bound into a global certificate.
