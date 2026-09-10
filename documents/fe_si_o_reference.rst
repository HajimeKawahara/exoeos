Fe-Si-O activity reference specification
========================================

This PR1 deliverable fixes equations, coefficients, standards, and independent
reference values for a future native JAX model. It adds no physical
``GibbsExcessModel``. The record is ``ma2001_fe_si_o_young2023_printed_v1``,
with ordered atomic components ``(Fe, Si, O)`` in one Fe-rich liquid phase.
It supplies neither a silicate model nor a metal-silicate equilibrium solve.

Source selection
----------------

Use the Ma epsilon formalism [Ma2001]_ with the **printed** coefficients
in Young et al., Methods equations (20)-(21) [Young2023]_. Restore the Fe
solvent term using the full Ma equations reproduced in Vogel et al.,
Appendix A1 [Vogel2018]_, and Buchan et al., equation (6) [Buchan2022]_.
Young assumes the Fe activity coefficient is one; this completion is a
distinct model with a composition-dependent Fe coefficient.

Let :math:`s=x_{\rm Si}`, :math:`o=x_{\rm O}`, and
:math:`x_{\rm Fe}=1-s-o`. All logarithms are natural. Define

.. math::

   a(T)&=12.41\frac{1873}{T}, & b(T)&=-\frac{16500}{T}, & c(T)&=-5\frac{1873}{T},\\
   l_{\rm Si}(T)&=-6.65\frac{1873}{T}, & l_{\rm O}(T)&=4.29-\frac{16500}{T}.

Here :math:`a=\epsilon_{\rm Si}^{\rm Si}`,
:math:`b=\epsilon_{\rm O}^{\rm O}`, and
:math:`c=\epsilon_{\rm Si}^{\rm O}=\epsilon_{\rm O}^{\rm Si}`;
:math:`l_i=\ln\gamma_i^0` is the source infinite-dilution coefficient in Fe.
Values are retained at their printed precision, without refitting or
treating estimated inputs as new measurements.

The oxygen comparison is resolved **as a version choice**:

.. list-table:: Coefficient multiplying ln(1-o) in the O correction
   :header-rows: 1

   * - Source
     - Coefficient
   * - Young (2023), printed equation (21)
     - :math:`+16500/T`
   * - Werlen (2026), arXiv v1 equation (10) [Werlen2026]_
     - :math:`+16500/T`
   * - Young author code [YoungCode]_
     - :math:`+1873/T`
   * - Pinned GCE Young_2023_Version/Equations.py [GCE]_
     - :math:`+1873/T`

The publication/code difference already exists within Young's sources.
Selecting the printed expression does not establish an upstream bug or the
authors' intended correction. Badro (2015),
`DOI <https://doi.org/10.1073/pnas.1505672112>`_, is Young's cited upstream
source, but its supplementary coefficients were not independently retrieved.
This is not a verified Badro coefficient set. The GCE comparison commit is
``31558873d8da460c3cd11986b574cb347621b43d``; its Si correction agrees, while

.. math::

   d_{\rm O}^{\rm printed}-d_{\rm O}^{\rm GCE}
   =\frac{16500-1873}{T}\ln(1-o).

This comparison is before adding Fe or changing standards.

Common free energy and Fe solvent
---------------------------------

The following scalar is derived here by integrating the selected solute
expressions and checking against the published full Fe equation. It is not
a fitted result or a quoted scalar equation from Ma.

.. math::

   Q(s,o)={}&-so-s\ln(1-o)-o\ln(1-s)\\
       &+\frac{s^2o^2}{2}\left[\frac1{1-s}+\frac1{1-o}-1\right],\\
   f_{\rm src}={}&l_{\rm Si}s+l_{\rm O}o+a[s+(1-s)\ln(1-s)]\\
       &+b[o+(1-o)\ln(1-o)]+cQ(s,o).

The printed Young corrections are :math:`d_{\rm Si}=\partial_s f_{\rm src}`
and :math:`d_{\rm O}=\partial_o f_{\rm src}`. The **completed** model uses

.. math::

   \ln\gamma_{\rm Fe}^{\rm src}&=f_{\rm src}-s d_{\rm Si}-o d_{\rm O},\\
   \ln\gamma_{\rm Si}^{\rm src}&=\ln\gamma_{\rm Fe}^{\rm src}+d_{\rm Si},\\
   \ln\gamma_{\rm O}^{\rm src}&=\ln\gamma_{\rm Fe}^{\rm src}+d_{\rm O}.

For an independent check, Vogel A1 simplifies, with :math:`A=(1-s)^{-1}`
and :math:`B=(1-o)^{-1}`, to

.. math::

   \ln\gamma_{\rm Fe}^{\rm src}={}&a[s+\ln(1-s)]+b[o+\ln(1-o)]+cso(1-A-B)\\
   &-\frac{c}{2}s^2o^2[3(A+B-1)+sA^2+oB^2].

The generator evaluates the original five-term A1 and full solute equations
independently of scalar derivatives. Vogel's accepted-manuscript A2 prints
a different reciprocal-term sign inside the negative solute cross term.
The adopted minus sign is in Buchan (6) and Young (20)-(21); Vogel A2 is
not used. The scalar and A1 supply an additional consistency check.

Formal symmetric endmembers
---------------------------

Young describes pure-species standards at the specified temperature and
1 bar. The completed Fe-rich formula nevertheless has formal pure-solute
limits :math:`f_{\rm src}(s=1)=l_{\rm Si}+a` and
:math:`f_{\rm src}(o=1)=l_{\rm O}+b`, generally nonzero. Consequently
``f_src`` cannot be returned through ExoEOS's symmetric ``gex_RT`` contract.
Define the explicit formal-endmember shift

.. math::

   \boldsymbol{h}&=(0,l_{\rm Si}+a,l_{\rm O}+b),\\
   f_{\rm sym}&=f_{\rm src}-s h_{\rm Si}-o h_{\rm O}\\
      &=a(1-s)\ln(1-s)+b(1-o)\ln(1-o)+cQ(s,o).

All three pure-endmember scalar limits are zero. This is a **formal
continuation**, not calibrated pure Si/O liquid thermochemistry or stability.
Activities and the consumer's standard potentials must both be transformed:

.. math::

   \ln\gamma_i^{\rm sym}&=\ln\gamma_i^{\rm src}-h_i,\\
   \mu_i^{\circ,\rm sym}&=\mu_i^{\circ,\rm src}+RT h_i.

Then :math:`\mu_i=\mu_i^\circ+RT[\ln x_i+\ln\gamma_i]` and reaction free
energies are unchanged **within the completed model**. Restoring Fe itself
changes Young's approximation. Absolute standard thermochemical data are
not supplied. Specifically, :math:`h_{\rm Si}=5.76(1873/T)` and
:math:`h_{\rm O}=4.29-33000/T`.

For Henry mole-fraction standards, use
:math:`\mu_i^{\circ,H}=\mu_i^{\circ,\rm src}+RT l_i` and
:math:`\ln\gamma_i^H=\ln\gamma_i^{\rm src}-l_i` for solutes; Fe is unchanged.
Formal symmetric standards differ from these Henry standards by
:math:`RT\epsilon_i^i`. No missing reference is assigned zero. Weight-percent
coefficients cannot be substituted without composition-basis conversion.

Limits and applicability
------------------------

The smooth mathematical domain is :math:`T>0`, :math:`s,o\ge0`,
:math:`s+o<1`. Pressure is positive, in Pa, and is not modeled. The reference
pressure is :math:`10^5` Pa; high-pressure validity is not established.
Young gives no calibrated T/P/composition box for these equations. The
reference temperatures (1873, 2350, 3000 K) and compositions are numerical
checks, not experimental observations or equilibrium states. The intended
physical use is Fe-rich liquid, whose phase validity needs independent
assessment. The old 1700 K, 0.73 GPa pilot is not a calibration target.

Pure Fe gives :math:`\ln\boldsymbol\gamma^{\rm src}=(0,l_{\rm Si},l_{\rm O})`
and :math:`\ln\boldsymbol\gamma^{\rm sym}=(0,-a,-b)` exactly. Solute-zero
edges have analytic limits: expand products containing
:math:`\ln(1-x)/x` before evaluation, using ``log1p`` in a native model.
Do not clip fractions or insert fictitious traces. Pure Si/O scalar limits
and the present-component coefficient are defined; absent-component
derivatives can diverge, so finite ``lngamma`` vectors are unsupported there.
With all interactions zero, :math:`f_{\rm sym}=0`; source linear terms are
absorbed into the standards.

Reproduction and next implementation
------------------------------------

``tests/reference/fe_si_o_ma2001.json`` contains seven states, all three
components, conversion shifts, source hashes, and both printed/code solute
corrections. ``ln_gamma_source`` means the completed model in the source
reference convention, not Young's printed activity vector. The offline
``tests/reference/generate_fe_si_o_ma2001.py`` uses 60-digit Decimal
arithmetic, rounded to binary64 in JSON. Direct comparison tolerances are
``rtol=atol=5e-12``; amount finite differences use ``rtol=atol=2e-9`` for
differencing error. Source precision and unquantified experimental error are
separate. These calculated equation references are independent of a future
JAX backend; they are not measured activities.

Tests compare independent extensive scalar derivatives to the analytic
activities, check Euler and limits, and preserve chemical potentials and
the reaction free energy of :math:`2\mathrm{FeO}+\mathrm{Si}\to
\mathrm{SiO_2}+2\mathrm{Fe}` under standard conversion.

.. code-block:: console

   python tests/reference/generate_fe_si_o_ma2001.py --check
   JAX_ENABLE_X64=1 python -m pytest tests/unittests/fe_si_o_reference_test.py tests/unittests/gibbs_excess_test.py tests/unittests/api/public_import_contract_test.py
   ./update_doc.sh

For the optional source comparison, obtain GCE at the pinned commit and run:

.. code-block:: console

   python tests/reference/generate_fe_si_o_ma2001.py --check --gce-checkout /path/to/GlobalChemicalEquilibrium_Release

This verifies the source hash and evaluates only the two pinned expressions,
without importing its solver. Four nonzero-solute cases compare directly;
the upstream expressions have removable zero-solute singularities. Ordinary
tests require no network or external thermodynamic package.

PR2 can implement :math:`f_{\rm sym}` behind the existing
``solution_state(model, T_K, P_Pa, x_phase).lngamma`` API. The output excludes
ideal mixing, standards, and gas pressure terms. Acceptance still requires
JIT/VMAP, float64, temperature/parameter derivatives, amount scaling,
Gibbs-Duhem, Hessian symmetry, and documented domain handling. ExoGibbs owns
standard thermochemistry, bar-to-Pa conversion, and equilibrium. MELTS
silicate references remain PR3; H/S/C/N extensions follow separately.

References
----------

.. [Ma2001] Ma (2001), *Thermodynamic description for concentrated metallic
   solutions using interaction parameters*. https://doi.org/10.1007/s11663-001-0011-0
.. [Young2023] Young, Shahar, and Schlichting (2023), *Earth shaped by
   primordial H2 atmospheres*, Methods (20)-(21).
   https://doi.org/10.1038/s41586-023-05823-0
   Author PDF: https://faculty.epss.ucla.edu/~eyoung/reprints/Young_etal_2023_Nature.pdf
.. [Vogel2018] Vogel et al. (2018), *The dependence of metal-silicate
   partitioning of moderately volatile elements on oxygen fugacity and
   Si contents of Fe metal*, Appendix A1. https://doi.org/10.1016/j.gca.2018.06.022
   Accepted manuscript: https://discovery.ucl.ac.uk/id/eprint/10120068/17/Jennings_1Authors_accepted_version.pdf
.. [Buchan2022] Buchan et al. (2022), *Planets or asteroids? A geochemical
   method to constrain the masses of White Dwarf pollutants*, (6)-(9).
   https://academic.oup.com/mnras/article/510/3/3512/6472245
.. [Werlen2026] Werlen et al. (2026), *The Effects of Non-ideal Mixing in
   Planetary Magma Oceans and Atmospheres*, v1. https://arxiv.org/abs/2602.05917v1
.. [YoungCode] Author code, pinned before removal of the original file.
   https://github.com/eyoungucla/chems/blob/0a8e03361db27aa72782cde22cb2d204b513224c/Exoplanet_atmosphere_model_vMCMC_coreT_3000K_dist.py#L1581
.. [GCE] Global Chemical Equilibrium release, GPL-3.0, comparison only.
   https://github.com/ExoInteriors/GlobalChemicalEquilibrium_Release/blob/31558873d8da460c3cd11986b574cb347621b43d/Young_2023_Version/Equations.py#L111-L117
