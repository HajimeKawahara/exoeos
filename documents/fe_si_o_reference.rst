Fe-Si-O liquid activities and reference
==========================================

``MaFeSiOLiquid`` implements a native JAX ``GibbsExcessModel`` for ordered
atomic mole fractions ``(Fe, Si, O)`` in one Fe-rich liquid phase. Its defaults
use the pinned record ``ma2001_fe_si_o_young2023_printed_v1``, including the
Fe solvent and formal endmember standards. This is a metal-only model;
silicate activities and metal-silicate equilibrium remain separate work.

Native model API
----------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from exoeos import MaFeSiOLiquid, solution_state

   model = MaFeSiOLiquid()
   T, P = 2350.0, 1.0e5  # K, Pa
   x = jnp.array([0.85, 0.10, 0.05])  # Fe, Si, O
   model.validate_state(T, P, x)
   state = jax.jit(solution_state)(model, T, P, x)
   shift_RT = model.standard_state_shift_RT(T)
   lngamma_source = state.lngamma + shift_RT

``state.gex_RT`` is the scalar :math:`f_{\rm sym}` defined below;
``state.lngamma`` contains all three natural-log activity coefficients.
Neither includes ideal mixing, standard potentials, or gas pressure terms.
If the consumer supplies source standard potentials divided by :math:`RT`,
it must also apply ``mu0_formal_RT = mu0_source_RT + shift_RT``.
This preserves chemical potentials within the completed model.

The constructor is
``MaFeSiOLiquid(interaction_K=(12.41*1873, -16500, -5*1873))``.
``interaction_K`` is a differentiable array of shape ``(3,)``, ordered
``(Si-Si, O-O, Si-O)`` and expressed in K. Dividing it by temperature gives
:math:`(a,b,c)`. It is the immutable model's only numerical PyTree leaf;
the source infinite-dilution expressions :math:`l_i(T)` remain fixed.
``standard_state_shift_RT(T)`` uses the current interactions, including
overrides. Custom parameters are mathematical variants without established
physical calibration.

The model records ``components``, ``activity_basis``,
``standard_state_convention``, ``reference_model_id``, and
``reference_pressure_Pa``. The reference identifier describes the default
parameter set. Use external ``jax.vmap`` for batches; input and parameter
dtypes participate in promotion to at least float32.

Call ``validate_state(T, P, x)`` eagerly before JAX transformations. It checks
finite real positive T/P, finite real interactions, and normalized
nonnegative finite real fractions with positive Fe. Normalization is checked
within eight machine epsilons of the input composition dtype.
Solute fractions must remain below one before and after normalization;
positive Fe smaller than floating-point resolution cannot avoid a singularity.
It raises for unsupported states and is never
called implicitly by ``gex_RT`` or ``solution_state``. The kernel retains
its static shape checks; notably, its extensive construction normalizes
fractions internally, so it cannot detect an unnormalized original input.
Validation checks the mathematical domain, not liquid stability or an
experimental calibration range.

Source selection
----------------

Use the Ma epsilon formalism [Ma2001]_ with the **printed** coefficients
in Young et al., Methods equations (20)-(21) [Young2023]_. Restore the Fe
solvent term using the full Ma equations reproduced in Vogel et al.,
Appendix A1 [Vogel2018]_, and Buchan et al., equation (6) [Buchan2022]_.
Young assumes the Fe activity coefficient is one; this completion is a
distinct model with a composition-dependent Fe coefficient.

Let :math:`s=x_{\rm Si}`, :math:`o=x_{\rm O}`, and
:math:`x_{\rm Fe}=1-s-o`. All logarithms are natural. For the defaults, define

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
edges have analytic limits. The native expression removes the quotients
:math:`\ln(1-x)/x` algebraically and uses ``xlog1py`` for products with
logarithms. It uses exact zero-product limits in the reciprocal terms at
pure Si/O; it does not clip fractions or insert fictitious traces.
Direct ``gex_RT`` calls give zero at these two formal endpoints, but absent
component derivatives can diverge and ``solution_state(...).lngamma`` is NaN
there. ``validate_state`` rejects them for activity calculations. The
present-component coefficient has a zero limit when approached through
valid compositions; no finite endpoint activity vector is supplied.
With all interactions zero, :math:`f_{\rm sym}=0`; source linear terms are
absorbed into the standards.

Reproduction and verification
-----------------------------

``tests/reference/fe_si_o_ma2001.json`` contains seven states, all three
components, conversion shifts, source hashes, and both printed/code solute
corrections. ``ln_gamma_source`` means the completed model in the source
reference convention, not Young's printed activity vector. The offline
``tests/reference/generate_fe_si_o_ma2001.py`` uses 60-digit Decimal
arithmetic, rounded to binary64 in JSON. Direct comparison tolerances are
``rtol=atol=5e-12``; amount finite differences use ``rtol=atol=2e-9`` for
differencing error. Source precision and unquantified experimental error are
separate. These calculated equation references are independent of the native
JAX implementation; they are not measured activities. The fixture and
generator retain their original PR1 contents, including the historical
reference-only status in the JSON record.

Tests compare independent extensive scalar derivatives to the analytic
activities, check Euler and limits, and preserve chemical potentials and
the reaction free energy of :math:`2\mathrm{FeO}+\mathrm{Si}\to
\mathrm{SiO_2}+2\mathrm{Fe}` under standard conversion. Native model tests
also cover JIT/VMAP, floating dtypes, temperature and parameter derivatives,
amount scaling, Gibbs-Duhem, Hessian symmetry, and the validation boundary.

.. code-block:: console

   python tests/reference/generate_fe_si_o_ma2001.py --check
   JAX_ENABLE_X64=1 python -m pytest tests/unittests/ma_fe_si_o_test.py tests/unittests/fe_si_o_reference_test.py tests/unittests/gibbs_excess_test.py tests/unittests/api/public_import_contract_test.py
   ./update_doc.sh

For the optional source comparison, obtain GCE at the pinned commit and run:

.. code-block:: console

   python tests/reference/generate_fe_si_o_ma2001.py --check --gce-checkout /path/to/GlobalChemicalEquilibrium_Release

This verifies the source hash and evaluates only the two pinned expressions,
without importing its solver. Four nonzero-solute cases compare directly;
the upstream expressions have removable zero-solute singularities. Ordinary
tests require no network or external thermodynamic package.

The native backend uses the existing Gibbs-excess kernel without changing
its responsibilities. ExoGibbs owns standard thermochemistry, bar-to-Pa
conversion, and equilibrium. Independent external silicate states are now
recorded in :doc:`melts_silicate_reference`; they supply no native silicate
backend. H/S/C/N extensions follow separately. These deliverables do not
establish a coupled nonideal calculation for both melt and metal phases.

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
