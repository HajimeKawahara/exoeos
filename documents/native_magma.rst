Native JAX magma: MgO--FeO--SiO2
============================================================

``exoeos.magma`` evaluates a two-component MELTS liquid, pure crystalline
forsterite, the pure Mg orthopyroxene endpoint and three SiO2 polymorphs in JAX,
including their standard Gibbs energies. The examples progress from
forsterite cooling and enstatite insertion to equilibrium phase amounts
after enstatite appears, then to multiphase equilibrium with silica solids.
No external MELTS runtime is needed for evaluation, derivatives, the cooling
examples, or ordinary tests. This is the native JAX Gibbs model for silicate
magma. The binary multiphase example allows one liquid, forsterite, Mg orthopyroxene,
quartz, tridymite and cristobalite; it is not the full MgO--SiO2 phase diagram.
The Fe(II) extension below adds a ternary liquid and a binary
forsterite--fayalite olivine solution, with a separate closed liquid + olivine
equilibrium example for Fe/Mg partitioning.

Gibbs energy and component basis
--------------------------------

Let :math:`Q=\mathrm{SiO_2}` and :math:`F=\mathrm{Mg_2SiO_4}` be liquid
endmembers. They are composition coordinates, not a statement that the
liquid contains only those molecules. The model is

.. math::

   G_l=n_Q g_Q^0+n_F g_F^0
       +R_bT(n_Q\ln x_Q+n_F\ln x_F)
       +W\frac{n_Q n_F}{n_Q+n_F},\qquad x_i=\frac{n_i}{n_Q+n_F}.

The MELTS constants are :math:`R_b=8.3143` J/(mol K) and
:math:`W=3421` J/mol. Using oxide mole fractions in the entropy term would
define a different model.

The original binary functions have scalar ``T`` in K and ``P`` in Pa:

.. list-table:: API in ``exoeos.magma``
   :header-rows: 1
   :widths: 43 57

   * - Function
     - Result
   * - ``liquid_standard_gibbs(T, P)``
     - J/mol, shape (2,), ordered **SiO2, Mg2SiO4**.
   * - ``liquid_gibbs(T, P, n)``
     - Extensive G in J; ``n`` is shape (2,) in mol, in that same order.
   * - ``forsterite_gibbs(T, P)``
     - Crystalline Mg2SiO4 standard G in J/mol.
   * - ``enstatite_gibbs(T, P)``
     - Pure Mg orthopyroxene G in J per mol **MgSiO3**.
   * - ``silica_gibbs(T, P)``
     - J/mol SiO2, shape (3,), ordered **quartz, tridymite, cristobalite**,
       also exported as ``SILICA_POLYMORPHS``.

Use positive finite T/P and finite nonnegative amounts; validate material
inputs before entering JAX transformations. Static shapes are checked by
the functions. Enable x64 before calculations. Zero component amounts use
the continuous energy limit and an absent phase has exactly zero G.
Composition derivatives apply to positive interiors. The absent component's
ideal potential tends to minus infinity; an absent phase has no composition
or chemical potentials. Boundary energy evaluation is not a boundary
chemical-potential API.

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)
   import jax.numpy as jnp
   from exoeos.magma import liquid_gibbs

   T, P = 2000.0, 1.0e5
   n = jnp.array([0.25, 0.75])
   G = jax.jit(liquid_gibbs)(T, P, n)             # J
   mu = jax.grad(liquid_gibbs, 2)(T, P, n)       # J/mol
   entropy = -jax.grad(liquid_gibbs, 0)(T, P, n) # J/K
   volume = jax.grad(liquid_gibbs, 1)(T, P, n)   # m3
   values = jax.jit(jax.vmap(liquid_gibbs, in_axes=(0, None, None)))(
       jnp.array([2000.0, 2050.0, 2100.0]), P, n
   )

For existing reduced interfaces, divide the dimensional G by the caller's
common R*T. This conversion does not change :math:`R_b` inside the physical
mixing entropy. ``total_solution_gibbs_RT`` gives the same result with the
same standards, interaction and gas constant. Standards retain their T/P
dependence when taking thermal and pressure derivatives.

Standard states
---------------

The solid uses the Berman reference at 298.15 K and 1 bar:
:math:`H_r=-2174420` J/mol, :math:`S_r=94.010` J/(mol K), and

.. math::

   C_{p,s}(T)=238.64-2001.3/\sqrt{T}-1.1624\,10^8/T^3.

Integrate Cp and Cp/T for H and S, then use G=H-TS and the integrated
Berman pressure polynomial. Liquid F starts from the solid reference at
2163 K, adds fusion entropy 57.2 J/(mol K), and uses liquid Cp=271 J/(mol K).
Liquid Q follows the dedicated ``gibbs.c`` SiO2 formula, with a branch
change at 1480 K. G and its first temperature derivative are continuous;
the heat capacity can jump. The high-temperature branch is selected at
equality. The general tabulated SiO2 fusion formula is not used.

Both liquids use the integrated Kress pressure polynomial relative to
1 bar, with reference temperature 1673 K. Internal volumes are J/(bar mol),
so pressure is converted from Pa to bar before integration. Differentiating
the returned G with respect to Pa gives SI volume. Pressure checks are
formula checks, not high-pressure calibration.

Conservation and minimization
-----------------------------

An initial oxide inventory of M mol MgO and S mol SiO2 gives
:math:`a=M/2` mol F and :math:`b=S-M/2` mol Q. This nonnegative basis requires
:math:`0\le M/S\le2` with S>0. Crystallizing :math:`\xi` mol forsterite leaves
:math:`n_F=a-\xi` and :math:`n_Q=b`, conserving Mg, Si, O and mass. Crystals
remain in the closed system. Minimize

.. math::

   G_{\rm total}(\xi)=G_l(T,P,[b,a-\xi])+\xi g_{F,s}^0,
   \qquad 0\le\xi\le a.

Its derivative is

.. math::

   \frac{dG_{\rm total}}{d\xi}=g_{F,s}^0-\mu_{F,l},\qquad
   \mu_{F,l}=g_{F,l}^0+R_bT\ln x_F+W x_Q^2.

A nonnegative initial slope selects zero crystals. A negative initial
slope favors crystallization; an interior minimum satisfies equality of
the solid and liquid potentials. The example bisects the derivative of
this same G. It requires T>W/(2R_b), where the binary liquid is strictly
convex. The pure F case compares the two standards directly; at equality
any phase fraction minimizes G, and the example returns zero crystals.
General phase equilibrium and phase selection remain ExoGibbs responsibilities.

Cooling example
---------------

Run from the source checkout with the optional plotting dependency:

.. code-block:: bash

   JAX_ENABLE_X64=1 MPLBACKEND=Agg python examples/magma_cooling.py \
       --output documents/_static/magma

For M=1.5 mol, S=1 mol, P=1 bar, the restricted onset is **2077.600179 K**.

.. list-table:: Closed equilibrium cooling
   :header-rows: 1

   * - T (K)
     - Crystalline forsterite (mol)
     - Residual liquid x_F
   * - 2100
     - 0
     - 0.750000
   * - 2050
     - 0.220119
     - 0.679438
   * - 2000
     - 0.425209
     - 0.565059

.. figure:: _static/magma/gibbs_minima.png
   :alt: Gibbs minima shift from zero crystals at 2100 K to finite amounts on cooling.
   :width: 85%

   The energy minimum determines the crystalline amount; points mark the minima.

.. figure:: _static/magma/cooling_curve.png
   :alt: Crystal amount increases and liquid forsterite fraction decreases below the restricted onset.
   :width: 85%

   Dashed lines mark the restricted onset. Cooling proceeds from left to right.

Enstatite and silica solids are omitted from this minimization. The next
section tests enstatite insertion without calculating its equilibrium amount.
Competing phase stability must be examined before interpreting this as the
physical binary phase diagram. MgO/SiO2>2 requires a different basis.

Enstatite insertion and comparison with MELTS
-----------------------------------------------

``enstatite_gibbs(T, P)`` returns the pure Mg **orthopyroxene** endpoint,
in J per mol MgSiO3. It is not a minimum over enstatite polymorphs or a
pyroxene solid-solution model. The MELTS endmember basis is Mg2Si2O6,
so one native mole corresponds to two API moles.

The tabulated Mg2Si2O6 standard is monoclinic, including for orthopyroxene.
The total orthopyroxene endpoint also needs its site energy minus the
monoclinic pure-reference energy. At the pure Mg end there is no ordering
freedom or configurational site entropy. Reducing the published expression
gives, per native mole, :math:`\Delta H=-5020.8` J/mol,
:math:`\Delta S=-2.3237936` J/(mol K), and
:math:`\Delta V=-0.0619232` J/(bar mol). With :math:`P_b=P/10^5`,

.. math::

   g_{\rm En}=\frac{1}{2}\left[g_{\rm CEn}^0(T,P)
        -5020.8+2.3237936T-0.0619232(P_b-1)\right].

The monoclinic standard follows the Berman Cp and pressure integrals in
``sol_struct_data.h``. The endpoint correction comes from
`orthopyroxene.c <https://github.com/magmasource/MAGMA/blob/705a0fb315e5054d18275a580562f6121c8e458c/sources/orthopyroxene.c>`_
at the same pinned revision. Comparing a raw endmember standard alone
would miss this correction.

.. figure:: _static/magma/enstatite_melts_comparison.png
   :alt: JAX orthopyroxene energies overlay independent MELTS points at three pressures; the lower panel shows their signed differences.
   :width: 85%

   Pure Mg orthopyroxene at 1, 500 and 5000 bar. Lines are JAX; circles are
   actual pinned MELTS evaluations, converted from Mg2Si2O6 to MgSiO3.
   The maximum absolute difference across these 12 states is 7e-10 J/mol.
   Pressure comparisons check formulas, not empirical high-pressure accuracy.

One mol of MgSiO3 consumes half a mol each of liquid Q and F. For a
present liquid with both amounts positive, the insertion derivative is

.. math::

   \Delta g_{\rm En}=g_{\rm En}-\frac{\mu_{Q,l}+\mu_{F,l}}{2}
      =\left.\frac{\partial}{\partial\eta}
       \left[G_l(T,P,[n_Q-\eta/2,n_F-\eta/2])+\eta g_{\rm En}\right]
       \right|_{\eta=0}.

``examples/magma_cooling.py::enstatite_insertion_energy`` evaluates this
derivative using the same JAX liquid G. A negative value proves the supplied
state is unstable to this candidate. A nonnegative value checks only Mg
orthopyroxene and does not establish stability against every omitted phase.
Enstatite insertion conserves Mg, Si, O and mass.

On the existing 1 bar, 1.5 mol MgO + 1 mol SiO2 closed cooling path, the
derivative crosses zero at **1908.603653 K**, with 0.584374215 mol
forsterite. At 2000 K it is +2578.340908 J/mol; at 1900 K it is
-167.806610 J/mol. The continuation below the crossing remains a restricted
liquid + forsterite calculation and is unstable when orthopyroxene is allowed.
The following equilibrium example also allows enstatite and calculates
the phase amounts beyond this crossing.

.. figure:: _static/magma/enstatite_saturation.png
   :alt: Enstatite insertion energy crosses zero near 1909 K; JAX and independently solved MELTS restricted paths agree.
   :width: 85%

   Cooling proceeds from left to right. Circles use a separate bisection
   of actual MELTS liquid chemical potentials against pure forsterite G.
   The upper panel tests enstatite insertion on each independently computed
   path; the lower panel compares the restricted forsterite amounts.
   The shaded continuation is unstable to enstatite insertion.

``tests/reference/enstatite_source_v1.json`` independently integrates Cp,
Cp/T and V and evaluates the previously transcribed site polynomials at
the exact Mg endpoint. It records G, S, Cp and V at 12 T/P states.
``tests/reference/enstatite_melts_v1.json`` separately records 12 actual
MELTS endpoint evaluations and 12 restricted cooling states. The latter
use fresh workers and 40 bisections of native chemical potentials, with
no JAX thermodynamic model or MELTS phase-assemblage solver. The maximum
JAX/backend differences are 2.2e-7 J/mol for insertion energies and
2.1e-11 mol for forsterite amounts. Tests allow 1e-6 J/mol and 1e-9 mol,
respectively. Source agreement and these finite backend comparisons do
not validate a complete experimental phase diagram.

Regenerate the figures and numerical summary without an external runtime:

.. code-block:: bash

   JAX_ENABLE_X64=1 MPLBACKEND=Agg python examples/magma_enstatite.py \
       --output documents/_static/magma

Regenerate either independent fixture:

.. code-block:: bash

   python tests/reference/generate_enstatite_reference.py \
       --output tests/reference/enstatite_source_v1.json
   python tests/reference/generate_enstatite_reference.py --runtime /path/to/pinned/runtime \
       --python /path/to/worker/python --output tests/reference/enstatite_melts_v1.json

Equilibrium after enstatite appears
-----------------------------------

``examples/magma_equilibrium.py`` minimizes the same G over liquid,
pure forsterite (Fo) and pure Mg orthopyroxene (En). With retained crystals,
:math:`\xi` mol Fo and :math:`\eta` mol MgSiO3 leave

.. math::

   n_Q=b-\eta/2,\qquad n_F=a-\xi-\eta/2,\qquad
   G_{\rm total}=G_l(T,P,[n_Q,n_F])+\xi g_{\rm Fo}+\eta g_{\rm En}.

All four amounts must be nonnegative. These constraints conserve Mg, Si,
O and mass, including when previously crystallized Fo is consumed.
The two crystallization derivatives are :math:`g_{\rm Fo}-\mu_F`
and :math:`g_{\rm En}-(\mu_Q+\mu_F)/2`.

For T>W/(2R_b), the liquid is strictly convex in composition. The small
example compares the liquid + Fo edge, the liquid + En edge, and the
all-solid state :math:`(\xi,\eta)=(a-b,2b)` when a>=b. Each liquid/solid
edge is a one-dimensional minimization; the pure-composition cases compare
energies directly. A three-phase stationary point has a flat direction
that reaches these edges, so an interior two-dimensional optimizer is
unnecessary. The all-liquid state is included explicitly. Absent liquid
contributes exactly zero G and is never assigned chemical potentials.
This eager, scalar example adds no solver dependency to ExoEOS; general
phase selection remains an ExoGibbs responsibility.

At fixed pressure a binary three-phase equilibrium is invariant, consistent
with the `phase rule <https://goldbook.iupac.org/terms/view/P04533>`_.
At **1908.603653 K**, the common liquid has :math:`x_F=0.398497376`.
Its Q:F ratio is preserved while liquid and Fo react into En. For the
original 1.5 mol MgO + 1 mol SiO2 inventory, the equilibrium endpoints are

.. math::

   [n_Q,n_F,\xi,\eta]
   =[0.25,0.165625785,0.584374215,0]
   \quad\hbox{and}\quad [0,0,0.5,0.5].

Every convex combination has the same G at the invariant. Temperature,
pressure and bulk inventory alone do not fix the amount along this segment;
an additional enthalpy or heat-removal condition would be needed. A prescribed
temperature path therefore has a jump, not an interval of three-phase
coexistence. Below the invariant the original inventory is all solid in
this phase set: **0.5 mol Fo + 0.5 mol MgSiO3**. Forsterite decreases during
the reaction instead of following the unstable Fo-only continuation.

``equilibrium(T, P=1e5, mgo_mol=1.5, sio2_mol=1.)`` returns
``amounts_mol`` in **[liquid Q, liquid F, solid Fo, solid En]** order,
``g_J``, ``degenerate``, and ``equilibrium_endpoints_mol``. It requires x64
and the same T/P/inventory domain as the Fo-only example. Distinct endpoints
whose energies differ by at most 1e-7 J per initial mol Si flag degeneracy;
``amounts_mol`` then selects the endpoint with least liquid as a convention,
and the full conserved segment is retained. The equilibrium selector itself
is not a JAX-transformable or differentiable phase solver.

.. list-table:: Equilibrated amounts at 1900 K and 1 bar
   :header-rows: 1

   * - Initial MgO / SiO2 (mol/mol)
     - Liquid [Q, F] (mol)
     - Fo (mol)
     - En (mol MgSiO3)
   * - 1.5 / 1
     - [0, 0]
     - 0.5
     - 0.5
   * - 0.9 / 1
     - [0.232967246, 0.132967246]
     - 0
     - 0.634065507

.. figure:: _static/magma/enstatite_equilibrium.png
   :alt: Closed cooling with Fo and En compared with MELTS property roots; liquid disappears for MgO/SiO2=1.5 but persists with En for 0.9.
   :width: 100%

   JAX lines and independent restricted equilibria using MELTS properties
   (circles). Dotted vertical segments represent the nonunique invariant
   amounts. Each plotted amount is mol Si in that phase: liquid Q+F, mol Fo,
   or mol MgSiO3. The 0.9/1 inventory consumes all Fo at the invariant and
   continues as liquid + En. Omitted silica solids can affect that path.

An absent liquid also needs a stability test. With both solids present,
their supporting chemical potentials are
:math:`\lambda_Q=2g_{\rm En}-g_{\rm Fo}`, :math:`\lambda_F=g_{\rm Fo}`.
Minimize :math:`g_l(x)-[(1-x)\lambda_Q+x\lambda_F]` over liquid composition
:math:`x=x_F`. A positive minimum means liquid formation raises G.
This line extends below x=1/2 as a chemical-potential reference; a physical
Fo+En mixture itself only spans 1/2<=x<=1.

.. figure:: _static/magma/enstatite_liquid_stability.png
   :alt: The minimum liquid energy relative to the Fo-En supporting line changes from negative to zero to positive on cooling.
   :width: 85%

   The liquid minimum is negative at 1925 K, zero at the invariant and
   positive at 1900 K. Circles are minima found independently from actual
   MELTS chemical potentials. This verifies liquid disappearance for the
   Fo+En inventory without differentiating an absent phase.

``tests/reference/magma_equilibrium_melts_v1.json`` stores 13 equilibria
for the two inventories and a separately solved invariant. Its generator
uses pinned MELTS liquid G/mu and pure olivine/orthopyroxene properties,
supporting-line stability, and SciPy roots. It imports neither the JAX
thermodynamic model nor this example's equilibrium selector. The comparison
is to the **same restricted phase set using MELTS properties**, not the
full MELTS phase-assemblage solver or an experimental phase diagram.
The maximum amount difference is 9.9e-12 mol, total-G difference 1.5e-5 J,
and invariant-temperature difference 3.1e-11 K in this run. Ordinary tests
read the saved fixture and check conservation, coexistence, disappearance,
invariant segments, scaling, and energies against a feasible 2-D grid.

.. code-block:: bash

   JAX_ENABLE_X64=1 MPLBACKEND=Agg python examples/magma_equilibrium.py \
       --output documents/_static/magma
   # Reference regeneration only; requires the pinned runtime and SciPy:
   python tests/reference/generate_magma_equilibrium_reference.py \
       --runtime /path/to/pinned/runtime --python /path/to/worker/python \
       --output tests/reference/magma_equilibrium_melts_v1.json

Silica polymorphs and multiple solids
-------------------------------------

``silica_gibbs`` extends the native JAX Gibbs model with quartz, tridymite
and cristobalite. Each includes its MELTS alpha/beta thermal branch and
Berman pressure integral. At pressure in bar, the switch temperatures are
848+0.0237(P-1), 383, and 535+0.0480(P-1) K, respectively; equality selects
the alpha branch. The low-temperature lambda heat capacity is integrated
from the source's shifted lower limit (373 K for quartz, 298.15 K for the
other two, before the pressure shift). These tabulated branch joins are
not smoothed or assumed to be exactly continuous. JAX derivatives describe
each branch, not a derivative through a structural switch or phase selection.

The pinned **rhyolite-MELTS 1.0.2** runtime has constant offsets from the
unadjusted public MAGMA silica formulas:

.. list-table:: Included standard-state offsets
   :header-rows: 1

   * - Polymorph
     - Offset (J/mol SiO2)
     - Evidence
   * - Quartz
     - -1291
     - Public ``RHYOLITE_ADJUSTMENTS`` in ``gibbs.c`` and runtime checks.
   * - Tridymite
     - -2625
     - Transcribed constant difference from the hash-pinned runtime.
   * - Cristobalite
     - -450
     - Transcribed constant difference from the hash-pinned runtime.

The last two offsets are **not present in the pinned public source**.
They are explicit runtime transcriptions, checked over 48 T/P states
including both sides of the alpha/beta switches, not adjustments fitted to
the cooling curves. Their source-build identity is unestablished.
Without these offsets the model is a different standard-state convention
and gives different phase selection. The runtime convention is fixed here;
there is no new calibration or empirical phase-diagram claim.

.. figure:: _static/magma/silica_melts_comparison.png
   :alt: Three SiO2 polymorph energies relative to quartz at 1, 500 and 5000 bar, with independent MELTS points and numerical residuals.
   :width: 100%

   Lines are native JAX, circles are actual pinned MELTS properties, and
   the lower panels show differences after including the declared offsets.
   The maximum absolute difference is 2.4e-10 J/mol. These pressure checks
   validate formulas, not experimental high-pressure applicability.

``examples/magma_multiphase.py`` minimizes the combined G with all five
solid candidates. Its amount order is **[liquid Q, liquid F, Fo, En,
quartz, tridymite, cristobalite]**, in mol of one-Si formula units. Write
:math:`s=\sum_j s_j` for the total silica-solid amount. Conservation gives

.. math::

   n_Q=b-\eta/2-s,\qquad n_F=a-\xi-\eta/2,\qquad
   G=G_l+\xi g_{\rm Fo}+\eta g_{\rm En}+\sum_j s_j g_j.

All amounts are nonnegative. Each liquid/single-solid edge is convex for
T>W/(2R_b), so bisection finds its minimum. Comparing those edges with
every feasible all-solid pair suffices: two conserved components and a
strictly convex liquid ensure that a multiphase minimum has an endpoint
in this list. Same-composition silica ties are retained as distinct
endpoints. There is no optimizer dependency or general solver in the package.

``equilibrium`` returns ``amounts_mol``, ``g_J``, ``phases``, ``liquid_x_F``,
``liquid_MgO_SiO2``, ``degenerate`` and ``equilibrium_endpoints_mol``.
The residual liquid oxide ratio is **2*x_F**, not an oxide mole fraction.
Both liquid-composition fields are ``None`` after the liquid disappears.
An equal-energy endpoint set describes its convex hull of permitted amounts;
the least-liquid endpoint is only a reporting convention. Ties use
1e-7 J per initial mol Si. ``phases`` lists amounts above 1e-10 of initial Si;
this reporting threshold does not enter the minimization.

``cooling_path`` independently equilibrates each decreasing temperature
with the full bulk inventory, retaining crystals. It returns states and
``events`` with ``appeared``, ``disappeared`` and ``T_bracket_K``. These are
sampled brackets; they do not assert an exact event temperature. This is
closed equilibrium cooling, not fractional crystallization or kinetics.

.. figure:: _static/magma/multiphase_cooling.png
   :alt: Four closed cooling paths show solid appearance, consumption of forsterite, liquid disappearance and residual liquid composition, with MELTS comparison points.
   :width: 100%

   Each inventory contains 1 mol SiO2 initially. Lines are JAX; circles are
   independently solved equilibria using MELTS properties with the same
   candidates. Shaded bands bracket sampled phase changes. No residual-liquid
   curve is drawn after solidification.

At 1 bar the predicted sequences on cooling from 2600 to 1600 K are:

.. list-table:: Crystallization in the declared model
   :header-rows: 1

   * - MgO / SiO2
     - Sequence (L: liquid, Trd: tridymite)
     - Solids at 1800 K (mol)
   * - 0.1
     - L -> L+Trd -> En+Trd
     - En 0.1, Trd 0.9
   * - 0.5
     - L -> L+En -> En+Trd
     - En 0.5, Trd 0.5
   * - 0.9
     - L -> L+Fo -> L+En -> En+Trd
     - En 0.9, Trd 0.1
   * - 1.5
     - L -> L+Fo -> Fo+En
     - Fo 0.5, En 0.5

The En+Trd+L invariant is **1818.067260 K**, with liquid
**x_F=0.208047861** (MgO/SiO2=0.416095723). Below it the silica-rich
inventories have no liquid. The Fo+En+L invariant remains 1908.603653 K.
The fixed runtime selects tridymite over cristobalite on these 1 bar paths;
cristobalite is evaluated and rejected by energy, not suppressed.
Its quartz/tridymite crossing is 582.694782 K. This unusually low model
crossing must not be read as an experimental silica transition. The lower
temperature standards are formula checks; extrapolated liquid behavior at
such temperatures is not a calibrated cooling prediction. Coesite,
stishovite, other MgSiO3 polymorphs and solid solutions remain omitted.

The saved ``silica_melts_v1.json`` contains 48 silica standard states,
four native finite-difference derivative checks, 21 independent restricted
equilibria and two transition roots. Its generator uses fresh MELTS workers,
SciPy roots and global supporting-line checks; it imports neither JAX
thermodynamics nor the example selector. Maximum differences in this run
are 9.9e-12 mol for amounts and 1.5e-5 J for total G. Comparisons concern
the same candidates using MELTS properties, not MELTS full phase selection.
Ordinary tests use saved values and verify conservation, phase stability,
invariant endpoint sets, scaling, derivatives and feasible simultaneous-solid
trial energies without an external runtime.

.. code-block:: bash

   JAX_ENABLE_X64=1 MPLBACKEND=Agg python examples/magma_multiphase.py \
       --output documents/_static/magma
   # Customize inventory, pressure (Pa) and sampling:
   JAX_ENABLE_X64=1 MPLBACKEND=Agg python examples/magma_multiphase.py \
       --mgo 0.9 --pressure 100000 --temperature-min 1700 --step 1
   # Reference regeneration only:
   python tests/reference/generate_silica_reference.py \
       --runtime /path/to/pinned/runtime --python /path/to/worker/python \
       --output tests/reference/silica_melts_v1.json

Fe(II), ternary liquid and olivine partitioning
------------------------------------------------------------

Olivine is the solid solution :math:`(\mathrm{Mg,Fe})_2\mathrm{SiO_4}`
between forsterite (Fo, Mg2SiO4) and fayalite (Fa, Fe2SiO4). Thus the new
crystallization example describes **olivine precipitation**, with a variable
crystal composition. All iron is Fe(II); there is no Fe(III), metal, redox
equilibrium, oxygen buffer or oxygen exchange.

The added Gibbs-energy model is the **Q--F--A restriction of the MELTS
liquid**, with pairwise regular-solution interactions, together with the
**Fo--Fa restriction of MELTS olivine**, with two cation sites and a symmetric
excess term. Both include temperature- and pressure-dependent standard
states. The equations and coefficients are ported from the pinned MAGMA
revision below; no new parameters are fitted to the partitioning results.

The liquid basis extends to :math:`[Q,F,A]=[\mathrm{SiO_2},\mathrm{Mg_2SiO_4},
\mathrm{Fe_2SiO_4}]`, exported as ``TERNARY_COMPONENTS``. For oxide amounts
M=MgO, E=FeO and S=SiO2 in mol,

.. math::

   [b,a,c]=[S-(M+E)/2,\ M/2,\ E/2].

A nonnegative basis requires M,E>=0 and M+E<=2S. The same extensive
``liquid_gibbs`` now accepts either two or three component amounts.
``liquid_standard_gibbs(T,P,include_fe=True)`` returns all three standards;
the default and ``COMPONENTS`` retain their original two-component order.
``include_fe`` is a static choice: use a closure or
``jax.jit(..., static_argnames=("include_fe",))``. Shape-(3,) with zero Fe
recovers the original energy, without a composition floor.

For the liquid, :math:`N_l=n_Q+n_F+n_A` and :math:`x_i=n_i/N_l`.
Its extensive energy separates into standard, ideal-mixing and excess terms:

.. math::

   G_l=\underbrace{\sum_{i=Q,F,A}n_i g_{i,l}^0(T,P)}_{G_l^{\mathrm{std}}}
       +\underbrace{R_bT\sum_{i=Q,F,A}n_i\ln x_i}_{G_l^{\mathrm{id}}}
       +\underbrace{N_l\sum_{i<j}W_{ij}x_ix_j}_{G_l^{\mathrm{E}}},

.. math::

   (W_{QF},W_{QA},W_{FA})=(3421,\ 23660.9,\ -37256.7)\ \mathrm{J/mol}.

Here :math:`R_b=8.3143` J/(mol K), G is in J, the standards are in J/mol,
and amounts are in mol. The liquid interaction coefficients are independent
of T and P. Setting :math:`n_A=0` recovers the earlier binary energy;
the new liquid terms are the A standard and mixing contribution and the
Q--A and F--A interactions. These are endmember fractions, not oxide fractions.

``olivine_gibbs(T,P,[n_Fo,n_Fa])`` returns extensive G in J. Its binary
reduction of the published ``olivine.c`` model uses
:math:`N_{ol}=n_{Fo}+n_{Fa}` and :math:`y_i=n_i/N_{ol}`:

.. math::

   G_{ol}=n_{Fo}g_{Fo}^0+n_{Fa}g_{Fa}^0
       +2R_bT\sum_{i=Fo,Fa}n_i\ln y_i
       +N_{ol}W_{ol}y_{Fo}y_{Fa},

.. math::

   W_{ol}=20300+0.015(P/10^5-1)\ \mathrm{J/mol}.

The first two terms are the crystalline standards, followed by the
two-site ideal mixing and the excess term; P is supplied in Pa.
There are two cation sites per formula unit. With only Fe/Mg at magmatic
temperatures their equilibrium Fe occupancies are equal, so the ideal
entropy has the factor **2**. The remaining interaction includes an excess
volume of :math:`0.015y_{Fo}y_{Fa}` J/(bar mol). No internal ordering solve
is needed on this face. These are single-phase properties, not a convex
envelope or a general low-temperature olivine unmixing calculation.
Both solid endpoints and absent phases have continuous energy limits;
composition derivatives require positive amounts in each present phase.

The new Fe standards are constructed explicitly as follows. At
:math:`T_r=298.15` K and 1 bar, the Fa reference has
:math:`H_r=-1479360` J/mol and :math:`S_r=150.930` J/(mol K):

.. math::

   \begin{aligned}
   C_{p,Fa}(T)&=248.93-1923.9/\sqrt{T}-1.3910\times10^8/T^3,\\
   h_{Fa}(T)&=H_r+\int_{T_r}^{T}C_{p,Fa}(\theta)\,d\theta,\\
   s_{Fa}(T)&=S_r+\int_{T_r}^{T}\frac{C_{p,Fa}(\theta)}{\theta}\,d\theta.
   \end{aligned}

The heat capacity is in J/(mol K); h and s denote the 1-bar thermal parts.
The crystalline Fa and liquid A standards are

.. math::

   \begin{aligned}
   g_{Fa}^0(T,P)&=h_{Fa}(T)-Ts_{Fa}(T)+\Pi_B(T,P),\\
   g_{A,l}^0(T,P)&=h_{Fa}(T_m)-Ts_{Fa}(T_m)+(T_m-T)\Delta S_f\\
       &\quad+C_{p,l}\left[T-T_m-T\ln(T/T_m)\right]+\Pi_K(T,P).
   \end{aligned}

Here the 1-bar fusion reference is :math:`T_m=1490` K,
:math:`\Delta S_f=59.9` J/(mol K), and the constant liquid
:math:`C_{p,l}=240.2` J/(mol K). Define :math:`\Delta T_s=T-298.15`,
:math:`\Delta T_l=T-1673`, and :math:`\Delta p=P/10^5-1`, using numerical
T in K and P in Pa. The integrated Berman and Kress pressure terms, in J/mol,
are respectively

.. math::

   \begin{aligned}
   \Pi_B&=4.630\left[
       (1+26.546\times10^{-6}\Delta T_s+79.482\times10^{-10}\Delta T_s^2)
       \Delta p-\frac{0.730\times10^{-6}}{2}\Delta p^2\right],\\
   \Pi_K&=(5.420+5.84\times10^{-4}\Delta T_l)\Delta p\\
       &\quad+\frac{-2.79\times10^{-5}-2.3\times10^{-8}\Delta T_l}{2}\Delta p^2
       +\frac{14.6\times10^{-10}}{6}\Delta p^3.
   \end{aligned}

The pressure increment is evaluated in bar; the volume coefficients use
J/(bar mol). Thus both terms vanish at 1 bar, and differentiation with
respect to API pressure in Pa returns SI volume.
``fayalite_gibbs`` uses the **olivine endmember's Berman EOS**; the separate
pure-fayalite phase in MELTS uses a Vinet EOS. The Q and F liquid standards
and the crystalline Fo standard retain the construction described above.

Retaining the crystals, :math:`u` mol Fo and :math:`v` mol Fa in one olivine
leave liquid :math:`[b,a-u,c-v]`. Minimize

.. math::

   G_{total}(u,v)=G_l(T,P,[b,a-u,c-v])+G_{ol}(T,P,[u,v]),
   \quad 0\le u\le a,\quad 0\le v\le c.

The same model gives :math:`\mu_i=(\partial G/\partial n_i)_{T,P,n_{j\ne i}}`.
Writing :math:`g_l^{\mathrm{E}}=\sum_{i<j}W_{ij}x_ix_j`, its explicit potentials are

.. math::

   \begin{aligned}
   \mu_{i,l}&=g_{i,l}^0+R_bT\ln x_i+\sum_{j\ne i}W_{ij}x_j-g_l^{\mathrm{E}},\\
   \mu_{Fo,ol}&=g_{Fo}^0+2R_bT\ln y_{Fo}+W_{ol}y_{Fa}^2,\\
   \mu_{Fa,ol}&=g_{Fa}^0+2R_bT\ln y_{Fa}+W_{ol}y_{Fo}^2.
   \end{aligned}

Here :math:`W_{ji}=W_{ij}`. The constrained total G conserves Mg, Fe, Si, O
and mass. Interior coexistence requires
:math:`\mu_{Fo,ol}=\mu_{F,l}` and :math:`\mu_{Fa,ol}=\mu_{A,l}`.
The resulting exchange coefficient is an **output**, not a prescribed 0.3:

.. math::

   K_D^{Fe/Mg}=\frac{(Fe/Mg)_{ol}}{(Fe/Mg)_l}
       =\frac{v/u}{(c-v)/(a-u)}.

``examples/magma_olivine.py::equilibrium`` uses two nested bisections of
derivatives of this same JAX G. At fixed total crystal amount it equilibrates
Fe/Mg exchange, then equilibrates the crystal amount. Before creating an
absent solid it minimizes its insertion energy over Fo/Fa composition.
The example requires positive Q, F and A inventories, T>=1600 K and
1<=P/bar<=5000. The ternary liquid is strictly convex above 1539.756 K
(a composition-independent curvature bound for these W values); olivine
is also strictly convex in the example's domain. These restrictions support
the unique minimum and are not empirical calibration limits. Positive Q
ensures some liquid remains. Pure endpoints are supported by the property
API, while the example deliberately excludes zero-inventory faces and
all-solid/degenerate cases. The eager selector is not a differentiable
equilibrium API.

At 1 bar with M=1.2, E=0.3 and S=1 mol, olivine first appears at
**1987.882448 K**. Results below describe a closed system in which existing
crystals continue to equilibrate, not fractional crystallization:

.. list-table:: Fe/Mg partitioning from total G
   :header-rows: 1

   * - T (K)
     - Olivine (mol)
     - Liquid Fe/(Mg+Fe)
     - Olivine Fe/(Mg+Fe)
     - Kd
   * - 2000
     - 0
     - 0.200000
     - undefined
     - undefined
   * - 1900
     - 0.292098
     - 0.275259
     - 0.082021
     - 0.235253
   * - 1800
     - 0.496054
     - 0.347613
     - 0.124432
     - 0.266718
   * - 1600
     - 0.705063
     - 0.344209
     - 0.190809
     - 0.449252

.. figure:: _static/magma/olivine_partitioning.png
   :alt: JAX cooling curves and independent MELTS points agree for phase amounts, Fe fractions and the changing Fe/Mg exchange coefficient.
   :width: 90%

   Cooling proceeds to the right; the dashed line is olivine onset. Olivine
   is Mg-rich relative to the liquid. Fe enrichment of the residual liquid
   is not monotonic because retained crystals reequilibrate as Kd changes.
   The liquid's Fe/(Mg+Fe) is x_A/(x_F+x_A), not x_A. Amounts in the upper
   panel are mol Si and sum to one. Absent-olivine composition and Kd are
   returned as ``None``; incipient composition is a separate diagnostic.

.. figure:: _static/magma/olivine_melts_comparison.png
   :alt: Signed JAX minus MELTS energy differences are about 1e-5 J for the ternary liquid and 1e-9 J for olivine at three pressures.
   :width: 85%

   Supplied liquid [Q,F,A]=[0.25,0.6,0.15] mol and olivine [Fo,Fa]=[0.8,0.2]
   mol, at 1, 500 and 5000 bar. Differences are shown explicitly rather
   than hidden by the roughly megajoule absolute energy scale.

``olivine_melts_v1.json`` records **15 distinct liquid states, 60 olivine
states including both endpoints, and 30 independent restricted equilibria**
for two bulk Fe/Mg ratios. The generator uses actual hash-pinned alphaMELTS
2.3.2 / rhyolite-MELTS 1.0.2 properties, verifies oxide/component inventories,
disables oxygen buffering, and solves native chemical-potential roots using
SciPy. It imports neither JAX thermodynamics nor the example solver.
Maximum observed differences are 1.25e-5 J for liquid/total G, 1.4e-9 J for
olivine G, 1.1e-11 mol for phase amounts and 2.7e-12 for Kd. These finite
checks concern the **same liquid + olivine phase set**, not MELTS full phase
selection or validation of an experimental ternary phase diagram.
Pyroxenes (including Fe-bearing ones), silica solids and all other candidates
are omitted from this new equilibrium example; they can destabilize its path.
The earlier Fe-free examples keep their original phase sets.

.. code-block:: bash

   JAX_ENABLE_X64=1 MPLBACKEND=Agg python examples/magma_olivine.py \
       --output documents/_static/magma
   # Optional independent reference regeneration; SciPy + pinned runtime:
   python tests/reference/generate_olivine_reference.py \
       --runtime /path/to/pinned/runtime --python /path/to/worker/python \
       --output tests/reference/olivine_melts_v1.json

Normal tests require no MELTS or SciPy. They check independent reference
energies/potentials/partitioning, conservation, coexistence, an independent
feasible composition grid, onset, scaling, zero-phase and pure-endmember
energies, two-site entropy, pressure units, JIT/vmap and joint derivatives.
The additional source is
`olivine.c at the same MAGMA revision <https://github.com/magmasource/MAGMA/blob/705a0fb315e5054d18275a580562f6121c8e458c/sources/olivine.c>`_
(SHA256 ``872fa74c5c7601dccdb41560c70a996af558917ff1e82589e277dd459a93a080``).

Provenance and independent checks
---------------------------------

The port uses MAGMA revision ``705a0fb315e5054d18275a580562f6121c8e458c``:
`standard-state equations <https://github.com/magmasource/MAGMA/blob/705a0fb315e5054d18275a580562f6121c8e458c/sources/gibbs.c>`_,
`liquid coefficients <https://github.com/magmasource/MAGMA/blob/705a0fb315e5054d18275a580562f6121c8e458c/includes/liq_struct_data.h>`_,
`solid coefficients <https://github.com/magmasource/MAGMA/blob/705a0fb315e5054d18275a580562f6121c8e458c/includes/sol_struct_data.h>`_, and
`liquid interaction <https://github.com/magmasource/MAGMA/blob/705a0fb315e5054d18275a580562f6121c8e458c/includes/param_struct_data_v34.h>`_.

``tests/reference/magma_source_v1.json`` records independent numerical
integration of the published Cp, Cp/T and V, with source hashes.
It tests G, entropy, heat capacity and volume at 11 T/P states, including
both sides of 1480 K. This is source-formula validation, not a compiled
MAGMA run. Existing saved MELTS liquid standards are also checked.

``tests/reference/magma_melts_v1.json`` separately records 13 supplied
states from the hash-pinned alphaMELTS 2.3.2 / rhyolite-MELTS 1.0.2 runtime.
The maximum absolute standard-state differences are 4.7e-10 J/mol for the
liquids and 1.4e-9 J/mol for solid F. Maximum differences for mixed G and
interior chemical potentials are 1.7e-5 J and 2.4e-5 J/mol, respectively
(test tolerance 3e-5). Pure F liquid oxide conversion fails in the pinned
worker with negative SiO2; that unavailable case is retained explicitly.
The JAX endpoint is covered by the continuous energy and standard-state checks.

Regenerate either independent fixture without importing the JAX model:

.. code-block:: bash

   python tests/reference/generate_magma_reference.py \
       --output tests/reference/magma_source_v1.json
   python tests/reference/generate_magma_reference.py --runtime /path/to/pinned/runtime \
       --python /path/to/worker/python --output tests/reference/magma_melts_v1.json

Only the second command needs MELTS and its Python worker dependencies.
Ordinary tests read the saved JSON files. They also check composition
derivatives, extensivity, elemental/mass conservation, onset, coexistence,
energy boundaries, jit/vmap, and joint T/P/composition differentiation.
The public source revision's identity with the distributed thermodynamic C
build remains unestablished. Formula consistency, these finite binary
comparisons, and experimental phase-diagram validation are distinct claims.
