Native JAX magma: MgO--SiO2
===========================

``exoeos.magma`` evaluates a two-component MELTS liquid and pure crystalline
forsterite entirely in JAX, including their standard Gibbs energies.
No external MELTS runtime is needed for evaluation, derivatives, the cooling
example, or ordinary tests. This first model restricts equilibrium to one
liquid plus forsterite; it does not reproduce the full MgO--SiO2 phase diagram.

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

The three functions have scalar ``T`` in K and ``P`` in Pa:

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

Enstatite and silica solids are omitted. Their insertion conditions and
competing stability must be examined before interpreting this as the
physical binary phase diagram. MgO/SiO2>2 requires a different basis.

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
