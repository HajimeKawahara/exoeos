Thermodynamically consistent table interpolation
============================================================

The original ``ChabrierDebrasEOS`` interpolates each published field
independently. It remains the backend for reproducing the source table's
values and response columns. A separate, opt-in ``HelmholtzTable`` constructs
one total **mass-specific** Helmholtz free energy and derives pressure,
entropy and response quantities from it. Its interpolation is thermodynamically
consistent, but this alone does not establish physical accuracy or stability.
The H/He reconstruction is experimental and requires checking its residuals
and local stability over the intended application domain.

No mean molar mass, molecular species assignment or composition derivatives
are inferred from a fixed-composition table. The new state uses kg/m3 and
J/kg, whereas the analytic ``MolarHelmholtzEOS`` contract uses mol/m3 and
J/mol. Both paths apply the same thermodynamic derivative identities in
their declared basis.

Using the two H/He backends
------------------------------------------------------------

Enable JAX 64-bit mode before loading the source table. Construction and
file access run on the host; subsequent potential/state evaluation uses JAX.

.. code-block:: python

   import jax
   from exoeos import ChabrierDebrasTableLoader

   jax.config.update("jax_enable_x64", True)
   original = ChabrierDebrasTableLoader(variant="Y0275").load()
   potential = original.to_helmholtz()

   T, rho = 1.0e4, 1.0e3  # K, kg/m3
   a = potential.specific_helmholtz(T, rho)  # J/kg
   state = potential.state_trho(T, rho)
   residuals = original.helmholtz_residuals(potential, T, rho)
   state.P, state.s, state.cv, state.cp, state.is_stable
   residuals.P, residuals.u, residuals.s, residuals.nabla_ad

``state_trho`` returns ``MassHelmholtzThermodynamicState``. The residual
method returns a ``MassThermodynamicState`` containing signed differences,
**potential minus original T--density interpolation**, for all nine original
fields. These include the entropy-pressure derivative and adiabatic gradient,
which are not imposed as independent nodal constraints. Relative residuals
require a caller-selected scale; division by a nearly zero response is
usually uninformative. Residual evaluation supports ``jax.jit`` and
``jax.vmap``.

The potential's ``state_tp(T, P, density_bounds=(lo, hi))`` inverts its own
pressure and requires an explicit density bracket in kg/m3. Supply a
monotonic single-phase bracket containing the desired simple root. It does
not interpolate the independent source TP table or search for the stable
phase. Invalid or unbracketed queries, and roots with nonpositive
:math:`P_\rho`, return NaNs. Root derivatives use the
implicit pressure equation, so they remain derivatives of the potential
backend. The original TP table remains a separate reference.

For another rectangular fixed-composition table, construct
``HelmholtzTable.from_helmholtz(temperatures, mass_densities, a)`` from
specific Helmholtz energies. Alternatively,
``HelmholtzTable.from_thermodynamic_data(temperatures, mass_densities, P, u, s)``
uses the nodal constraints below. Arrays are temperature-major with shape
``(n_temperature, n_density)``. Optional source response columns
``dlnrho_dlnT_P``, ``dlnrho_dlnP_T`` and ``dlns_dlnT_P`` constrain second
derivatives when supplied together. ``to_helmholtz()`` supplies these from
the Chabrier--Debras T--density table.
Each axis must contain at least three finite, positive, strictly increasing
coordinates; all input fields must be finite and have the declared shape.
Finite differences are second order, with one-sided boundary estimates.

Potential reconstruction and interpolation
------------------------------------------------------------

When a table supplies specific internal energy :math:`u` and entropy
:math:`s`, its nodal Helmholtz energy follows algebraically:

.. math::

   a_{ij}=u_{ij}-T_i s_{ij}.

This preserves the source's energy and entropy reference conventions.
It does not make independently tabulated pressure or response columns
derivatives of the reconstructed energy. Their agreement must be measured
after interpolation. Tables without a compatible complete set of inputs
require an additional reconstruction model and are not automatically fitted.

The interpolated quantity is :math:`q=a/T` on natural logarithmic
coordinates :math:`t=\ln T` and :math:`r=\ln\rho`, where :math:`\rho` is
mass density. A tensor-product **biquintic Hermite** polynomial uses one
shared set of nodal derivatives through order two in each coordinate.
Adjacent cells therefore match the potential and all its derivatives
through total order two. First- and second-derivative thermodynamic
quantities are continuous across interior cell boundaries where their
denominators are nonsingular; third derivatives
need not be continuous.

For thermodynamic source data, the first nodal derivatives are constrained by

.. math::

   q=u/T-s,\qquad q_t=-u/T,\qquad q_r=P/(\rho T).

Thus the reconstructed potential and its first derivatives recover the
supplied :math:`u`, :math:`s` and :math:`P` at nodes in exact arithmetic.
Evaluation subtracts a nearby nodal constant from :math:`q` before the
polynomial contraction and adds it back afterward. This preserves the same
interpolant while reducing cancellation from extreme entropy offsets.
Higher mixed derivatives are estimated locally and shared by
neighboring cells. This is a constrained nodal interpolation, not a
statistical fit or an imposed stability correction.

Bilinear potential interpolation cannot provide continuous first
derivatives. Bicubic Hermite interpolation is generally only :math:`C^1`,
so its second-derivative responses can jump. The selected biquintic
construction is :math:`C^2` and local. In the numerical method comparison,
a global cubic spline was sensitive to extreme values elsewhere in the
source rectangle; local interpolation avoids that nonlocal contamination.
The use of Helmholtz derivatives and biquintic interpolation follows the
approach described by
`Swesty (1996) <https://doi.org/10.1006/jcph.1996.0162>`_ and
`Timmes and Swesty (2000) <https://doi.org/10.1086/313304>`_.
The log-coordinate reconstruction and its source constraints here are
specific to this implementation.

Mass-specific derivative identities
-----------------------------------

All derivatives hold the table's composition fixed. With subscripts denoting
temperature and mass-density derivatives, the new backend evaluates

.. math::

   P &= \rho^2 a_\rho, & s &= -a_T, &
   u &= a+Ts, & h &= u+P/\rho, & g &= a+P/\rho, \\
   P_T &= \rho^2 a_{T\rho}, &
   P_\rho &= 2\rho a_\rho+\rho^2 a_{\rho\rho}, \\
   c_v &= -T a_{TT}, &
   c_p &= c_v+\frac{T P_T^2}{\rho^2 P_\rho}, \\
   c_s^2 &= P_\rho+\frac{T P_T^2}{\rho^2 c_v}, &
   \nabla_{\rm ad} &= \frac{P P_T}{\rho^2 P_\rho c_p}.

Energies use J/kg, entropy and heat capacities use J/(kg K), pressure uses
Pa, and sound speed uses m/s. In particular, no mean-molar-mass factor is
needed in the sound-speed formula. The Maxwell identity
:math:`s_\rho=-P_T/\rho^2` follows from the same potential. Other response
identities are given in :doc:`thermodynamic_derivatives`, using its molar
basis consistently.

Accuracy, stability and scope
-----------------------------

Compare potential-derived quantities against the original backend at both
nodes and cell interiors. A small interpolation residual is not an
experimental uncertainty estimate. Source rounding, source inconsistency,
and local derivative estimates can all affect the reconstruction.

The reproducible
:download:`validation program <../examples/validate_helmholtz_table.py>`
records source residuals, consistency checks and stability counts in
:download:`validation.json <../results/helmholtz_table/validation.json>`.
The report separates consistency checks from source fidelity.

.. code-block:: bash

   JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python examples/validate_helmholtz_table.py \
       --cache-directory "$HOME/.cache/exoeos/DirEOS2021" \
       --output results/helmholtz_table/validation.json

Across all three variants and the full source grid, maximum relative nodal
residuals are below :math:`6.1\times10^{-13}` for pressure,
:math:`3.2\times10^{-13}` for internal energy and
:math:`3.7\times10^{-10}` for entropy. Sampled, scaled Maxwell, heat-capacity
and first-law identity residuals are below :math:`3.84\times10^{-12}`.
These arithmetic checks do not bound differences inside cells.

The following diagnostics use all 1,254 logarithmic cell centers per variant
in :math:`1000\leq T\leq80000` K and
:math:`200\leq\rho\leq9000` kg/m3. Pressure differences are
:math:`100|P_{\rm potential}-P_{\rm original}|/|P_{\rm original}|`.
This sampling box is not a certified validity domain.

.. list-table:: Source pressure residuals and local thermal stability
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Variant
     - Median (%)
     - 95th percentile (%)
     - Maximum (%)
     - :math:`c_v\leq0` count
   * - Y0275
     - 1.19
     - 13.75
     - 31.71
     - 13 / 1254
   * - Y0292
     - 1.28
     - 13.78
     - 32.18
     - 13 / 1254
   * - Y0297
     - 1.35
     - 13.78
     - 32.26
     - 14 / 1254

All these sampled pressures and pressure-density derivatives are positive.
The negative heat capacities show why smooth, consistent interpolation must
not be assumed stable. The report also retains all nine source-field
residuals, full-rectangle diagnostics and independent TP-table comparisons.
Entropy is more sensitive: its 95th-percentile relative residual is
73.1--83.2%, and its maximum is 2903--3141% across variants; 34--41 sampled
points have nonpositive reconstructed entropy. These large departures limit
the reconstruction's practical use even in this diagnostic box.

Stable homogeneous states require at least :math:`c_v>0` and
:math:`P_\rho>0`. Potential consistency does not enforce either condition.
The Chabrier--Debras source rectangle already contains cautioned regimes
and numerical artifacts, and the reconstructed model must be screened
separately. No convexification, phase equilibrium or H/He demixing is
performed. The original backend remains available for every comparison.

The closed source rectangle is the numerical domain. Out-of-domain or
nonfinite inputs return NaNs; no extrapolation or clipping is performed.
For T--density calls, unstable interior responses are retained and ``state.is_stable`` reports
the two local sign conditions above. This flag does not certify global phase
stability. Singular response denominators can produce infinities or NaNs.
Evaluation accepts one scalar state; use ``jax.vmap`` for batches and
``jax.jit`` for compiled evaluation. Third-order automatic derivatives are
piecewise defined and can jump at cell boundaries.

The composition-dependent Marcum silicate--hydrogen backend retains its
published-field interpolation. Its nonrectangular TP/composition grid is
not converted to the fixed-composition T--density Helmholtz contract.
