Thermodynamics from Helmholtz derivatives
=========================================

``thermodynamic_state_trho(model, T, rho, x)`` generates caloric properties
and response functions from the first and second temperature/density
derivatives of one **total molar Helmholtz free energy**. Its
``HelmholtzThermodynamicState`` is an immutable JAX PyTree. All derivatives
hold composition fixed and describe a single homogeneous phase.

Supplying a complete free energy
--------------------------------

The structural ``MolarHelmholtzEOS`` protocol requires
``molar_helmholtz(T, rho, x)`` in J/mol and component ``molar_masses`` in
kg/mol. Temperature is a scalar in K, ``rho`` is a scalar **molar** density
in mol/m3, and ``x`` is a normalized mole-fraction vector of shape ``(K,)``.
A custom model can provide any twice-differentiable total free energy that
meets this contract.

Fixed-composition mass-specific tables have a separate
:doc:`potential-consistent interpolation backend <potential_tables>`.
It applies the same identities in the mass basis without assigning molar
masses or introducing composition derivatives.

For existing residual EOS models, ``HelmholtzThermodynamics(residual, ideal)``
constructs

.. math::

   a(T,\rho,x) = a^0(T,\rho,x) + RT\alpha^r(T,\rho,x).

``ideal`` supplies ``molar_helmholtz`` and ``molar_masses`` in the same
component order as ``residual``. Its density dependence must obey
:math:`\partial a^0/\partial\rho=RT/\rho` so pressure inversion through the
residual model remains consistent. ``IdealGas`` implements this closure
with constant component :math:`c_p`, reference enthalpies/entropies and
ideal mixing. A custom ideal free energy can instead represent
temperature-dependent heat capacities. A residual potential alone cannot
determine total enthalpy, entropy or heat capacities; the ideal contribution
and its reference convention are required.

The wrapper provides ``state_trho(T, rho, x)`` and, when the residual model
provides pressure inversion, ``state_tp(T, P, x, phase="vapor")``.
``state`` is an alias for the latter. Root selection belongs to the residual
EOS. Pressure in the returned state is evaluated from the total potential
at that root. Fugacity remains available through the existing residual
``state_trho``/``state_tp`` API and is not a field of this caloric state.

Derivative identities
---------------------

Write :math:`a_T`, :math:`a_\rho`, :math:`a_{TT}`, :math:`a_{T\rho}` and
:math:`a_{\rho\rho}` for derivatives at fixed composition, and let
:math:`M=\sum_i x_i M_i`. The engine evaluates

.. math::

   s &= -a_T, & u &= a + Ts, & h &= u + P/\rho, & g &= a + P/\rho, \\
   P &= \rho^2 a_\rho, &
   P_T &= \rho^2 a_{T\rho}, &
   P_\rho &= 2\rho a_\rho + \rho^2 a_{\rho\rho}, \\
   c_v &= -T a_{TT}, &
   c_p &= c_v + \frac{T P_T^2}{\rho^2 P_\rho}.

Here :math:`P_T=(\partial P/\partial T)_{\rho,x}` and
:math:`P_\rho=(\partial P/\partial\rho)_{T,x}`. The volumetric expansion
coefficient, compressibilities, sound speed and adiabatic gradient follow
without further free-energy differentiation:

.. math::

   \alpha_P &= \frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_{P,x}
              = \frac{P_T}{\rho P_\rho}, \\
   \kappa_T &= -\frac{1}{V}\left(\frac{\partial V}{\partial P}\right)_{T,x}
              = \frac{1}{\rho P_\rho}, &
   \kappa_S &= \kappa_T\frac{c_v}{c_p}, \\
   c_s^2 &= \left(\frac{\partial P}{\partial\rho_{\rm mass}}\right)_{s,x}
          = \frac{1}{M}\left(P_\rho + \frac{T P_T^2}{\rho^2 c_v}\right), \\
   \nabla_{\rm ad} &= \left(\frac{\partial\ln T}{\partial\ln P}\right)_{s,x}
                    = \frac{P P_T}{\rho^2 P_\rho c_p}.

Also :math:`Z=P/(\rho RT)`, :math:`\rho_{\rm mass}=\rho M`, and
:math:`n=\rho N_A`. These are the usual Helmholtz derivative identities;
the `teqp derivative documentation <https://teqp.readthedocs.io/en/latest/derivs/derivs.html>`_
gives equivalent relations using reduced Helmholtz derivatives.

.. list-table:: Output units and selected aliases
   :header-rows: 1
   :widths: 45 20 35

   * - Quantity
     - Alias
     - SI unit
   * - Molar Helmholtz, internal, enthalpy and Gibbs energies
     - ``a``, ``u``, ``h``, ``g``
     - J/mol
   * - Molar entropy, constant-volume and constant-pressure heat capacities
     - ``s``, ``cv``, ``cp``
     - J/(mol K)
   * - Molar density, pressure
     - ``rho``, ``P``
     - mol/m3, Pa
   * - Compressibility factor, adiabatic gradient
     - ``Z``, ``nabla_ad``
     - Dimensionless
   * - ``sound_speed``, ``sound_speed_squared``
     -
     - m/s, m2/s2
   * - ``isothermal_compressibility``, ``isentropic_compressibility``
     -
     - 1/Pa
   * - ``thermal_expansion``
     -
     - 1/K
   * - ``pressure_temperature_derivative``, ``pressure_density_derivative``
     -
     - Pa/K, Pa m3/mol

Atmospheric and RCE use
-----------------------

This example uses an ideal H2/He closure with illustrative constant heat
capacities. Replace ``IdealEOS`` with a compatible residual model to include
its non-ideal response. A temperature-dependent ideal closure is needed
where constant heat capacities are inadequate.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from exoeos import HelmholtzThermodynamics, IdealEOS, IdealGas

   ideal = IdealGas(
       molar_masses=jnp.array([2.01588e-3, 4.002602e-3]),
       molar_heat_capacities=jnp.array([28.84, 20.786]),
   )
   eos = HelmholtzThermodynamics(IdealEOS(), ideal)
   x = jnp.array([0.85, 0.15])
   temperatures = jnp.array([800.0, 1000.0])
   pressures_bar = jnp.array([0.1, 1.0])

   profile = jax.jit(jax.vmap(
       lambda T, P: eos.state_tp(T, P, x, phase="vapor")
   ))(temperatures, pressures_bar * 1.0e5)

   cp_mass = profile.cp / profile.mean_molar_mass  # J/(kg K)
   adiabatic_gradient = profile.nabla_ad
   sound_speed = profile.sound_speed              # m/s

For one known-density state, use ``eos.state_trho(T, rho, x)``. For an
independent complete free-energy model, use
``thermodynamic_state_trho(model, T, rho, x)`` directly, including with
``IdealGas``. The engine and wrapper evaluate one state per call; use
``jax.vmap`` for profiles and keep phase strings static under ``jax.jit``.
Register custom model parameters as PyTree leaves when transforming them.

These responses hold composition fixed. They do not include the effects of
chemical re-equilibration, dissociation, condensation or latent heat in an
equilibrating atmosphere. An RCE caller must supply those effects through
its chosen closure. All state quantities use SI: convert upstream pressure
in bar to Pa, and divide molar heat capacity by ``mean_molar_mass`` to obtain
mass-specific heat capacity.

Domain and numerical behavior
-----------------------------

Supply positive temperature/density/molar masses, normalized nonnegative
mole fractions, and a twice-differentiable free energy within its declared
model domain. Stable homogeneous response normally requires
:math:`c_v>0` and :math:`P_\rho>0`. Inputs and responses are not clipped,
renormalized or stabilized: singular denominators can produce infinities,
negative :math:`c_s^2` produces a NaN sound speed, and invalid model inputs
can propagate NaNs. No phase-equilibrium or stability search is performed.

Further automatic differentiation requires the corresponding higher
derivatives of the model and, for TP calls, a differentiable density root.
Critical/spinodal roots need particular care. Use JAX 64-bit mode for
quantitative derivative checks or poorly conditioned dense-fluid states.
The :download:`state contract <thermodynamic_state_contract.md>` records
the complete field names and how this state differs from residual and
tabulated states.
