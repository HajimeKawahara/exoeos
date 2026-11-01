Material chemical potentials and local sensitivity
==================================================

``gibbs_response(gibbs_j, temperature_k, pressure_pa, amounts_mol)`` evaluates
an extensive Gibbs energy and its component chemical potentials using JAX.
The same entry point accepts a liquid, alloy, gas, or custom material scalar.
It preserves the supplied model and component basis.

The callback contract is ``gibbs_j(T, P, n) -> G``: temperature in K, absolute
pressure in Pa, component amounts in mol, and scalar energy in J. Both T and P
are scalars; amounts form a nonempty vector. The returned ``GibbsResponse`` is
a JAX PyTree with fields ``gibbs_j`` [J] and ``chemical_potentials_j_mol``
[J/mol of the input component coordinates].

.. math::

   \mu_i = \left.\frac{\partial G}{\partial n_i}\right|_{T,P,n_{j\ne i}}.

For a homogeneous extensive scalar, scaling all amounts by the same factor
scales G while preserving the chemical potentials. The API does not normalize
amounts, select phases, clip values, or insert an amount floor. The caller
supplies the material's standard-state convention and checks its supported
domain. The helper checks static shapes; finite returned values alone do not
certify material validity or equilibrium stability.

An ideal example
----------------

This example includes T-dependent standard potentials and a simple linear
pressure term, so both dependencies remain inside the differentiated scalar.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from exoeos import gibbs_response
   from exoeos.constants import MOLAR_GAS_CONSTANT as R

   jax.config.update("jax_enable_x64", True)

   def energy(T, P, n):
       standards = jnp.array([17., -29.])*T + jnp.array([2e-6, 3e-6])*P
       return n @ standards + R*T*jnp.sum(n*jnp.log(n/n.sum()))

   evaluate = jax.jit(lambda T, P, n: gibbs_response(energy, T, P, n))
   inputs = (jnp.array(2100.), jnp.array(1e7), jnp.array([0.6, 1.4]))
   state = evaluate(*inputs)

This example's chemical potentials are exactly
:math:`\mu_i=\mu_i^0(T,P)+RT\ln(n_i/\sum_j n_j)`.
It requires positive amounts. Zero amounts in logarithmic mixing laws,
absent phases, and unsupported component faces need model-specific handling;
the evaluator does not manufacture a derivative for them.

Directional and reverse responses
---------------------------------

Use JAX transformations on the evaluator. No separate material solver or
finite-difference step is needed for the derivative itself.

.. code-block:: python

   direction = (jnp.array(13.), jnp.array(2e5), jnp.array([0.03, -0.02]))
   state, tangent = jax.jvp(evaluate, inputs, direction)
   dmu = tangent.chemical_potentials_j_mol

The tuple describes a derivative per unit of a chosen input coordinate:
13 K, 200000 Pa, and the specified mol changes per unit coordinate. Thus
``dmu`` has units J/mol per unit coordinate. It includes

.. math::

   d\mu = G_{nT}\,dT + G_{nP}\,dP + G_{nn}\,dn.

With ``dT = dP = 0``, this evaluates a Hessian-vector product without requesting
the full Hessian. ``jax.vjp`` applies the transpose of the same local response:

.. code-block:: python

   def potentials(T, P, n):
       return evaluate(T, P, n).chemical_potentials_j_mol

   mu, pullback = jax.vjp(potentials, *inputs)
   weights = jnp.array([1., -1.])
   dT_weight, dP_weight, dn_weights = pullback(weights)

These outputs are gradients of the weighted chemical-potential difference
with respect to T, P, and n. Their units follow the chosen output weights
and input units. ``jax.vmap`` supplies explicit batching. Bind the Gibbs
callback before ``jax.jit`` or mark that argument static.

Using existing material scalars
-------------------------------

An existing three-argument extensive scalar, such as ``magma.liquid_gibbs``,
can be passed directly. A scalar with additional arguments can be wrapped:

.. code-block:: python

   from exoeos import phosphorus_mqm

   def pair_energy(T, P, pairs):
       return phosphorus_mqm.gibbs_j(
           T, pairs, jnp.zeros(2), model="you2020")

   pair_state = gibbs_response(
       pair_energy, 2173.15, 1e5, jnp.array([1.8, 0.15, 0.4]))

This example uses zero pure standards to isolate the binary mixing scalar;
it does not define standards for a coupled reaction calculation. The pressure
derivative is zero because this particular binary scalar is pressure
independent. Chemical potentials refer to the supplied FeFe, PP and FeP
coordinates. Conversion to atomic Fe/P potentials must respect the model's
atom matrix; a mole of a pair coordinate is not necessarily a mole of atoms.

For callbacks using material parameters that depend on T or P, evaluate
those parameters **inside** the callback. Capturing standards evaluated at
one fixed state instead intentionally differentiates at fixed standards.
An outer JAX transform can also differentiate captured material parameters.

Material response versus equilibrium response
---------------------------------------------

The derivatives above hold the supplied material coordinates independent.
If a model includes internal pair or site variables, those variables relax
only if the callback already implements their differentiable relaxation,
or if the caller composes the material response with a separately computed
equilibrium tangent. This API does not infer a relaxation rule.

ExoGibbs owns local constrained equilibrium and its fixed-support derivative.
ExoInventory owns the coupled reservoir balance and pressure response.
Material-term attribution along such an equilibrium response can use this
evaluator, while a chemistry-specific decomposition remains in the consumer
example. All derivatives require the relevant smooth branch. They do not
establish phase-boundary differentiability, global stability, or validity
outside the material's calibrated domain.

Validation
----------

``tests/unittests/material_response_test.py`` checks analytic ideal-mixture
T/P/amount responses; nonideal central differences; JVP/VJP transpose
consistency; extensive scaling over forty orders of magnitude; the MQM pair
coordinate basis; captured parameter derivatives; and JIT/batching behavior.
The helper changes no thermodynamic scalar, standard state, or model parameter.

.. autofunction:: exoeos.gibbs_response

.. autoclass:: exoeos.GibbsResponse
