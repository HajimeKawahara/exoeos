Supplied-composition external MELTS evaluator
============================================================

``examples/melts_liquid_evaluator.py`` evaluates a requested liquid composition
with the same pinned alphaMELTS 2.3.2 / rhyolite-MELTS 1.0.2 runtime as
:doc:`melts_silicate_reference`. Its separate model identifier is
``alphamelts_2_3_2_rhyolite_melts_1_0_2_supplied_liquid_v1``.
It calls the external property methods directly; it does not equilibrate a
bulk composition before returning a different liquid. The existing fixture
supplies runtime hashes, component order, and basis metadata. Its three
numerical states are used only for validation, never interpolation.

Run from a source checkout
--------------------------

Prepare the separately installed, pinned runtime and Python environment using
the instructions in :doc:`melts_silicate_reference`. No MELTS dependency is
added to ExoEOS imports or ordinary tests. The example performs no download,
installation, or backend redistribution. A host Python process needs NumPy;
the external worker additionally needs ``tinynumpy==1.2.1``.

The callable interface is:

.. code-block:: python

   from examples.melts_liquid_evaluator import evaluate_liquid

   result = evaluate_liquid(
       temperature=1473.15,       # K
       pressure=5e7,              # Pa
       component_moles=amounts,   # 19 endmember amounts, mol, in component_order
       runtime="/path/to/alphamelts-py-2.3.2-ubuntu_22_04-x86_64",
       python_executable="/tmp/exoeos-melts-env/bin/python",
   )

``component_moles`` must be a length-19 vector of finite nonnegative amounts,
with a positive total. Its order is the ``component_order`` in the existing
fixture and result, also exported as ``COMPONENTS`` by the example. Zero
components remain exactly absent; unavailable potentials and activities are
JSON ``null`` (Python ``None``). No trace floor is inserted. Positive finite
T/P are required; exactly 273.15 K is excluded because the pinned wrapper
rejects zero Celsius. These input checks do not define a calibration domain.
Unsupported compositions or failed backend properties raise an exception.

For the CLI, ``request.json`` contains ``T_K``, ``P_Pa``, and
``component_moles``; ``common_R_J_mol_K`` is optional:

.. code-block:: console

   python examples/melts_liquid_evaluator.py \
     --runtime /path/to/alphamelts-py-2.3.2-ubuntu_22_04-x86_64 \
     --python /tmp/exoeos-melts-env/bin/python \
     --input request.json --output result.json

Each call launches a fresh process with its own temporary working directory
and the bundled native-library search path. Runtime file hashes are checked
before importing the external wrapper. This isolates its shared native state
and generated files. The worker converts K to Celsius and Pa to bar, explicitly
sets ``Log fO2 Path=None``, and calls
``calcPhaseProperties("liquid", oxide_grams)`` followed by
``calcEndMemberProperties("liquid", oxide_grams)``. It checks both returned
oxide compositions, the absolute phase mass, reconstructed endmember amounts,
and mole fractions against the request. Thus an unnoticed iron oxidation
adjustment cannot be accepted as a closed-oxygen property evaluation.

Returned thermodynamics and basis
---------------------------------

The result includes full ``mu_J_mol``, pure-liquid endmember ``mu0_J_mol`` at
the supplied T/P, total ``gibbs_J``, mass in grams, requested and returned
component amounts, oxide masses, and endmember mole fractions. It also records
the component-to-oxide and component-to-element formula matrices, element
order and finite element amounts. Mg and all original background elements
are retained. The formal halogen oxide labels include negative oxygen
stoichiometry, as in the backend basis.

The backend uses :math:`R_b=8.3143\ {\rm J\,mol^{-1}\,K^{-1}}`.
``activity``, ``ln_activity``, and ``ln_gamma`` retain that convention:

.. math::

   \ln a_b=(\mu-\mu^0)/(R_bT),\qquad
   \ln\gamma_b=\ln a_b-\ln x.

``mu_RT`` and ``mu0_RT`` instead use the explicitly recorded
``common_R_J_mol_K`` (default 8.31446261815324). For consumers that retain a
common-R ideal term, ``ln_activity_common_R`` and ``ln_gamma_common_R`` give

.. math::

   \ln a_c=(\mu-\mu^0)/(R_cT),\qquad
   \ln\gamma_c=\ln a_c-\ln x.

Consequently, :math:`\ln\gamma_c` is not merely a constant multiple of
:math:`\ln\gamma_b`. Full potentials already contain ideal and excess mixing;
adding either again would double count it. The Gibbs energy comes from the
separate phase property method, with an Euler identity check against the
component potentials.

``oxide_mu_J_mol`` is available only when present components and oxides form
a square full-rank subspace. Otherwise it contains null values and an explicit
status. The VapoRock Appendix B basis transformation documented in
:doc:`melts_silicate_reference` preserves full Gibbs energies and balanced
reaction energies. It does not establish pure-oxide standards or allow a
permutation of the GCE silicate activity coefficients. No oxide activities
or standard potentials are supplied.

Provenance includes pinned backend hashes and source information, actual
backend version, runtime/interpreter paths, Python/NumPy/tinynumpy versions,
float dtype, worker/caller commands, logs, evaluator and reference hashes,
ExoEOS commit, and hashes of changed tracked files. The evaluator's own hash
also identifies its content before it becomes tracked. The release binary
is pinned; its thermodynamic C build commit remains independently unverified.

Acceptance and limits
---------------------

Run the optional numerical acceptance with the real runtime:

.. code-block:: console

   python examples/melts_liquid_evaluator.py \
     --runtime /path/to/alphamelts-py-2.3.2-ubuntu_22_04-x86_64 \
     --python /tmp/exoeos-melts-env/bin/python \
     --validate --output /tmp/melts-liquid-validation.json

The command performs 120 fresh evaluations. It re-evaluates all three saved
liquids, tests a factor-2.5 amount scaling of each (extensive G and invariant
mu), and checks central differences of G against every present endmember mu.
For the first saved liquid it also changes Fe, Si, and O independently in
both directions while preserving all other element budgets. Each perturbation
is 2% of the maximum admissible step in its direction and is implemented
through SiO2, Fe2O3, and Fe2SiO4 amounts. Those three independent endmember
amount derivatives are then checked at each new composition.

The saved-reference, scaling, and composition relative tolerance is 5e-9.
Finite differences use relative amount steps of 1e-3 and absolute tolerance
0.02 J/mol. A pinned-runtime run reproduced all states and had maximum
:math:`|\partial G/\partial n_i-\mu_i|=0.00579` J/mol.
Offline tests cover conversion, input rejection, absent components, both
R conventions, process isolation, and rejection of changed compositions or
unavailable present-component properties. They do not replace the backend
acceptance command.

These checks establish local equation consistency at the saved 1423.15 /
1473.15 K, 50 / 500 MPa states and the declared nearby compositions. They
do not establish experimental uncertainty, a calibrated interpolation box,
a differentiable JAX model, or a stable liquid assemblage at arbitrary inputs.
``phase_policy`` explicitly marks equilibrium and stability as unchecked.
Competing solids, common alloy/gas reaction standards, a reduced dissolved-H2
construction, and acceptance of a final coupled equilibrium against a fresh
backend evaluation remain separate ExoGibbs integration work.
