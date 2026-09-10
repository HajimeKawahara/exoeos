External MELTS silicate reference
========================================

``tests/reference/melts_silicate_v1.json`` records three independently
calculated silicate liquid states using alphaMELTS 2.3.2 with
**rhyolite-MELTS 1.0.2** (calculation mode 1). Its identifier is
``alphamelts_2_3_2_rhyolite_melts_1_0_2_morb_v1``. This is a reference fixture,
not a native ExoEOS silicate model. It adds no mandatory dependency and
does not provide a JAX activity function, fitted surrogate, or coupled
metal-silicate equilibrium calculation.

Provenance and phase conditions
----------------------------------------

The generator uses the official Linux x86_64 Python release [AMrelease]_.
The fixture pins its archive, Python wrappers, C library, and bundled GSL
libraries by SHA-256. The release source revision is
``9b92b1e0e6d1538a0f498572d6c6651d5d3aca7e``; the loaded binary reports
``2.3.2 (Jul 21 2026 17:06:24)``. The thermodynamic C source commit used to
build that binary has not been independently established. The backend is
AGPL-3.0 software and is not redistributed with ExoEOS.

Input amounts come from the pinned official MORB tutorial [AMtutorial]_,
including 0.2 g H2O. Their original total is **100.35 g**, not 100 g; the
fixture keeps these amounts in ``bulk_oxide_mass_g``. Every state includes
the resulting phase masses, phase oxide weight percentages, liquid
composition, chemical potentials, activities, and solver message.

The external calculation uses ``calcEquilibriumState(1, 0)``: fixed T/P,
no fractionation, default backend phase selection, and no explicitly
suppressed phases. ``Log fO2 Path`` is explicitly ``None``. Input oxygen
is closed; the reported ``log10_fO2`` is an output, not an imposed buffer.
``database_phase_names`` records available names rather than asserting that
every phase is enabled by the backend's default policy.

.. list-table:: Reported equilibrium assemblages and liquid mass fractions
   :header-rows: 1
   :widths: 15 13 15 57

   * - T (K)
     - P (Pa)
     - Liquid fraction
     - Phase instances
   * - 1473.15
     - 5e7
     - 0.81915910
     - liquid1, olivine1, plagioclase1
   * - 1423.15
     - 5e7
     - 0.38327437
     - liquid1, olivine1, clinopyroxene1, plagioclase1, spinel1
   * - 1473.15
     - 5e8
     - 0.37513745
     - liquid1, clinopyroxene1, clinopyroxene2, plagioclase1, spinel1

The two clinopyroxene instances in the third state are retained separately.
The corresponding calculated log10 oxygen fugacities are -9.37197054,
-9.61839462, and -9.50568631. These are equation references for a hydrous
basaltic host. They are neither measured activities nor a calibrated
T/P/composition box, and they do not establish coexistence with the
``MaFeSiOLiquid`` model.

Units, components, and standards
----------------------------------------

The fixture uses K and Pa. The generator converts to the alphaMELTS Python
inputs [AMwrapper]_ using ``T_C = T_K - 273.15`` and
``P_bar = P_Pa / 1e5``. This differs from ThermoEngine ``MELTSmodel``'s
Celsius/MPa interface; ThermoEngine was not used for these calculations.
Chemical potentials are J/mol of the named component, phase Gibbs energies
are J, masses are g, and activity quantities are dimensionless.

Liquid mole fractions use the ordered MELTS endmembers, not oxide or
elemental mole fractions. Their order is recorded in ``component_order``:

.. code-block:: text

   SiO2, TiO2, Al2O3, Fe2O3, MgCr2O4, Fe2SiO4, MnSi0.5O2,
   Mg2SiO4, NiSi0.5O2, CoSi0.5O2, CaSiO3, Na2SiO3, KAlSiO4,
   Ca3(PO4)2, CO2, SO3, Cl2O-1, F2O-1, H2O

The JSON preserves the backend's lower-case labels, including its formal
halogen oxide labels. Endmembers absent from these inputs have ``x=0``;
their unavailable chemical potentials, activities, and logarithms are
``null``. No trace amounts or assumed zero potentials are substituted.

The selected ``mu0`` is the pure liquid endmember reference at the same
T/P. Use the wrapper's ``activity`` field, which is reconstructed from
``mu`` and ``mu0``; ``activity0`` can use a fixed structural or ordering
reference [AMwrapper]_. With the backend's own
:math:`R=8.3143\ {\rm J\,mol^{-1}\,K^{-1}}`, present components satisfy

.. math::

   \ln a_i=\frac{\mu_i-\mu_i^\circ}{RT},\qquad
   \ln\gamma_i=\ln a_i-\ln x_i.

Thus :math:`\mu_i-\mu_i^\circ` contains **ideal and excess mixing**; it
is :math:`RT\ln a_i`, not :math:`RT\ln\gamma_i`. The fixture's ``gex_RT``
uses the separately reported liquid Gibbs energy :math:`G_\ell`:

.. math::

   \frac{g^E}{RT}
   =\frac{G_\ell/n_\ell-\sum_{i:x_i>0} x_i\mu_i^\circ}{RT}
    -\sum_{i:x_i>0}x_i\ln x_i
   =\sum_{i:x_i>0}x_i\ln\gamma_i.

This finite-state identity does not supply a differentiable silicate model.
The full ``mu`` already includes mixing. Before combining it with another
phase, a consumer must establish consistent elemental and standard-state
references; adding another ideal or excess term would double count mixing.

Oxide basis conversion
----------------------

Let :math:`\nu_{ij}` describe liquid endmember :math:`i` in oxide basis
:math:`j`, with the columns in ``oxide_order``. For example,
:math:`{\rm Fe_2SiO_4}=2{\rm FeO}+{\rm SiO_2}` and
:math:`{\rm KAlSiO_4}=\tfrac12{\rm K_2O}+\tfrac12{\rm Al_2O_3}+{\rm SiO_2}`.
VapoRock Appendix B [VapoRock]_ gives the chemical-potential transformation:

.. math::

   \boldsymbol n_{\rm ox}=\nu^T\boldsymbol n_\ell,\qquad
   \boldsymbol\mu_\ell=\nu\boldsymbol\mu_{\rm ox}.

The generator obtains oxide amounts from phase mass, oxide weight
percentages, and oxide molecular weights, then solves for component
amounts. It does not use the runtime's liquid molecular-weight helper or
its ``calcEndMemberProperties('bulk')`` oxide-property helper. Potentials
are transformed only on the defined square subspace of present components
and oxides. Missing values remain ``null``.

This changes the chemical basis while preserving Gibbs energy and balanced
reaction free energies. Transforming endmember ``mu0`` by the same matrix
would not establish pure-oxide standard states. Accordingly, the fixture
supplies ``oxide_mu_J_mol`` but no oxide activities or activity coefficients.

Reproduction and offline checks
-------------------------------

Regeneration is optional and requires the pinned Ubuntu 22.04 x86_64
runtime. From the ExoEOS checkout, prepare a separate Python environment,
download the official archive [AMrelease]_, and verify its SHA-256:

.. code-block:: console

   python -m venv /tmp/exoeos-melts-env
   /tmp/exoeos-melts-env/bin/pip install numpy tinynumpy==1.2.1
   curl --fail --location --output /tmp/alphamelts.zip \
     https://github.com/magmasource/alphaMELTS/releases/download/v2.3.2/alphamelts-py-2.3.2-ubuntu_22_04-x86_64.zip
   sha256sum /tmp/alphamelts.zip
   unzip /tmp/alphamelts.zip -d /tmp/exoeos-melts-runtime
   /tmp/exoeos-melts-env/bin/python tests/reference/generate_melts_silicate.py \
     --runtime /tmp/exoeos-melts-runtime/alphamelts-py-2.3.2-ubuntu_22_04-x86_64 \
     --check

The archive hash must be
``97ec2cdb53cae69822a41b5639d8b93edf95b2c361bc15e64e53b192ea9e1425``.
The generator checks all recorded runtime file hashes before loading the
backend. Each state runs in a fresh subprocess and temporary working
directory, isolating library global state and generated files. It loads
the external Python wrapper only inside these workers and sets the bundled
library search path. It performs no download or installation itself.

``--check`` regenerates and compares without changing the fixture; omitting
it writes the fixture. Recorded numerical reproduction tolerances are
``rtol=atol=5e-9``. They concern this pinned computation, not experimental
uncertainty. Ordinary unit tests use the committed JSON offline, checking
phase mass and elemental balance, component-basis reconstruction,
chemical-potential and excess-energy identities, and basis conversion.
No MELTS installation is required for normal ExoEOS imports or tests.

.. [AMrelease] `alphaMELTS v2.3.2 release
   <https://github.com/magmasource/alphaMELTS/releases/tag/v2.3.2>`_.
.. [AMtutorial] `Pinned official MORB tutorial input
   <https://github.com/magmasource/alphaMELTS/blob/9b92b1e0e6d1538a0f498572d6c6651d5d3aca7e/examples/tutorial/py/tutorial.py>`_.
.. [AMwrapper] `Pinned MELTSengine wrapper and property conventions
   <https://github.com/magmasource/alphaMELTS/blob/9b92b1e0e6d1538a0f498572d6c6651d5d3aca7e/bases/alphamelts/py/meltsengine.py>`_.
.. [VapoRock] Wolf et al., *VapoRock*, Appendix B, arXiv:2208.09582v2.
   `<https://arxiv.org/html/2208.09582v2#A2>`_.
