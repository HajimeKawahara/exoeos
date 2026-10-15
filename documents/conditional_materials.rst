Conditional native M2 materials
===============================

``exoeos.magma_volatiles`` and ``exoeos.associated_metal`` package the
constitutive expressions previously used by the finite-source examples.
They supply full extensive Gibbs energies in J for K/Pa/mol inputs and
support JAX differentiation on their positive composition interiors.
They do not solve equilibrium or establish empirical wet-BSE calibration.
Packaged coefficient records retain their original source hashes and citations.

Volatile-bearing liquid
-----------------------

``magma_volatiles.COMPONENTS`` is ``magma.DRY11_COMPONENTS`` followed by
H2O, molecular H2 and He. ``gibbs_j`` combines the native dry eleven-component
G with the integrated Thompson water law, Hirschmann H2 dilution, and the
selected Guillot--Sator dry-host trace-He scalar. This replaces the native
hydrous-water contribution. The caller supplies actual H2O/H2/He gas standards
on a common R*T and one-bar basis, plus the declared dry-component molar
masses for He. Molecular H2 and water do not count toward He dry-host mass.

Water requires a positive dry host and cannot exceed its oxygen amount.
``helium_capacity`` interpolates log capacity in reciprocal temperature within
the selected simulated host's tabulated range; unavailable temperatures return
NaN. The native SiO2 standard retains its 1480 K branch. Composition boundaries,
table knots and material branch switches are not smooth derivative interiors.
``water_parameters`` exports the exact scalar coefficients for independent
interval proofs. Capacity and water continuations remain conditional models.

Associated alloy
----------------

``associated_metal.COMPONENTS`` declares twenty chemical species with separate
atom formulas, including K and Na. ``gibbs_j`` adds actual supplied standards,
ideal species mixing and the extensive Ma/P/Jung perspectives. Na enters the
excess-energy Fe coordinate while retaining a separate physical Na ledger.
``interaction_parameters`` exposes the selected temperature continuation;
the optional H--O coefficient is explicit.

``standard_potentials_rt`` requires actual Fe/Si/O/H host standards, eight
named gas standards, an explicit K offset and explicit Na reference fit.
The host standards include the Ma convention shift before paper-specific
free-O/H offsets. The caller selects the numerical composition domain and
records those offsets. JANAF/continued standards are limited to 2100--2400 K.
The declared finite-standard sodium fit is selected with
``sodium_reference_parameters``; its projection and temperature policy have no
implicit default.

Validation and ownership
------------------------

``tests/unittests/conditional_native_test.py`` compares extensive energies,
chemical potentials and curvature against the independently maintained example
expressions, including the complete dry host and volatile contribution.
Equilibrium, phase appearance and global certificates belong to ExoGibbs;
finite budgets, atmospheric columns and pressure roots belong to ExoInventory.
