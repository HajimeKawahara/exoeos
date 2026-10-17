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

Alternative water contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``gibbs_j(..., water_model=...)`` selects one water contribution; its default
``"thompson2025"`` preserves the original scalar. H2, trace He, dry standards,
and the caller's offsets are unchanged. Model names are static JAX arguments.

* ``"power_law"`` requires a positive ``water_capacity_1bar`` (C) and
  ``water_pressure_exponent`` (alpha). The binary dry-host/water relation is
  ``w = C * (f_H2O / bar)**alpha``, with
  ``w = mass_H2O / (mass_dry + mass_H2O)``. The caller selects the empirical
  relation and its domain; ExoEOS supplies its integrable extensive scalar.
  ExoGibbs' ``solubility.h2o_lichtenberg2021`` can supply C at 1e5 Pa and
  alpha from its logarithmic derivative. ExoEOS does not duplicate that fit.
* ``"melts_mode1"`` uses ``magma.liquid_gibbs`` on the hydrous twelve-component
  basis, including its original water standard and interactions.
* ``"melts_mode4"`` uses ``magma_carbonate.hydrous_liquid_gibbs``. This new
  twelve-component restriction fixes carbon and carbonate at zero; positive
  hydrous amounts support first and second derivatives tangent to that face.
  It matches the thirteen-component energy at zero carbon, without claiming
  a derivative in the excluded carbon direction. Mode 4 changes the water
  standard and water interactions, while preserving all 55 dry pairs.

For the power law, let S be dry mass divided by water molar mass and h be
water mol. The additional dimensionless extensive energy is

.. math::

   G_w/(RT) = h(\mu^0_{w,g}/RT - \log C/\alpha + \delta_w)
       + [S\log(S/(S+h)) + h\log(h/(S+h))]/\alpha.

This includes the host response and gives
``mu_water/RT = mu0_gas/RT + log(w/C)/alpha + delta_water`` before H2 dilution.
With molecular H2 present, its existing host dilution also contributes to
water's chemical potential. Neither H2 nor He enters the water mass fraction.
Native hydrous choices instead add only
``n_water * (R*T*(gas_standard_rt + water_offset_rt) - g0_native_water_gas)``
to their full native G. ``magma_fluid.water_gas_standard_gibbs`` supplies the
ideal one-bar native reference, preserving the native liquid--gas difference.
This alignment is not a solubility fit and does not add a second water term.

The numerical domain is not an empirical validity claim. In particular,
Lichtenberg's high-pressure data compilation does not validate a 2173 K BSE
melt, and neither native MELTS option establishes wet-BSE accuracy there.
Selecting a water contribution cannot remove restrictions on a separate alloy.
``tests/unittests/conditional_water_test.py`` checks fugacity recovery, dry-host
response, extensivity, the common gas gauge, curvature, and the restricted
mode-4 face. The original conditional tests remain the default parity check.

Two-power water isotherm
~~~~~~~~~~~~~~~~~~~~~~~~

``water_model="two_power_law"`` takes length-two arrays in
``water_capacity_1bar`` and ``water_pressure_exponent``. ExoEOS integrates
``w=sum(C_i*p**a_i)`` without owning a particular empirical fit. One concrete
consumer is ExoGibbs' ``h2o_sun_yao2026_terms``. Evaluate its T/P/composition
coefficients inside the differentiated liquid callback; freezing composition
before differentiation would omit Na/Al host chemical potentials.

Let h be water mol, S be dry mass / water molar mass, and z=ln(p), where p
is the dimensionless driving pressure in bar. With w=h/(S+h), the extra
scalar, before the existing H2 and He terms, is

.. math::

   G_w/(RT) = h(\mu^0_{w,g}/RT + \delta_w + z)
      - S\int_{-\infty}^{z} \frac{w(v)}{1-w(v)}\,dv,
   \qquad w(v)=\sum_{i=1}^{2} C_i e^{a_i v}.

The unique monotone inverse uses log-space bisection and implicit JAX
first/second derivatives. A transformed 64-point Gauss rule evaluates the
integral, including derivatives of supplied coefficients. Tests compare it
with independent SciPy integration and the exact single-exponent limit.
The supported numerical domain is C_i>0, 0.25<=a_i<=1, S>0, and 0<=w<=0.5.
The existing dry11 host-oxygen constraint already enforces w<0.5, so this
adds no interior composition cap to that source model. G is zero at h=0;
water chemical potentials and Hessians are only supported in the positive
interior. Scaling all amounts scales G and preserves chemical potentials.

Before H2 dilution, dG_w/dh/(RT) equals the supplied gas standard plus
ln(p) and the optional offset. The unchanged molecular-H2 host mixing
adds ``ln(n_dry_plus_water/(n_dry_plus_water+n_H2))`` to water's chemical
potential. Thus a coupled ideal-gas consumer replays the bare isotherm at
``p_eff=p_H2O*(1+n_H2/n_dry_plus_water)`` when its offset is zero. This is
an explicit constitutive extension, not the unmodified binary-fluid law.
He is excluded from both the water mass basis and the H2 host count.
Partial-pressure empirical laws require an ideal gas or an additional,
explicitly justified nonideal mapping; ExoEOS does not silently equate
partial pressure and fugacity. No fitted Sun--Yao coefficients or provider
imports are added to ExoEOS.

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
