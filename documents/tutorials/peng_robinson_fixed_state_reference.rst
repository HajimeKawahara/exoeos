.. This file is generated from the sibling .ipynb by convert_notebooks.py.
.. Do not edit this RST file directly.

:download:`Download the executable notebook <peng_robinson_fixed_state_reference.ipynb>`

A2. Peng–Robinson fixed-state reference against teqp
====================================================

This demonstration validates the ExoEOS Peng–Robinson (PR)
implementation against the independent `teqp v0.23.2
implementation <https://github.com/usnistgov/teqp/tree/5f62a6f515d517e39c3fb035c11a03524ffa3ad6>`__.
It reuses the frozen C–H–O composition from demonstration A1 and fixes

.. math::


   T=300\ \mathrm{K},\qquad P=40\ \mathrm{bar},\qquad
   (x_{\mathrm{CO}},x_{\mathrm{H_2O}},x_{\mathrm{CO_2}},x_{\mathrm{H_2}})
   =(0.4,0.4,0.1,0.1).

The compared observables are molar density :math:`\rho`, reconstructed
pressure :math:`P`, compressibility factor :math:`Z`, component log
fugacity coefficients :math:`\ln\phi_i`, and reduced residual Gibbs
energy :math:`g^r/(RT)`. teqp is used only as an external numerical
reference; install it with
``python -m pip install -e ".[docs,reference]"`` before executing this
notebook.

Matched model contract
----------------------

Every convention that can change a PR result is fixed before evaluation.
Both implementations receive the same parameter arrays; no teqp fluid
database is used.

+-----------------------------------+------------------------------------------------------------+
| Item                              | Setting                                                    |
+===================================+============================================================+
| :math:`T_c`, :math:`P_c`,         | Same arrays printed below, in K, Pa, and dimensionless     |
| :math:`\omega`                    | units                                                      |
+-----------------------------------+------------------------------------------------------------+
| PR version                        | PR76                                                       |
+-----------------------------------+------------------------------------------------------------+
| Alpha function                    | :math:`\alpha_i=[1+\kappa_i(1-\sqrt{T/T_{c,i}})]^2`,       |
|                                   | :math:`\kappa_i=0.37464+1.54226\omega_i-0.26992\omega_i^2` |
+-----------------------------------+------------------------------------------------------------+
| PR constants                      | :math:`\Omega_a=0.45723552892138218938`,                   |
|                                   | :math:`\Omega_b=0.077796073903888455972`                   |
+-----------------------------------+------------------------------------------------------------+
| :math:`k_{ij}`                    | Explicit :math:`4\times4` zero matrix                      |
+-----------------------------------+------------------------------------------------------------+
| Mixing rule                       | :math:`a=\sum_i\sum_jx_ix_j(1-k_{ij})\sqrt{a_i a_j}` and   |
|                                   | :math:`b=\sum_i x_i b_i`                                   |
+-----------------------------------+------------------------------------------------------------+
| Root                              | Largest physical real root, :math:`Z>B` (vapor root)       |
+-----------------------------------+------------------------------------------------------------+
| Gas constant                      | :math:`R=8.31446261815324\ \mathrm{J\,mol^{-1}\,K^{-1}}`   |
+-----------------------------------+------------------------------------------------------------+

teqp’s canonical PR helper switches to the PR78 high-:math:`\omega`
branch. To make the choice unambiguous, the reference instead uses
teqp’s generalized cubic model with the PR76 alpha represented exactly
as a Mathias–Copeman function with coefficients :math:`(\kappa_i,0,0)`.
All four species here have :math:`\omega<0.491`, where the PR76 and PR78
correlations coincide; a separate high-:math:`\omega` unit test protects
the PR76-only branch choice.

.. code:: ipython3

    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    from jax import config

    config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import numpy as np
    import teqp

    from exoeos import (
        PengRobinsonEOS,
        get_critical_properties,
        state_tp,
        state_trho,
    )
    from exoeos.constants import MOLAR_GAS_CONSTANT

    TEQP_VERSION = "0.23.2"
    PR_OMEGA_A = 0.45723552892138218938
    PR_OMEGA_B = 0.077796073903888455972

    species = ("CO", "H2O", "CO2", "H2")
    composition = np.asarray([0.4, 0.4, 0.1, 0.1])
    temperature = 300.0  # K
    pressure = 40.0e5  # Pa

    records = tuple(get_critical_properties(name) for name in species)
    critical_temperatures = np.asarray(
        [record.critical_temperature for record in records]
    )
    critical_pressures = np.asarray(
        [record.critical_pressure for record in records]
    )
    acentric_factors = np.asarray([record.acentric_factor for record in records])
    binary_interaction_parameters = np.zeros((len(species), len(species)))
    gas_constant = float(MOLAR_GAS_CONSTANT)

    pr76_kappa = (
        0.37464 + 1.54226 * acentric_factors - 0.26992 * acentric_factors**2
    )
    teqp_alpha = [
        {"type": "Mathias-Copeman", "c": [float(value), 0.0, 0.0]}
        for value in pr76_kappa
    ]
    teqp_specification = {
        "kind": "cubic",
        "model": {
            "type": "PR",
            "Tcrit / K": critical_temperatures.tolist(),
            "pcrit / Pa": critical_pressures.tolist(),
            "acentric": acentric_factors.tolist(),
            "alpha": teqp_alpha,
            "kmat": binary_interaction_parameters.tolist(),
            "R / J/mol/K": gas_constant,
        },
    }

    # teqp 0.23.2 implements these fields but its JSON schema omits the explicit R.
    teqp_model = teqp.make_model(teqp_specification, validate=False)
    exoeos_model = PengRobinsonEOS(
        jnp.asarray(critical_temperatures),
        jnp.asarray(critical_pressures),
        jnp.asarray(acentric_factors),
        jnp.asarray(binary_interaction_parameters),
    )

    assert teqp.__version__ == TEQP_VERSION
    metadata = teqp_model.get_meta()
    np.testing.assert_allclose(teqp_model.get_R(composition), gas_constant, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(teqp_model.get_kmat(), binary_interaction_parameters, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(metadata["OmegaA"], PR_OMEGA_A, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(metadata["OmegaB"], PR_OMEGA_B, rtol=0.0, atol=0.0)
    assert metadata["kind"] == "Peng-Robinson"
    assert metadata["alpha"] == teqp_alpha

    print(f"teqp version: {teqp.__version__}")
    print(f"R = {teqp_model.get_R(composition):.14f} J mol^-1 K^-1")
    print(f"{'species':>7} {'x':>7} {'Tc [K]':>16} {'Pc [Pa]':>20} {'omega':>14}")
    for name, fraction, record in zip(species, composition, records):
        print(
            f"{name:>7} {fraction:7.3f} "
            f"{record.critical_temperature:16.10f} "
            f"{record.critical_pressure:20.10f} "
            f"{record.acentric_factor:14.10f}"
        )


.. parsed-literal::

    teqp version: 0.23.2
    R = 8.31446261815324 J mol^-1 K^-1
    species       x           Tc [K]              Pc [Pa]          omega
         CO   0.400   132.8598946339   3498194.6661988837   0.0497000000
        H2O   0.400   647.0960000000  22063999.9999977536   0.3442920843
        CO2   0.100   304.1282000030   7377298.3734467523   0.2239400000
         H2   0.100    33.1443326883   1296357.6060553084  -0.2190000000


Independent vapor-root construction
-----------------------------------

The fixed state is deliberately in a three-root region, so selecting the
vapor root is observable rather than a no-op. The teqp reference density
is obtained without calling any ExoEOS root routine. teqp supplies its
mixed :math:`a` and :math:`b` parameters, from which the standard PR
cubic is formed:

.. math::


   Z^3+(B-1)Z^2+(A-3B^2-2B)Z+(B^3+B^2-AB)=0,

where :math:`A=aP/(RT)^2` and :math:`B=bP/(RT)`. The largest real root
satisfying :math:`Z>B` is selected. teqp then independently reconstructs
pressure and evaluates :math:`Z` and the fugacity coefficients at that
density.

.. code:: ipython3

    def evaluate_teqp_vapor(model, T, P, x):
        R = model.get_R(x)
        attraction = model.get_a(T, x)
        covolume = model.get_b(T, x)
        A = attraction * P / (R * T) ** 2
        B = covolume * P / (R * T)
        roots = np.roots(
            [1.0, B - 1.0, A - 3.0 * B**2 - 2.0 * B, B**3 + B**2 - A * B]
        )
        real_roots = roots.real[np.abs(roots.imag) < 1.0e-12]
        physical_roots = real_roots[real_roots > B]
        if physical_roots.size != 3:
            raise RuntimeError("The fixed state must have three physical PR roots.")

        Z_root = np.max(physical_roots)
        rho = P / (Z_root * R * T)
        partial_densities = rho * x
        Ar00 = model.get_Ar00(T, rho, x)
        Ar01 = model.get_Ar01(T, rho, x)
        Z = 1.0 + Ar01
        calculated_pressure = rho * R * T + model.get_pr(T, partial_densities)
        lnphi = np.log(model.get_fugacity_coefficients(T, partial_densities))
        gres_RT = x @ lnphi

        np.testing.assert_allclose(Z, Z_root, rtol=2.0e-13, atol=2.0e-14)
        np.testing.assert_allclose(
            calculated_pressure / (rho * R * T), Z, rtol=2.0e-13, atol=2.0e-14
        )
        np.testing.assert_allclose(
            gres_RT, Ar00 + Ar01 - np.log(Z), rtol=2.0e-13, atol=2.0e-14
        )
        return {
            "rho": rho,
            "P": calculated_pressure,
            "Z": Z,
            "lnphi": lnphi,
            "gres_RT": gres_RT,
            "roots": roots,
        }


    teqp_reference = evaluate_teqp_vapor(
        teqp_model, temperature, pressure, composition
    )
    exoeos_state = state_tp(
        exoeos_model,
        temperature,
        pressure,
        jnp.asarray(composition),
        phase="vapor",
    )
    exoeos_at_reference_density = state_trho(
        exoeos_model,
        temperature,
        teqp_reference["rho"],
        jnp.asarray(composition),
    )

    print("PR roots from teqp parameters:", teqp_reference["roots"])
    print(f"Selected vapor density: {teqp_reference['rho']:.12f} mol m^-3")


.. parsed-literal::

    PR roots from teqp parameters: [0.77349736 0.11650519 0.07514243]
    Selected vapor density: 2073.221567707672 mol m^-3


Fixed-state comparison
----------------------

The table reports ExoEOS minus teqp. A relative tolerance of
:math:`2\times10^{-10}` is tight enough to expose a change in PR
constants, alpha convention, mixing rule, gas constant, or root while
allowing the two libraries to use different derivative and root
implementations. Pressure is also evaluated directly by ExoEOS at the
teqp reference density, so its agreement is not merely a consequence of
using pressure as a ``state_tp`` input.

.. code:: ipython3

    def print_comparison(name, reference, candidate):
        absolute_error = candidate - reference
        relative_error = absolute_error / abs(reference)
        print(
            f"{name:<20} {reference: .12e} {candidate: .12e} "
            f"{absolute_error: .3e} {relative_error: .3e}"
        )


    print(f"{'observable':<20} {'teqp':>19} {'ExoEOS':>19} {'abs error':>12} {'rel error':>12}")
    print_comparison("rho [mol m^-3]", teqp_reference["rho"], float(exoeos_state.rho))
    print_comparison("P [Pa]", teqp_reference["P"], float(exoeos_state.P))
    print_comparison("Z", teqp_reference["Z"], float(exoeos_state.Z))
    for index, name in enumerate(species):
        print_comparison(
            f"lnphi({name})",
            teqp_reference["lnphi"][index],
            float(exoeos_state.lnphi[index]),
        )
    print_comparison(
        "g^r/(RT)", teqp_reference["gres_RT"], float(exoeos_state.gres_RT)
    )

    rtol = 2.0e-10
    atol = 2.0e-12
    np.testing.assert_allclose(teqp_reference["P"], pressure, rtol=rtol, atol=1.0e-5)
    np.testing.assert_allclose(float(exoeos_state.rho), teqp_reference["rho"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(float(exoeos_state.P), teqp_reference["P"], rtol=rtol, atol=1.0e-5)
    np.testing.assert_allclose(float(exoeos_state.Z), teqp_reference["Z"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(np.asarray(exoeos_state.lnphi), teqp_reference["lnphi"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(float(exoeos_state.gres_RT), teqp_reference["gres_RT"], rtol=rtol, atol=atol)

    np.testing.assert_allclose(float(exoeos_at_reference_density.P), teqp_reference["P"], rtol=rtol, atol=1.0e-5)
    np.testing.assert_allclose(float(exoeos_at_reference_density.Z), teqp_reference["Z"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(np.asarray(exoeos_at_reference_density.lnphi), teqp_reference["lnphi"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(float(exoeos_at_reference_density.gres_RT), teqp_reference["gres_RT"], rtol=rtol, atol=atol)

    print("All fixed-state PR observables agree with teqp.")


.. parsed-literal::

    observable                          teqp              ExoEOS    abs error    rel error
    rho [mol m^-3]        2.073221567708e+03  2.073221567708e+03  2.728e-12  1.316e-15
    P [Pa]                4.000000000000e+06  4.000000000000e+06  4.657e-09  1.164e-15
    Z                     7.734973557808e-01  7.734973557808e-01 -2.220e-16 -2.871e-16
    lnphi(CO)             7.492890545020e-02  7.492890545020e-02  1.388e-17  1.852e-16
    lnphi(H2O)           -5.899275983086e-01 -5.899275983086e-01  2.220e-16  3.764e-16
    lnphi(CO2)           -2.279263634316e-01 -2.279263634316e-01  3.886e-16  1.705e-15
    lnphi(H2)             1.991696422115e-01  1.991696422115e-01  3.331e-16  1.672e-15
    g^r/(RT)             -2.088751492654e-01 -2.088751492654e-01  1.110e-16  5.315e-16
    All fixed-state PR observables agree with teqp.


Interpretation and limitations
------------------------------

Agreement at this non-ideal three-root state validates the implemented
PR76 Helmholtz expression, composition derivatives, pressure inversion,
and vapor-root selection against a separate EOS code path. It is an
implementation reference, not evidence that classical PR with
:math:`k_{ij}=0` accurately describes this water-containing mixture. The
composition remains frozen, and no phase or chemical equilibrium is
calculated.

The reference is intentionally narrow: one state makes convention drift
easy to diagnose and inexpensive to retain as a unit-test oracle.
Broader physical validation should use experimental data and
independently sourced interaction parameters.
