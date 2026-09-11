Fe-Si-O-H alloy control
========================================

``MaFeSiOHLiquid`` adds atomic H to the native :doc:`fe_si_o_reference`
through a scalar extensive excess energy. It supplies a consistent control
with no excess interactions involving H. It is neither a calibrated H-bearing
alloy nor a reproduction of the four-component author-code solvent treatment.
The default record is ``ma2001_fe_si_o_h_no_h_interaction_v1``; the dry model
and its existing reference record remain unchanged.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from exoeos import MaFeSiOHLiquid, solution_state

   model = MaFeSiOHLiquid()
   T, P = 1873.0, 1.0e5  # Numerical inputs in K, Pa; no phase-stability claim.
   x = jnp.array([0.68, 0.08, 0.04, 0.20])  # Fe, Si, O, H atomic fractions
   model.validate_state(T, P, x)
   state = jax.jit(solution_state)(model, T, P, x)
   shift_RT = model.standard_state_shift_RT(T)

The constructor ``MaFeSiOHLiquid(dry_model=MaFeSiOLiquid())`` accepts a dry
model, including custom dry interactions. It is an immutable JAX PyTree with
the dry interaction parameters as its differentiable leaves. Custom values
remain uncalibrated variants. The public methods are ``gex_RT(T, P, x)``,
``validate_state(T, P, x)``, and ``standard_state_shift_RT(T)``. Use the
existing ``solution_state`` / ``total_gex_RT`` kernels and external ``vmap``;
there is no new equilibrium or activity kernel.

Scalar construction and standards
---------------------------------

For dry amounts :math:`n_d=n_{Fe}+n_{Si}+n_O`, total amount
:math:`n=n_d+n_H`, dry fractions :math:`y_i=n_i/n_d`, and four-component
fractions :math:`x_i=n_i/n`, define

.. math::

   \frac{G^E}{RT}=n_d\psi_d(T,P,\mathbf y),\qquad
   \psi_4=(1-x_H)\psi_d(T,P,\mathbf y).

Differentiating this scalar with respect to all four amounts gives

.. math::

   \ln\gamma_i^{(4)}=\ln\gamma_i^{(d)}(\mathbf y)
   \quad(i=Fe,Si,O),\qquad \ln\gamma_H^{(4)}=0.

The complete potential, assembled by the consumer, is

.. math::

   G=\sum_i n_i\mu_i^\circ+RT\sum_i n_i\ln x_i+RTn_d\psi_d.

Ideal activities always use :math:`x`, including for Fe/Si/O. Using
:math:`y` in those terms would omit the reciprocal host dilution. The native
``gex_RT`` and activity coefficients exclude ideal mixing and standard
thermochemistry. Setting every dry interaction to zero recovers an ideal
four-component solution; setting :math:`n_H=0` recovers the dry excess model.

``standard_state_shift_RT`` appends zero to the dry Fe/Si/O conversion. Add
it to source standard potentials divided by :math:`RT`, just as for the dry
model. The zero H **shift** preserves the independently supplied H standard;
it does not set the absolute H standard potential to zero. ExoGibbs owns
the source Okuchi-based H standard reconstruction, reaction standards,
pressure closure, and partition calibration. The example below chooses
synthetic H/H2 standards explicitly and does not supply that reconstruction.

Domain and limits
-----------------

Call ``validate_state`` eagerly before JAX transformations. It checks finite,
positive T/P, four normalized nonnegative real fractions, a positive dry
amount, and the dry model's Fe-positive activity domain and finite parameters.
Normalization uses eight epsilons of the input fraction dtype. It does not
modify inputs. Kernel calls retain shape checks only.

Pure Fe, dry solute-zero edges, and :math:`x_H=0` have finite excess
derivatives. Full ideal chemical potentials of absent components have the
usual logarithmic limits. At fixed valid dry composition, the scalar tends
to zero as :math:`x_H\to1`, while dry activity coefficients retain their
dry-composition values. Those values depend on the approach; pure H is
excluded and has no invented scalar or activity endpoint. Pure dry Si/O
activity endpoints remain excluded.

The intended physical host is Fe-rich liquid. There is no established
finite-H calibration domain or stable-phase evidence in this implementation.
The dry source does not define a calibrated T/P/composition box. Pressure
dependence is absent; accepting positive P is only a mathematical check.
Any use outside independent host/phase evidence is a conditional mechanism
study. No H interaction fit, S/C/N extension, melt model, or partition
prediction is added.

Runnable reference and acceptance
----------------------------------------

.. code-block:: console

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python examples/fe_si_o_h_control.py > /tmp/fe_si_o_h_control.json
   PYTHONPATH=src python -m pytest tests/unittests/ma_fe_si_o_h_test.py tests/unittests/ma_fe_si_o_test.py tests/unittests/api/public_import_contract_test.py

The example selects the local source path explicitly and records its actual
import path, commit, dirty status, source/fixture SHA-256 hashes, parameters,
versions, dtype, command, domain, and phase/component ledger. At a numerical
1873 K and 1 bar, it closes finite H between alloy and an ideal H2/He gas,
retaining finite inert He. Synthetic standard potentials
:math:`\mu^\circ_H/(RT)=\mu^\circ_{H2}/(RT)=0` are prescribed independently;
the exchange is :math:`2H_{alloy}=H2_{gas}`. This is a software consistency
control with element and chemical residual tolerance :math:`10^{-12}`,
not experimental H partition evidence or a coupled melt-metal-gas reference.

Tests check the independent dry reference under H dilution, ideal and dry
limits, extensivity, Euler, Gibbs-Duhem, symmetric amount derivatives,
finite-difference chemical potentials including four-component ideal mixing,
JIT/VMAP, differentiable dry parameters, dtypes, standard conversion,
mathematical boundaries, and finite H/He accounting for three budgets.
Experimental partition residuals and common-standard/phase validation remain
separate acceptance gates.
