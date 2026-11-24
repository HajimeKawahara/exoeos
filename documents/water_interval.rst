Two-power water quadrature bounds
========================================

``exoeos.water_interval.two_power_water_quadrature_bound`` encloses the
mathematical integration error of the existing two-power water rule. It is
an eager post-evaluation tool; the material scalar and its derivatives are
unchanged. NumPy and the existing outward interval arithmetic suffice.

For nonnegative capacity terms at one common driving pressure, define

.. math::

   t_i=C_i p^{a_i},\qquad w=t_1+t_2\leq w_{\max}\leq \tfrac12.

The continuous integral and the real-arithmetic version of the stored rule are

.. math::

   I=\int_{-\infty}^{\log p}\frac{w(e^z)}{1-w(e^z)}\,dz,
   \qquad
   Q=s\sum_j q_j\frac{t_1u_j^{\beta_1}+t_2u_j^{\beta_2}}
                            {u_j(1-t_1u_j^{\beta_1}-t_2u_j^{\beta_2})}.

Here ``s``, ``beta``, nodes and weights are the provider's stored/evaluated
binary64 coefficients, treated as exact constants of this mathematical
comparison. In particular, the proof does not replace them with ideal exact
Gauss coefficients or assume that the transformed powers are integers.

For each geometric-series order, interval arithmetic encloses every moment:

.. math::

   E_m\geq\max_{0\leq k\leq m}\left|
     s\sum_jq_ju_j^{k\beta_1+(m-k)\beta_2-1}
     -\frac{1}{ka_1+(m-k)a_2}\right|.

The binomial coefficients, weighted by the two capacity terms, sum to
``w**m``. Both series have nonnegative terms. Their tails therefore give

.. math::

   |Q-I|\leq \sum_{m=1}^{M}E_m w_{\max}^m
     +\frac{w_{\max}^{M+1}}{(M+1)a_{\min}(1-w_{\max})}
     +\frac{s w_{\max}^{M+1}}{1-w_{\max}}
        \sum_jq_ju_j^{(M+1)\beta_{\min}-1}.

All operations in the bound are widened outwards. Logarithms use the existing
range-reduced atanh series. Fractional powers use a range-reduced exponential
Taylor series with an explicit remainder, followed by reciprocal and repeated
squaring. No platform-specific bound on ``log``, ``exp`` or ``pow`` is assumed.

The capacity mixture can vary continuously over its complete simplex; the
bound includes either term vanishing as a limit. Multiplying the integral
bound outwards by ``S = dry_mass / water_molar_mass`` gives its contribution
to the extensive Gibbs/RT error. On a phase-component mole basis, an outward
upper bound on ``S/N`` must be supplied by the consumer.

Example
-------

.. code-block:: python

   from exoeos.water_interval import two_power_water_quadrature_bound

   bound = two_power_water_quadrature_bound(
       [.807805, .449387], water_fraction_max=.5, series_orders=64)
   print(bound.integral_error_upper)
   assert 18 * bound.integral_error_upper < 1e-8

The last assertion compares this one contribution with an illustrative
dimensionless budget. It is not a phase-stability result.

Scope of the certificate
------------------------

The bound applies at a **common driving pressure**, with a bound on the
actual ``sum(C_i*p**a_i)``. A finite pressure inversion need not reproduce the
requested ``h/(S+h)`` exactly. Pressure inversion, evaluation of capacities,
backend floating-point operations, derivative errors and minimization over
dry composition each need separate bounds before using this result in a
complete phase proof. Agreement with high-precision point calculations is
used in regression tests, not to establish uniform coverage. No empirical
water-solubility accuracy or equilibrium classification follows.

Mathematical contract
---------------------

The canonical ``mceg/contracts/water_interval.yaml`` records these three
equation definitions, their units and domains, the code binding, and separate
mathematical, complete-scalar and empirical Assessments. The first has a
scoped source/derivation review and executed regression evidence; the latter
two remain unknown. The numerical receipt stays in local
``results/mceg/20261124_water_interval/``. MCEG is optional review tooling.
See :doc:`mceg/index` for the generated mathematical records.
