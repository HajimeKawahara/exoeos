Reduced volatile-melt insertion bounds
======================================

``exoeos.magma_water_interval.reduced_two_power_melt_bound`` bounds one
specific mathematical scalar on the complete eleven-component dry simplex.
It is an eager verification tool, separate from the material evaluation and
equilibrium solver. It neither selects phases nor changes a water law.

The caller supplies enclosed coefficients of

.. math::

   q(x)=c\cdot x+r\sum_{j=1}^{11}x_j\log x_j
       +\tfrac12 x^{\mathsf T}Wx-(S\cdot x)I(t(x)),
   \qquad x_j\geq0,\quad\sum_jx_j=1,

.. math::

   t_i(x)=\exp\left(b_i+\frac{v_i\cdot x}{d\cdot x}\right),
   \qquad
   I(t)=\int_0^1
      \frac{t_1u^{a_1}+t_2u^{a_2}}
           {1-t_1u^{a_1}-t_2u^{a_2}}\,\frac{du}{u}.

Here ``q`` is the insertion cost divided by ``R*T`` and dry-component
amount. ``r`` is positive, ``W`` is symmetric with zero diagonal, ``S`` and
``d`` are positive, and ``a`` lies in ``[.25, 1]``. The function checks
``sum(t) < .5`` over the full simplex. Vanishing dry fractions are included
using the continuous extension ``x*log(x) = 0`` at zero.

Conditional volatile elimination
--------------------------------

For a caller using the volatile dry11 material, the coefficients can be
obtained by minimizing guest amounts at fixed dry composition and fixed
elemental supporting plane. This connection must be established by the
caller; the API does not inspect an arbitrary material callback.

For one dry-component mole, let ``h``, ``k`` and ``e`` be the water, hydrogen
and helium amounts, respectively. With hydrogen insertion coefficient
``b_H > 0``, ideal mixing between hydrogen and its ``1+h`` host gives

.. math::

   \inf_{k\geq0}q_{\mathrm H_2}(h,k)
      =(1+h)c_{\mathrm H},\qquad
   c_{\mathrm H}=\log(1-e^{-b_{\mathrm H}}).

An independent Henry-law helium term gives
``inf(q_He) = -M_He(x)*K_He*exp(-b_He)``. The water chemical potential is
``log(p)`` in the driving-pressure convention of the solubility law, so
its insertion coefficient ``b_w`` gives

.. math::

   p_*=e^{-b_w},\qquad
   h_*=(S\cdot x)\frac{w(p_*)}{1-w(p_*)},\qquad
   \inf_{h\geq0}q_w=-(S\cdot x)I(t(x)).

The hydrogen offset is included in ``b_w``. Both capacities retain their
dry-composition dependence through the linear-fractional predictors.
These eliminations require their optima to lie within the declared water
and oxygen domains when equality with the original constrained minimum is
claimed. Standards, pressure conversion, gauge, elemental projection and
the resulting coefficient enclosures are caller responsibilities.

How global coverage is established
----------------------------------

The reference composition identifies a small box; it is not assumed to be
an exact stationary point. The first reference coordinate is reconstructed
as the exact real value ``1 - sum(reference[1:])``. In the ten independent
simplex coordinates, interval LDL factorization must verify

.. math::

   \nabla^2q\succeq\kappa I,
   \qquad
   q(x)\geq q(x_0)-\frac{\|\nabla q(x_0)\|^2}{2\kappa}.

The water Hessian is bounded on that same local box. In particular, with
``l_i = log(t_i)`` and ``w = t_1+t_2``, positive integrands imply

.. math::

   I_{l_i}\leq\frac{t_i}{a_i(1-w)^2},\qquad
   I_{l_il_j}\leq\delta_{ij}I_{l_i}
       +\frac{2t_it_j}{(a_i+a_j)(1-w)^3}.

The chain rule supplies a norm bound for the Hessian of ``(S.x)*I``.
This is combined with the ideal-mixture Hessian and the unchanged
quadratic interaction matrix. A failed curvature certificate raises
``ValueError``; it does not establish an instability.

Outside the local box, recursive simplex-box subdivision combines
McCormick bounds for quadratic terms, a convex entropy dual, interval
linear-program dual bounds for the composition ratios, and positive-series
bounds for ``I``. A box is excluded only when its outward lower bound is
nonnegative. Approximate cut and dual candidates are allowed because every
accepted inequality is evaluated with outward arithmetic.

``complete=True`` requires zero unresolved boxes. The returned global lower
bound is the minimum of the local bound and the accepted exterior bounds.
Reaching ``max_boxes`` returns ``complete=False`` and no global bound.
Finite sampling supplies none of this coverage.

API example
-----------

The following analytic control has no composition-dependent water capacity.
It demonstrates the input shapes and result handling without requiring an
equilibrium solution or a second package.

.. code-block:: python

   import numpy as np
   from exoeos.magma_water_interval import reduced_two_power_melt_bound

   def exact(value):
       value = np.asarray(value, dtype=float)
       return np.stack([value, value], axis=-1)

   result = reduced_two_power_melt_bound(
       linear_rt=exact(np.full(11, np.log(11) + 1)),
       quadratic_rt=exact(np.zeros((11, 11))),
       entropy_coefficient=exact(1),
       water_equivalent=exact(np.ones(11)),
       water_log_terms=exact([-10, -10]),
       water_log_predictors=exact(np.zeros((2, 11))),
       predictor_counts=np.ones(11),
       water_exponents=[.5, .8],
       reference_composition=np.full(11, 1 / 11),
       local_margin=.01,
   )
   assert result["complete"]
   assert result["lower_rt_per_dry_mole"] > 0

The trailing pair on coefficient arrays contains lower and upper interval
endpoints. ``predictor_counts`` and ``water_exponents`` instead denote exact
supplied binary64 constants. The result includes the local lower bound,
gradient norm, curvature certificate, exterior counts and unresolved count.
The optional ``progress`` callback receives search counters; it does not
change the bounds.

Scope and mathematical contract
-------------------------------

The certificate covers the continuous reduced scalar with the supplied
coefficient enclosures. The caller must separately bound finite pressure
inversion and water quadrature before using it for the declared numerical
water rule; see :doc:`water_interval`. Backend elementary-function rounding,
coefficient-generation rounding outside the supplied enclosures, empirical
accuracy and a complete planetary equilibrium are not certified here.

The canonical ``mceg/contracts/magma_water_interval.yaml`` records the
equations, binding and these separate scopes. The global mathematical
bound is distinct from a complete-backend or empirical Claim. Execution
receipts and large search outputs remain local. See :doc:`mceg/index` for
the generated mathematical records.
