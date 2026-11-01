"""Material chemical potentials from a supplied extensive Gibbs scalar."""

from typing import Callable

import jax
from jax.typing import ArrayLike

from exoeos._arrays import common_dtype, scalar_array, vector_array
from exoeos.state import GibbsResponse


def gibbs_response(
    gibbs_j: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    temperature_k: ArrayLike,
    pressure_pa: ArrayLike,
    amounts_mol: ArrayLike,
) -> GibbsResponse:
    """Evaluate extensive Gibbs energy and its component chemical potentials.

    ``gibbs_j(T, P, n)`` must return a scalar energy in J, with temperature
    in K, absolute pressure in Pa and a nonempty component-amount vector in
    mol. It must be differentiable in the selected component coordinates at
    the supplied state. The returned potentials are ``dG/dn`` in J/mol of
    those coordinates, with T and P held fixed.

    Amounts are neither normalized nor floored. The callback defines the
    component basis, standards and material domain; e.g. a mole of an MQM
    pair coordinate need not carry one mole of atoms. Only static argument
    and scalar-output shapes are checked. This evaluator does not certify
    domain validity, an absent phase's potentials, or phase equilibrium.

    Apply ``jax.jvp`` or ``jax.vjp`` to the evaluator for material responses,
    including T/P changes and Hessian-vector products. Such responses need
    a twice differentiable callback. Internal variables remain independent
    coordinates unless the callback itself includes their differentiable
    relaxation. Close over material parameters (and their T/P dependence)
    in the callback; they remain differentiable in an outer JAX transform.
    Bind the callback before using ``jax.jit``, or mark it static.
    """
    if not callable(gibbs_j):
        raise TypeError("gibbs_j must be callable.")
    temperature = scalar_array(temperature_k, "temperature_k")
    pressure = scalar_array(pressure_pa, "pressure_pa")
    amounts = vector_array(amounts_mol, "amounts_mol")
    dtype = common_dtype((), temperature, pressure, amounts)
    temperature, pressure, amounts = (
        value.astype(dtype) for value in (temperature, pressure, amounts)
    )
    energy, potentials = jax.value_and_grad(gibbs_j, argnums=2)(
        temperature, pressure, amounts
    )
    return GibbsResponse(energy, potentials)
