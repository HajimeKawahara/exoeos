# ExoEOS

Differentiable equations of state for planetary atmospheres and fluids, built
with JAX.

The first implementation is a calorically perfect ideal-gas mixture. It
provides one small state interface that is intended to remain common to future
real-gas models.

## Installation

```bash
python -m pip install -e ".[test]"
```

## Quick start

All public quantities use SI units. Component molar masses are in `kg mol-1`,
component molar heat capacities are in `J mol-1 K-1`, temperature is in K, and
pressure is in Pa.

```python
import jax
import jax.numpy as jnp

from exoeos import IdealGas


eos = IdealGas(
    molar_masses=jnp.array([2.01588e-3, 4.002602e-3]),
    molar_heat_capacities=jnp.array([28.84, 20.786]),
)

x = jnp.array([0.85, 0.15])
state = eos.state(T=1000.0, P=1.0e5, x=x)

state.Z
state.mass_density
state.number_density
state.log_fugacity_coefficients
state.residual_gibbs
state.residual_enthalpy
state.cp
state.cv
state.thermal_expansion
state.adiabatic_gradient

jitted_density = jax.jit(
    lambda temperature: eos.state(temperature, 1.0e5, x).mass_density
)
density_gradient = jax.grad(jitted_density)(1000.0)
```

`x` contains normalized mole fractions on its last axis. ExoEOS does not
renormalize composition. Component molar masses and heat capacities are
required because `T`, `P`, and `x` alone do not determine mass density or
caloric properties. Reference enthalpies and entropies default to zero; pass
physical reference data when absolute values are needed.

The complete units, shape, reference-state, and transformation contract is in
[the thermodynamic-state contract](documents/thermodynamic_state_contract.md).

## Development

```bash
pytest tests/unittests
```

The current scope is the ideal gas. Cubic equations of state, including
Peng--Robinson, are reserved for later implementations behind the same state
contract.
