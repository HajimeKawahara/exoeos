# ExoEOS

Differentiable equations of state for planetary atmospheres and fluids, built
with JAX.

Residual equation-of-state models share one source of truth: the reduced
residual Helmholtz energy. A calorically perfect ideal-gas mixture remains
available through the original temperature-pressure state interface.

## Installation

```bash
python -m pip install exoeos
```

## Residual Helmholtz API

Models implement

```text
alphar(T, rho, x) = A_res / (n R T),
```

where `rho` is molar density in `mol m-3`. The initial `IdealEOS` is a zero
residual placeholder behind this interface. `A_res` excludes the complete
ideal-gas Helmholtz contribution.

```python
import jax
import jax.numpy as jnp

from exoeos import IdealEOS, psir, state_trho


eos = IdealEOS()
T = 1000.0
rho = 12.0
x = jnp.array([0.85, 0.15])

alpha_r = eos.alphar(T, rho, x)
psi_r = psir(eos, T, rho * x)
state = state_trho(eos, T, rho, x)

state.P
state.Z
state.alphar
state.mu_res_RT
state.lnphi
state.gres_RT

batched_Z = jax.vmap(state_trho, in_axes=(None, 0, 0, 0))(
    eos,
    jnp.array([800.0, 1000.0]),
    jnp.array([10.0, 12.0]),
    jnp.array([[0.9, 0.1], [0.85, 0.15]]),
).Z
```

`alphar` and `state_trho` accept one state at a time: scalar `T`, scalar
`rho`, and `x` with shape `(K,)`. `psir` instead accepts the component molar
density vector `rho_vec` with shape `(K,)`. Use `jax.vmap` for batches.
Compositions are neither clipped nor normalized.

The reduced fields `alphar`, `mu_res_RT`, and `gres_RT` are dimensionless.
`psir = A_res / (R T V)` has units of `mol m-3`. This differs from
`ThermodynamicState.residual_gibbs`, which is a molar energy in `J mol-1`.

## Caloric ideal-gas API

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
[the thermodynamic-state contract](https://github.com/HajimeKawahara/exoeos/blob/main/documents/thermodynamic_state_contract.md).

## Development

```bash
python -m pip install -e ".[test]"
pytest tests/unittests
```

The current scope is the ideal gas. Cubic equations of state, including
Peng--Robinson, are reserved for later implementations behind the same
residual Helmholtz contract.
