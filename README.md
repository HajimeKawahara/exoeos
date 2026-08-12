# ExoEOS

Differentiable equations of state and excess free-energy models for planetary
atmospheres, fluids, and melts, built with JAX.

Residual equation-of-state models use reduced residual Helmholtz energy as
their source of truth. Mole-fraction solution models use reduced molar excess
Gibbs energy, from which logarithmic activity coefficients are obtained by
automatic differentiation. A calorically perfect ideal-gas mixture remains
available through the original temperature-pressure state interface.

## Installation

```bash
python -m pip install exoeos
```

PyPI version `0.1.0` provides the caloric ideal-gas API only. Until a newer
release is published, install a repository checkout containing these changes
to use the residual Helmholtz and excess Gibbs APIs documented below:

```bash
python -m pip install .
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

## Excess Gibbs API

Solution models implement

```text
gex_RT(T, P, x) = g_ex / (R T),
```

where `P` is absolute pressure in Pa. The extensive helper and its amount
derivative are

```text
G_ex / (R T) = total_gex_RT(model, T, P, n)
             = n_total gex_RT(T, P, n / n_total),
ln(gamma_i) = partial [G_ex / (R T)] / partial n_i.
```

```python
import jax
import jax.numpy as jnp

from exoeos import IdealSolution, solution_state, total_gex_RT


model = IdealSolution()
T = 1600.0
P = 1.0e5
x = jnp.array([0.4, 0.6])

gex_RT = model.gex_RT(T, P, x)
total = total_gex_RT(model, T, P, x)
state = solution_state(model, T, P, x)

state.gex_RT
state.lngamma

batched_lngamma = jax.vmap(solution_state, in_axes=(None, 0, 0, 0))(
    model,
    jnp.array([1400.0, 1600.0]),
    jnp.array([1.0e5, 2.0e5]),
    jnp.array([[0.3, 0.7], [0.4, 0.6]]),
).lngamma
```

`IdealSolution` is the zero-excess placeholder: `gex_RT` and `lngamma` are
zero. It does not add the ideal-mixing Gibbs energy. The API uses only the
symmetric mole-fraction standard-state convention, `a_i = x_i gamma_i`, and
satisfies `gex_RT = sum_i x_i ln(gamma_i)` for normalized compositions.
The reference is the pure component, or a specified pure endmember, at the
same `T` and `P`.
Standard/endmember Gibbs energies, component-basis mapping, and phase
equilibrium remain responsibilities of the calling application.

`gex_RT` and `solution_state` accept one state at a time: scalar `T`, scalar
`P`, and normalized `x` with shape `(K,)`. `total_gex_RT` instead accepts a
component amount vector and forms `x = n / sum(n)`. Inputs are neither clipped
nor numerically validated; the extensive construction uses `n / sum(n)` by
definition. Use `jax.vmap` for batches.

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

The current concrete models are ideal placeholders. Non-ideal fluid EOS
backends and nonzero Gibbs-excess models are reserved for later implementations
behind the separate `HelmholtzEOS` and `GibbsExcessModel` contracts.
