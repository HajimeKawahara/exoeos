# Thermodynamic-state contract

This document defines the initial ExoEOS public contract. The contract keeps
units, shapes, and thermodynamic reference choices explicit so that an ideal
gas and later real-gas models can share the same call site.

## Public interface

The top-level package exports `IdealGas`, `ThermodynamicState`,
`EquationOfState`, and `__version__`.

The ideal-gas constructor is

```python
IdealGas(
    molar_masses,
    molar_heat_capacities,
    *,
    reference_enthalpies=None,
    reference_entropies=None,
    reference_temperature=298.15,
    reference_pressure=1.0e5,
)
```

Component arguments are aligned arrays of shape `(K,)`. A scalar constructor
argument is accepted for a one-component model. Reference enthalpies and
entropies default to zero arrays. The heat capacities are component molar
constant-pressure heat capacities and are constant with temperature in this
first caloric closure.

Every equation of state exposes

```python
state = eos.state(T, P, x)
```

and returns an immutable JAX PyTree named `ThermodynamicState`. Its canonical
fields, in stable order, are:

| Field | Meaning | SI unit |
| --- | --- | --- |
| `compressibility_factor` | Mixture compressibility factor | 1 |
| `mass_density` | Mixture mass density | `kg m-3` |
| `number_density` | Total particle number density | `m-3` |
| `molar_enthalpy` | Mixture molar enthalpy | `J mol-1` |
| `molar_entropy` | Mixture molar entropy | `J mol-1 K-1` |
| `molar_heat_capacity_cp` | Mixture constant-pressure heat capacity | `J mol-1 K-1` |
| `molar_heat_capacity_cv` | Mixture constant-volume heat capacity | `J mol-1 K-1` |
| `adiabatic_gradient` | `(partial ln T / partial ln P)_(s,x)` | 1 |
| `log_fugacity_coefficients` | Component `ln(phi_i)` | 1 |
| `residual_gibbs` | Mixture residual molar Gibbs energy | `J mol-1` |
| `residual_enthalpy` | Mixture residual molar enthalpy | `J mol-1` |
| `thermal_expansion` | Isobaric volumetric expansion coefficient | `K-1` |

The concise aliases are `Z`, `h`, `s`, `cp`, and `cv`. They refer to
`compressibility_factor`, `molar_enthalpy`, `molar_entropy`,
`molar_heat_capacity_cp`, and `molar_heat_capacity_cv`, respectively. There is
no separate unit conversion or change of thermodynamic basis in an alias.

`EquationOfState` is a structural protocol for objects with this `state`
operation; callers do not need to inherit from a package base class.
For a future cubic backend, phase and root selection must be bound as a static
model policy so that `state(T, P, x)` remains single-valued.

## Input and shape convention

Inputs use SI units:

- `T`: temperature in K
- `P`: absolute pressure in Pa
- `x`: mole fractions, with species on the last axis
- `molar_masses`: component molar masses in `kg mol-1`
- `molar_heat_capacities`: component molar heat capacities in `J mol-1 K-1`
- `reference_enthalpies`: component reference values in `J mol-1`
- `reference_entropies`: component reference values in `J mol-1 K-1`

For `K` components, `x` has shape `(..., K)`. The leading dimensions of `T`,
`P`, and `x` follow JAX broadcasting. Scalar state outputs have the resulting
batch shape; `log_fugacity_coefficients` has that batch shape followed by
`(K,)`.

The numerical domain is `T > 0`, `P > 0`, `x_i >= 0`, and
`sum(x, axis=-1) = 1`. ExoEOS performs static shape checks but treats numerical
validity as a caller contract. In particular, it does not clip or implicitly
normalize traced values. This keeps `state` safe to use under JAX tracing.
Calculations use at least `float32` so the SI value of the Boltzmann constant
remains representable; explicit `float64` inputs or model data are preserved
when JAX 64-bit mode is enabled.

## Ideal-gas closure

Let `R` be the molar gas constant, `k_B` the Boltzmann constant, and define the
mixture molar mass and heat capacity by

```text
M = sum_i x_i M_i
cp = sum_i x_i cp_i
```

The ideal-gas state is

```text
Z = 1
rho = P M / (R T)
number_density = P / (k_B T)
cv = cp - R
adiabatic_gradient = R / cp
thermal_expansion = 1 / T
ln(phi_i) = 0
G_residual = 0
H_residual = 0
```

With reference temperature `T_ref`, reference pressure `P_ref`, component
reference enthalpies `h_ref_i`, and component reference entropies `s_ref_i`,
the constant-component-`cp` convention is

```text
h = sum_i x_i [h_ref_i + cp_i (T - T_ref)]
s = sum_i x_i [s_ref_i + cp_i ln(T / T_ref)]
    - R ln(P / P_ref) - R sum_i x_i ln(x_i)
```

The limiting value `x_i ln(x_i) = 0` is used when `x_i = 0`. Zero default
reference arrays define a relative caloric model, not an absolute
thermochemical database.

## JAX behavior

The state computation and returned PyTree support `jax.jit`, `jax.vmap`, and
JAX automatic differentiation with respect to valid floating `T` and `P`
inputs, and with respect to compositions in the simplex interior (`x_i > 0`).
The entropy value uses the finite `x_i ln(x_i)` limit at `x_i = 0`, but its
composition derivative is physically singular there. Use a differentiable
simplex parameterization such as `softmax` when optimizing composition.
`IdealGas` is also a registered PyTree whose constructor arrays and scalar
reference values are numerical leaves, so model parameters can participate in
transformations. A model can be captured in a transformed function, passed as
a PyTree, or its bound `state` method can be transformed directly. Input
shapes and the number and ordering of components remain static during a
compiled call.

## ExoGibbs and ExoJAX boundaries

ExoEOS does not infer units from an upstream package. Convert explicitly at
the package boundary:

```text
P_Pa = P_bar * 1.0e5
n_cm-3 = number_density_m-3 * 1.0e-6
rho_g_cm-3 = mass_density_kg_m-3 * 1.0e-3
M_kg_mol-1 = M_g_mol-1 * 1.0e-3
```

ExoGibbs pressure values are in bar, so they require the first conversion.
Its equilibrium result `x` is a mole-fraction input, while its elemental
inventory `b` is not. ExoJAX commonly uses pressure in bar, number density in
`cm-3`, mass density in `g cm-3`, and molar-mass numbers in `g mol-1`; use all
four conversions as applicable. Temperature is in K in all three packages.
