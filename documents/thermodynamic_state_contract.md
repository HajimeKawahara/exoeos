# Thermodynamic-state contract

This document defines the initial ExoEOS public contracts. The fluid residual
layer uses reduced residual Helmholtz energy as its source of truth. The
solution layer uses reduced molar excess Gibbs energy. The existing
caloric ideal-gas interface remains separate from the residual
temperature-pressure inversion layer.

## Public interface

The top-level package exports `HelmholtzEOS`, `TPHelmholtzEOS`, `IdealEOS`,
`SecondVirialEOS`, `TRhoState`, `psir`, `state_trho`, `state_tp`,
`GibbsExcessModel`, `IdealSolution`, `SolutionState`, `total_gex_RT`,
`solution_state`, `IdealGas`, `ThermodynamicState`, `EquationOfState`, and
`__version__`.

## Residual Helmholtz interface

`HelmholtzEOS` is a structural protocol additive to the existing
`EquationOfState` temperature-pressure protocol. A residual model implements

```python
alphar(T, rho, x)
```

with the definition

```text
alphar = A_res / (n R T).
```

`A_res` excludes the complete ideal-gas Helmholtz contribution, including
ideal mixing. The inputs describe one state: `T` is a scalar temperature in K,
`rho` is a scalar total molar density in `mol m-3`, and `x` is a mole-fraction
vector with shape `(K,)`. The result is a scalar. `IdealEOS()` implements this
interface with `alphar = 0`.

`TPHelmholtzEOS` extends this structural protocol with the density hook

```python
molar_density(T, P, x, phase="vapor")
```

where `P` is absolute pressure in Pa and `phase` is a static phase/root
selector interpreted by the concrete EOS. The common
`state_tp(eos, T, P, x, phase="vapor")` function calls this hook and evaluates
the returned density through `state_trho`. Consequently both entry points use
the same residual derivatives and return the same `TRhoState` type. This is
the ExoGibbs-facing temperature-pressure layer: equilibrium calculations
typically require `rho` together with `lnphi`, rather than pressure and
`lnphi` computed from an already supplied density.

The component molar density vector is

```text
rho_vec_i = rho x_i,
rho = sum_i rho_vec_i,
x_i = rho_vec_i / rho.
```

The free-energy density helper is

```python
psi_r = psir(eos, T, rho_vec)
```

and is defined by

```text
psi_r = A_res / (R T V) = rho alphar.
```

Although `alphar` is dimensionless, `psi_r` has units of `mol m-3`.
Differentiation with respect to partial molar densities gives

```text
mu_res_i / (R T) = partial psi_r / partial rho_vec_i,
P_res / (R T) = -psi_r + sum_i rho_vec_i mu_res_i / (R T),
P = rho R T + P_res,
Z = P / (rho R T),
ln(phi_i) = mu_res_i / (R T) - ln(Z),
g_res / (R T) = sum_i x_i ln(phi_i)
                = alphar + Z - 1 - ln(Z).
```

`state_trho(eos, T, rho, x)` evaluates these derivatives and returns the
immutable JAX PyTree `TRhoState`:

| Field | Alias | Meaning | Unit |
| --- | --- | --- | --- |
| `molar_density` | `rho` | Total molar density | `mol m-3` |
| `pressure` | `P` | Absolute pressure | `Pa` |
| `compressibility_factor` | `Z` | Compressibility factor | 1 |
| `reduced_residual_helmholtz` | `alphar` | `A_res / (n R T)` | 1 |
| `reduced_residual_chemical_potentials` | `mu_res_RT` | Component `mu_res_i / (R T)` | 1 |
| `log_fugacity_coefficients` | `lnphi` | Component `ln(phi_i)` | 1 |
| `reduced_residual_gibbs` | `gres_RT` | `g_res / (R T)` | 1 |

In particular, `state_tp` returns `state.rho`, `state.Z`, `state.lnphi`, and
`state.gres_RT` along with the other fields in this table. Density inversion
and phase/root selection remain EOS responsibilities; the shared layer does
not impose a generic numerical root solver.

The numerical domain is `T > 0`, `rho > 0`, `x_i >= 0`,
`sum_i x_i = 1`, and `Z > 0`. ExoEOS performs static shape checks but does not
clip or normalize numerical inputs. `alphar`, `psir`, and `state_trho` operate
on one state; `state_tp` likewise requires scalar `T` and `P` and a vector
`x`. Use `jax.vmap` for batches. The `phase` string is static: capture it in a
transformed function or mark it static, and do not map it as an array. This
explicit scalar-state contract keeps partial-density differentiation
unambiguous.

The residual kernel supports `jax.jit` and first- and higher-order automatic
differentiation when the model does. Exact vacuum is outside the contract
because `rho_vec / sum(rho_vec)` has no defined composition there. A zero mole
fraction is allowed when the model is differentiable at that boundary.
Calculations use at least `float32` and include numerical model PyTree leaves
in dtype promotion. Model parameters must be registered as PyTree leaves to
participate in JAX transformations and this promotion rule.

`TRhoState.reduced_residual_gibbs` is dimensionless. It must not be confused
with `ThermodynamicState.residual_gibbs`, which is the dimensional molar
quantity `g_res` in `J mol-1`.

## Second-virial equation of state

`SecondVirialEOS(coefficients)` is the initial non-ideal Helmholtz backend.
`coefficients[i, j]` is a constant symmetric pair coefficient `B_ij` in
`m3 mol-1`; the constructor requires a nonempty square matrix, while symmetry
is a caller contract. For

```text
B_mix = sum_i sum_j x_i x_j B_ij,
```

the model and its derived state are

```text
alphar = rho B_mix,
Z = 1 + rho B_mix,
P = rho R T (1 + rho B_mix),
mu_res_i / (R T) = 2 rho sum_j B_ij x_j,
ln(phi_i) = 2 rho sum_j B_ij x_j - ln(Z),
g_res / (R T) = 2 rho B_mix - ln(Z).
```

For a temperature-pressure state, define the ideal-gas density
`rho_0 = P / (R T)` and discriminant

```text
D = 1 + 4 B_mix rho_0.
```

`SecondVirialEOS.molar_density` returns the mechanically stable low-density
root in cancellation-resistant form,

```text
rho = 2 rho_0 / (1 + sqrt(D)).
```

Its numerical domain is `T > 0`, `P > 0`, `x_i >= 0`,
`sum_i x_i = 1`, `D > 0`, and `Z > 0`. Only the static selector
`phase="vapor"` is supported. Numerical values, coefficient symmetry, and the
discriminant domain are caller contracts rather than traced runtime checks.
The second-virial expansion is a low-density truncation: results are credible
only where omitted third and higher virial terms are negligible. This first
model also treats `B_ij` as constants, so users must select coefficients
appropriate to the temperature range of interest.

## Excess Gibbs interface

`GibbsExcessModel` is a structural protocol separate from `HelmholtzEOS` and
`EquationOfState`. A solution backend implements

```python
gex_RT(T, P, x)
```

with the molar definition

```text
gex_RT = g_ex / (R T).
```

The inputs describe one state: `T` is a scalar temperature in K, `P` is a
scalar absolute pressure in Pa, and `x` is a normalized mole-fraction vector
with shape `(K,)`. The result is a dimensionless scalar. `IdealSolution()` is
the zero-excess placeholder. It supplies neither the ideal-mixing contribution
nor standard/endmember Gibbs energies.

For a component amount vector `n`, define

```text
n_total = sum_i n_i,
x_i = n_i / n_total,
G_ex / (R T) = n_total gex_RT(T, P, x),
ln(gamma_i) = partial [G_ex / (R T)] / partial n_i.
```

`total_gex_RT(model, T, P, n)` evaluates the third line. Its result scales in
the same amount unit as `n`; unlike molar `gex_RT`, it is not dimensionless
when `n` carries physical amount units. `solution_state(model, T, P, x)` treats
normalized `x` as the amounts of a one-mole system and obtains all
`ln(gamma_i)` values with `jax.value_and_grad`. The extensive value is divided
by the supplied total amount for the molar state field. It returns the
immutable JAX PyTree `SolutionState`:

| Field | Alias | Meaning | Unit |
| --- | --- | --- | --- |
| `reduced_excess_gibbs` | `gex_RT` | `g_ex / (R T)` | 1 |
| `log_activity_coefficients` | `lngamma` | Component `ln(gamma_i)` | 1 |

Because both values come from the same extensive scalar potential, a
normalized composition satisfies

```text
gex_RT = sum_i x_i ln(gamma_i).
```

This first contract uses only the symmetric mole-fraction standard-state
convention, `a_i = x_i gamma_i`, referenced to the pure component or specified
pure endmember at the same `T` and `P`. It does not cover unsymmetric
Henry-law, molality, electrolyte, site-fraction, or sublattice conventions.
Component ordering is a model/caller contract; the placeholder protocol does
not yet require component identifiers. A valid symmetric model has
`gex_RT(T, P, e_i) = 0` at every pure-component or pure-endmember composition
`e_i`.

The numerical domain is `T > 0`, `P > 0`, `n_i >= 0`, `sum_i n_i > 0`, and,
for `solution_state`, `sum_i x_i = 1`. ExoEOS performs static shape checks but
does not clip or perform traced value validation. The extensive construction
forms `n / sum(n)` by definition; a claimed `solution_state` mole-fraction
input that is not normalized remains outside the public contract. Exact zero
total amount is outside the contract. A zero component amount is allowed only
where the selected model is differentiable.
The operations support `jax.jit`, external `jax.vmap`, and higher amount
derivatives when `gex_RT` does. Calculations use at least `float32`, and
floating numerical model PyTree leaves participate in dtype promotion.

## Temperature-pressure ideal-gas interface

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

The caloric `IdealGas` model exposes

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
This caloric interface is separate from the residual Helmholtz `state_tp`
interface and its explicit static `phase` selector.

## Temperature-pressure ideal-gas input and shape convention

The following broadcasting contract applies specifically to
`IdealGas.state`. The residual Helmholtz and excess Gibbs scalar-state rules
are defined in their respective sections above.

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
rho_mass = P M / (R T)
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

The ideal-gas state computation and returned PyTree support `jax.jit`,
`jax.vmap`, and JAX automatic differentiation with respect to valid floating
`T` and `P` inputs, and with respect to compositions in the simplex interior
(`x_i > 0`).
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

For a condensed solution, ExoEOS supplies the scalar departure
`total_gex_RT(model, T, P, n)` and its amount derivatives. ExoGibbs remains
responsible for standard/endmember Gibbs energies, ideal mixing, mapping an
element inventory into model component order, phase amounts, and total-Gibbs
minimization. Passing only `lngamma` while retaining an ideal-mixture Hessian
would omit the derivatives of the activity coefficients; differentiating the
scalar departure preserves those terms.
