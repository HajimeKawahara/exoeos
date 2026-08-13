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
to use the residual Helmholtz, TP inversion, and excess Gibbs APIs documented
below:

```bash
python -m pip install .
```

## Documentation

Install the documentation dependencies and build the committed notebook-based
tutorials with:

```bash
python -m pip install -e ".[docs]"
./update_doc.sh
```

The independent-reference notebooks additionally require the pinned reference
backend:

```bash
python -m pip install -e ".[docs,reference]"
```

The executable notebooks are the editable sources. After changing one, run it
with `jupyter nbconvert --to notebook --execute --inplace <notebook>`, then run
`python documents/tutorials/convert_notebooks.py` to refresh its committed RST
and image assets.

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

`IdealEOS` also implements pressure inversion, so it can be compared with
other TP-capable residual models through the same entry point:

```python
from exoeos import state_tp

ideal_state = state_tp(IdealEOS(), T=1000.0, P=1.0e5, x=x)
```

`alphar` and `state_trho` accept one state at a time: scalar `T`, scalar
`rho`, and `x` with shape `(K,)`. `psir` instead accepts the component molar
density vector `rho_vec` with shape `(K,)`. Use `jax.vmap` for batches.
Compositions are neither clipped nor normalized.

Models that can invert pressure additionally implement `TPHelmholtzEOS` by
providing `molar_density(T, P, x, phase="vapor")`. The common
`state_tp(eos, T, P, x, phase="vapor")` entry point delegates density and
root selection to that hook, then evaluates `state_trho` and returns the same
`TRhoState`. This gives ExoGibbs both molar density and fugacity coefficients
from its natural `T`, `P`, and `x` inputs.

```python
from exoeos import SecondVirialEOS, state_tp


eos = SecondVirialEOS(
    jnp.array([[1.0e-4, 2.0e-5], [2.0e-5, 8.0e-5]])  # B_ij, m3 mol-1
)
state = state_tp(eos, T=700.0, P=2.0e5, x=jnp.array([0.4, 0.6]))

state.rho
state.Z
state.lnphi
state.gres_RT
```

`SecondVirialEOS` is the first non-ideal fluid model. It uses a constant,
symmetric pair-coefficient matrix and

```text
B_mix = sum_i sum_j x_i x_j B_ij,
alphar = rho B_mix,
Z = 1 + rho B_mix,
P = rho R T (1 + rho B_mix),
mu_res_i / (R T) = 2 rho sum_j B_ij x_j,
ln(phi_i) = mu_res_i / (R T) - ln(Z),
g_res / (R T) = 2 rho B_mix - ln(Z).
```

For `state_tp`, let `rho_0 = P / (R T)` and
`D = 1 + 4 B_mix rho_0`. The vapor root is evaluated as
`rho = 2 rho_0 / (1 + sqrt(D))`. Its domain requires `T > 0`, `P > 0`,
normalized nonnegative `x`, `Z > 0`, and `D > 0`. Only
`phase="vapor"` is supported. The second-virial truncation is a low-density
model and should be used only where neglected higher virial terms are small;
the constant coefficients also omit real temperature dependence.

`PengRobinsonEOS` adds a cubic EOS using critical temperatures in K, critical
pressures in Pa, acentric factors, and optional binary interaction parameters:

```python
from exoeos import PengRobinsonEOS


eos = PengRobinsonEOS(
    critical_temperatures=jnp.array([190.564]),
    critical_pressures=jnp.array([4_599_200.0]),
    acentric_factors=jnp.array([0.01142]),
)
vapor = state_tp(eos, T=150.0, P=1.0e6, x=jnp.array([1.0]))
liquid = state_tp(
    eos,
    T=150.0,
    P=1.0e6,
    x=jnp.array([1.0]),
    phase="liquid",
)
```

The implementation uses the original PR76 alpha correlation for every
component and the exact critical-condition coefficients
`Omega_a = 0.45723552892138218938` and
`Omega_b = 0.077796073903888455972`.

`PengRobinsonEOS.second_virial_coefficients(T)` returns the exact
low-density, density-form coefficient matrix of that PR model at `T`. It can
be passed directly to `SecondVirialEOS` when comparing PR with its consistent
second-virial truncation.

A small curated critical-property table is available through
`get_critical_properties(formula)`. It currently contains `CO`, `H2O`, `CO2`,
and `H2`, with source URLs retained in every record.

The optional `binary_interaction_parameters` matrix defaults to zero and uses
`a_ij = (1 - k_ij) sqrt(a_i a_j)`. `phase="vapor"` selects the largest
physical real root of the Peng-Robinson cubic, while `phase="liquid"`
selects the smallest. Both selectors return the same root in a one-root
region. Density derivatives are defined away from multiple roots; exact
critical and spinodal states are not differentiable root-selection points.

Use `float64` when evaluating very-low-pressure liquid roots: reconstructing a
small pressure from dense-liquid Helmholtz terms is ill-conditioned in
`float32` even when the selected density root is accurate.

Like `state_trho`, `state_tp` accepts one scalar state at a time. Use
`jax.vmap` for batches. The `phase` string is a static selector: capture it in
the transformed function or mark it static rather than mapping it.

The reduced fields `alphar`, `mu_res_RT`, and `gres_RT` are dimensionless.
`psir = A_res / (R T V)` has units of `mol m-3`. This differs from
`ThermodynamicState.residual_gibbs`, which is a molar energy in `J mol-1`.

`ZhangDuanEOS` implements the Zhang-Duan (2009) corresponding-states EOS for
C-O-H fluids. The published species parameters and the fitted H2O-CO2 and
H2O-CH4 interactions are available through `from_species`:

```python
from exoeos import ZhangDuanEOS


species = ("CO", "H2O", "CO2", "H2")
eos = ZhangDuanEOS.from_species(species)
state = state_tp(
    eos,
    T=1000.0,
    P=1.0e9,
    x=jnp.array([0.4, 0.4, 0.1, 0.1]),
)
```

The model also supports `CH4`, `O2`, and `C2H6`. It is intended for the
homogeneous-fluid calibration ranges reported in
[Zhang and Duan (2009)](https://doi.org/10.1016/j.gca.2009.01.021).
The principal mixture range is 673--2573 K and 1 MPa--10 GPa; H2O-CH4 starts
at 10 MPa, and the pure-species data ranges differ.
Pressure inversion selects the mechanically stable root connected to the
low-density branch and supports only `phase="vapor"`. Fugacity coefficients
are obtained by differentiating the residual Helmholtz energy rather than by
transcribing the paper's mixture fugacity equation. The implementation uses
the physical `P V / (R T)` compressibility; the scaled left-hand side printed
in Equation 8 does not reproduce the paper's Table 6 values.

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

`SecondVirialEOS` and `PengRobinsonEOS` are the non-ideal fluid backends.
Additional fluid EOS and nonzero Gibbs-excess models can be added behind the
separate `TPHelmholtzEOS` and `GibbsExcessModel` contracts.
