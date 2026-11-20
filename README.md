# ExoEOS

Differentiable equations of state and excess free-energy models for planetary
atmospheres, fluids, and melts, built with JAX.

ExoEOS provides fluid density and fugacity, solution activities, caloric
properties, tabulated EOS, and native JAX Gibbs models for silicate liquids
and minerals. It evaluates properties at a supplied state and composition;
phase-equilibrium solvers remain separate.

## Installation

Requires Python 3.9 or newer.

```bash
python -m pip install "exoeos>=0.3.0"
```

## Quick start

Evaluate CO2 density and fugacity with the Peng–Robinson EOS:

```python
import jax.numpy as jnp
from exoeos import PengRobinsonEOS, get_critical_properties, state_tp

co2 = get_critical_properties("CO2")
eos = PengRobinsonEOS(
    critical_temperatures=jnp.array([co2.critical_temperature]),
    critical_pressures=jnp.array([co2.critical_pressure]),
    acentric_factors=jnp.array([co2.acentric_factor]),
)
state = state_tp(eos, T=400.0, P=1.0e6, x=jnp.array([1.0]))

print(state.rho)    # Molar density, mol/m3
print(state.Z)      # Compressibility factor, dimensionless
print(state.lnphi)  # Log fugacity coefficients, dimensionless
```

Here `T` is in K, `P` is in Pa, and `x` contains normalized mole fractions
in component order. `state_tp` evaluates one state; use `jax.vmap` for batches.
Compositions are not clipped or normalized. Check each model's domain and
root policy in the [model guide](documents/model_guide.rst).
For heat capacities and sound speed, combine a residual EOS with an ideal
free energy using [HelmholtzThermodynamics](documents/thermodynamic_derivatives.rst).

## Documentation

- [Capability overview](documents/overview.rst) — available models and scope.
- [Model selection and usage](documents/model_guide.rst) — entry points,
  examples, and applicability.
- [Plot gallery](documents/feature_plots.rst) — reproducible comparisons.
- [Thermodynamic-state contract](documents/thermodynamic_state_contract.md) —
  units, shapes, reference states, and JAX transformations.
- [Material sensitivity](documents/material_sensitivity.rst) — chemical
  potentials and directional/adjoint responses from an extensive Gibbs scalar.

The [Japanese explanation](https://github.com/HajimeKawahara/doc_ExoEOS) is
maintained separately.

<details>
<summary>Model-specific guides and API entry points</summary>

| Capability | Entry point | Guide |
| --- | --- | --- |
| Residual fluid density and fugacity | `state_tp`, `state_trho` | [Peng–Robinson](documents/tutorials/peng_robinson_eos.rst), [Zhang–Duan](documents/tutorials/zhang_duan_eos.rst) |
| Solution Gibbs energy and activities | `solution_state`, `total_solution_state` | [Fe–Si–O](documents/fe_si_o_reference.rst), [Fe–Si–O–H](documents/fe_si_o_h_reference.rst) |
| Ideal-gas and total caloric properties | `IdealGas`, `HelmholtzThermodynamics` | [Helmholtz derivatives](documents/thermodynamic_derivatives.rst) |
| H/He and silicate–hydrogen tables | `ChabrierDebrasTableLoader`, `MarcumSilicateHydrogenTableLoader` | [Table models](documents/model_guide.rst#tables-caloric-quantities-and-density), [potential-consistent H/He](documents/potential_tables.rst) |
| Composite mass density | `mass_density_tp`, `AdditiveVolumeCompositeDensityProvider` | [Density providers](documents/model_guide.rst#tables-caloric-quantities-and-density) |
| Dry and hydrous silicate liquids | `exoeos.magma` | [Native magma and cooling examples](documents/native_magma.rst), [hydrous liquids](documents/native_hydrous_liquid.rst) |
| Dry and hydrous minerals | `exoeos.magma_solids` | [Dry minerals](documents/native_dry_solids.rst), [hydrous minerals](documents/native_hydrous_solids.rst) |
| Carbonate liquids and H2O–CO2 fluids | `exoeos.magma_carbonate`, `exoeos.magma_fluid` | [Carbonate and mixed fluids](documents/native_carbonate_liquid.rst), [H2O fluid connection](documents/native_magma_fluid.rst) |
| Conditional volatile liquids and alloys | `exoeos.magma_volatiles`, `exoeos.associated_metal` | [Conditional materials](documents/conditional_materials.rst) |

Native JAX magma models run without an external MELTS runtime. The optional
[supplied-composition MELTS evaluator](documents/melts_liquid_evaluator.rst)
and [independent silicate references](documents/melts_silicate_reference.rst)
describe separate checkout tools and their backend requirements.
The [C/N/S provider scope](documents/cns_provider_scope.rst) records remaining
standard-state and calibration limits.

[Experimental MCEG metadata](mceg/README.md) records model assumptions,
applicability, and scoped evidence.

</details>

## Development

<details>
<summary>Install from source and run tests</summary>

From a repository checkout:

```bash
python -m pip install -e ".[test]"
pytest tests/unittests
```

For a regular installation from the checkout, use `python -m pip install .`.

</details>

<details>
<summary>Build documentation and update notebooks</summary>

Follow the [documentation build guide](documents/mceg/build.rst) to install
the documentation extras, MCEG, and Graphviz, then run:

```bash
./update_doc.sh
```

Independent-reference notebooks additionally require the pinned reference
backend:

```bash
python -m pip install -e ".[docs,reference]"
```

The executable notebooks are the editable sources. After changing one,
execute it and refresh its committed RST and image assets:

```bash
jupyter nbconvert --to notebook --execute --inplace <notebook>
python documents/tutorials/convert_notebooks.py
```

</details>
