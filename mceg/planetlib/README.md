# Pinned Planetlib adoption

ExoEOS adopts four shared partial models from Planetlib commit
[`a83b3fb2`](https://github.com/HajimeKawahara/planetlib/tree/a83b3fb255c7a383c6a8ef502328266f4dd16bff).
The [standalone contract](contract.yaml) uses native MCEG schema `0.6` and contains
16 symbolic equations, four mathematical models, provider-owned bindings and
five pending assessments. Reading it needs no Planetlib checkout.

| Shared model | Provider correspondence | Restricted scope and previous records |
| --- | --- | --- |
| `planetlib.thermodynamics.ideal_mixing.model@1` | `total_solution_gibbs_RT` and `total_solution_state` with `IdealSolution()` | Two positive amounts, zero excess energy, standards supplied at the selected T/P. `n[0:2]` and `mu0_RT[0:2]` map to the two component quantities. No extra gas pressure term is added. |
| `planetlib.thermodynamics.mass_fraction_solute.model@1` | `mass_fraction_solute_state` | Two host components and one solute. Shared `F` is `state.gibbs_RT - host_gibbs_RT`; host responses are `state.host_mu_RT - host_mu_RT`. The [historical survey](../contracts/mass_fraction_solute.yaml) retains its numerical and physical assessments. |
| `planetlib.thermodynamics.helmholtz.model@1` | `thermodynamic_state_trho` and `HelmholtzThermodynamicState` | Total molar Helmholtz energy, molar density, fixed composition and a smooth stable phase. Pressure, entropy, cv, cp and squared sound speed map to the correspondingly named state fields/properties. Composition does not equilibrate during the sound-speed response. |
| `planetlib.solubility.hirschmann2012.endmember_model@2` | H2 chemical potential obtained by differentiating `magma_volatiles.gibbs_j` | `h2_offset_rt=0`, common gas standard and fugacity in bar. `X=n_H2/(sum(n_dry_endmembers)+n_H2O+n_H2)`. Melt pressure converts from Pa to GPa explicitly. The [historical source/basis review](../contracts/magma_volatiles.yaml) and its distinct oxide denominator remain intact. |

MCEG bindings identify scalar returns and named scalar arguments, but do not
infer array components, tuple fields or differentiated outputs. Bindings here
describe those mappings in their notes and omit `result` for state returns and
the implicit Hirschmann relation. Sample tests exercise the mappings directly.

## Maintenance and checks

Maintain scientific expressions only in the selected Planetlib catalog
`Equation.sympy` records. This is Planetlib policy. ExoEOS authors local
Subjects, Contexts, Claims and Bindings in [adoption.yaml](adoption.yaml), and
chooses immutable file/model pins in [imports.lock.json](imports.lock.json).
`contract.yaml` is a generated distribution artifact; never edit its copied
equations independently. There is no Python formula registry or local schema.

The pinned MCEG specification defines
[symbolic source precedence](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/symbolic.md),
[model connections](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/mathematical_models.md),
[bindings and review pins](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/equations.md), and
[assessment history](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/assessment_lifecycle.md).
Its optional generated-expression examples do not require another maintained
formula source. Native `load_contract`, `to_sympy`, `mathematical_model_digest`
and `check_contract` perform decoding, model identity, pin, source hash and
binding checks. Planetlib's existing `assemble` supplies external-definition
selection; no resolver is duplicated here.

From the ExoEOS root, use a separate development environment:

```bash
python -m pip install -e '.[test]' -r mceg/planetlib/requirements.txt
git clone https://github.com/HajimeKawahara/planetlib.git /tmp/planetlib-source
git -C /tmp/planetlib-source checkout a83b3fb255c7a383c6a8ef502328266f4dd16bff
python /tmp/planetlib-source/tools/planetlib.py build mceg/planetlib/adoption.yaml \
  --lock mceg/planetlib/imports.lock.json --output /tmp/exoeos-planetlib.yaml
diff -u mceg/planetlib/contract.yaml /tmp/exoeos-planetlib.yaml
python -m pytest -q tests/planetlib
python -m mceg check mceg/planetlib/contract.yaml \
  --assessment exoeos.planetlib.hirschmann.pending --root .
```

The separate CI job reproduces the contract and checks selected source bytes,
native definition/binding pins, two ideal-mixture states, three mass-fraction
states, two smooth nonideal Helmholtz states and three H2 equilibrium states.
Tests consume recorded SymPy expressions. The Helmholtz test supplies an
independent example potential to evaluate their consequences. Passing samples
does not establish all-input equivalence or physical validity.

Existing packaged evidence, documentation, archived contracts and execution
artifacts retain their bytes and assessment identities. This optional adoption
record is linked from the repository guide and is not inserted into the
old-reader package catalog. Neither Planetlib nor MCEG becomes an ExoEOS
numerical runtime dependency.

## Review boundary

All new correspondence assessments and the separate physical-applicability
assessment remain `unknown`. Planetlib mathematical checks and previous
provider reviews are not imported as support for these claims. Updating a
definition lock or source hash does not renew a review or execution. For a later
update, preserve the adoption snapshot and assessment history, add the changed
definition revision/variant explicitly, and review the corresponding local
claim and mapping. Revisit `mass_fraction_solute`'s historical
`numerical_verification` scope and `magma_volatiles`'s source/basis and
`physical_accuracy` scopes before claiming that earlier evidence covers the new
definition. Empirical applicability requires a separate target-specific review.
