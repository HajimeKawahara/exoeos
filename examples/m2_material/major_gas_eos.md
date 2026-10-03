# Conditional major-gas residual potential

`major_gas_eos.py` supplies one residual potential for the complete declared
M2 gas catalog. It uses the existing `SecondVirialEOS` and `state_tp`; it does
not change the core EOS API. The coefficients and primary-source transcription
from the [earlier diagnostic](gas_virial.py) and its
[source ledger](gas_virial_sources.json) are preserved byte for byte from
ExoEOS commit `3d36c978c4f60545e2961d6af1d30c549e635b9b`.

The factory `make_major_gas_eos(gas_species, gas_eos_options)` requires an
ordered, unique catalog containing the exact names `H2`, `He1`, and `H2O1`.
The consumer handles `gas_eos_options=None` as its existing ideal model.
Nonideal options must declare all three fields:

```python
options = {
    "h2_he_cm3_mol": 10.0,
    "water_cross_temperature_policy": "extrapolate",
    "trace_pair_policy": "zero",
}
```

The initial comparison comprises the six combinations of H2-He coefficients
0, 10, and 20 cm3/mol and water-cross policies `extrapolate` and `hold_2000`.
These are conditional constitutive alternatives, not empirical error bounds.
The H2-He coefficient is constant in temperature by explicit assumption.
Above 2000 K, `extrapolate` continues the recorded analytic H2-H2O and He-H2O
correlations. `hold_2000` holds just these two coefficients at their 2000 K
values. It leaves the pure-fluid coefficients unchanged and has a temperature
derivative corner at 2000 K. Neither choice asserts a measured cross
coefficient at the 2173.15 K source. The original diagnostic remains a
three-component conditioned calculation and is not substituted for this
full-catalog model.

## Thermodynamic basis and API

At temperature T in K, the symmetric matrix B is in m3/mol. Its major-three
block uses the declared coefficients. Every other pair, including a trace
species paired with a major species, is explicitly zero. For full-catalog
amounts n, N = sum(n), x = n/N, and Bmix = x.T B x. No subset is renormalized.
With molar density rho and pressure P in Pa, the single potential gives

```text
A^r/(NRT) = rho Bmix
I = P/(RT)
rho = 2 I / (1 + sqrt(1 + 4 I Bmix))
Z = 1 + rho Bmix
G^r/(NRT) = 2 rho Bmix - log(Z)
ln(phi_i) = 2 rho (B x)_i - log(Z)
```

Consequently `d(G^r/RT)/dn_i = ln(phi_i)` at fixed T/P, and adding the ideal
pressure term gives `d(G/RT)/dP = N/(rho R T)`. Even a species whose entire
coefficient row is zero has `ln(phi_i) = -log(Z)`. Setting that chemical
potential correction to zero would change the declared scalar model.

The model methods are:

| Method or property | Result |
| --- | --- |
| `coefficients(T_K)` | Full-catalog symmetric JAX matrix in m3/mol |
| `state(T_K, P_Pa, x)` | Existing ExoEOS state with `.rho`, `.Z`, `.lnphi`, `.gres_RT` |
| `gibbs_residual_rt(T_K, P_Pa, n)` | Extensive G^r/(RT), differentiated over full n |
| `mass_density(T_K, P_Pa, x, molar_masses_kg_mol)` | Gas mass density in kg/m3, with full-catalog masses |
| `metadata` | State-independent species order, options, schema, and basis |
| `parameters(T_K, P_Pa)` | Frozen coefficient matrix, formulas, file hashes, assumptions, and domain diagnostic |

The implementation range is 1000-3000 K. The source ledger distinguishes this
range from each correlation's physical evidence and uncertainty. The consumer
must evaluate `parameters` at each concrete T/P before a traced solve, retain
its receipt, and supply normalized nonnegative full-catalog fractions or
nonnegative amounts with positive total. Under JAX tracing these numerical
domain conditions are caller contracts, as in the underlying EOS. Eager
calls check scalar T/P, composition, and the strictly stable density root.

The existing material comparison functions accept explicit coefficients too:
`assess_material_state` and `published_basalt_h2_extrapolation` accept
`hydrogen_fugacity_coefficient=1.0`; `assess_sossi_water_state` additionally
accepts `water_fugacity_coefficient=1.0`. Nonideal calls retain the supplied
partial pressures and separately report the coefficients and actual
fugacities. The illustrative Henry law and the square-root fugacity predictors
use those fugacities. The coefficient-one defaults retain the existing report
structure and values, apart from current evaluator provenance hashes.
Fugacity may exceed total pressure; the sum-of-partial-pressures check still
uses actual partial pressures. No measured model coefficient or chemical
standard is refitted by this diagnostic extension.

## Full-simplex curvature for an independent verifier

At a fixed T/P, define bmin = min(B), b = max(abs(B)), and

```text
dmin = sqrt(1 + 4 I bmin)
rhomax = 2 I / (1 + dmin)
alpha = 1 - 2 rhomax b - 4 rhomax^2 b^2 / dmin
```

The bounds require a strictly positive discriminant and alpha. They hold on
the entire nonnegative catalog simplex, not just near the saved composition.
For a simplex-tangent variation v, the ideal-plus-residual molar Gibbs
Hessian is

```text
diag(1/x) + 2 rho B
    - 4 rho^2/(1 + 2 rho Bmix) (B x)(B x).T
```

The inequalities `abs(v.T B v) <= b (sum(abs(v)))^2`,
`abs(v.T B x) <= b sum(abs(v))`, and
`(sum(abs(v)))^2 <= sum(v_i^2/x_i)` give the lower bound
`alpha sum(v_i^2/x_i)`. Boundary faces follow by continuity. This establishes
a positive entropy curvature that can support a global Gibbs lower bound.
Separately, `1 - 2 rhomax b > 0` is a sufficient full Helmholtz concentration
curvature bound.

`parameters` reports binary64 diagnostic values and explicitly marks them
`certified=False`. A chemical-equilibrium verifier must independently enclose
the supplied frozen binary64 coefficients and these expressions with outward
arithmetic before using them in a global phase certificate. A diagnostic
float or local numerical Hessian is not itself such a certificate. The receipt
records both the original transcription hashes and hashes of the actual
example/core files used by the evaluator.

## Scope of the comparison and verification

The model does not bound omitted trace pairs or higher virials. A small total
trace mole fraction does not bound a trace chemical potential. It also does
not supply empirical accuracy for the high-temperature continuations or the
H2-He scenarios. Coupled atmospheric mass, elemental partition, and phase
acceptance require fresh source/parcel chemistry, density-based column
geometry, finite-inventory pressure closure, and affected phase evidence.
This provider alone establishes none of those scientific outcomes.

Targeted verification uses CPU only:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q \
  tests/unittests/m2_major_gas_eos_test.py \
  tests/unittests/second_virial_test.py tests/unittests/helmholtz_test.py
```

The tests cover scalar/fugacity and pressure/volume derivatives, extensivity,
Hessian symmetry, full-catalog trace dilution, zero-amount faces, species
ordering, the ideal zero-major limit, all six option combinations, the
temperature policy, JAX transformations, and invalid inputs. They do not run
the planetary closure or establish new material adoption.
