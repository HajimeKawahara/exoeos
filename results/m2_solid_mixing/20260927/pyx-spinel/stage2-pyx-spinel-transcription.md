# Full-site pyroxene and spinel transcription review

The frozen prototype is `stage2-pyx-spinel-prototype.py`; its generated data are
in `stage2-pyx-spinel-models.json`. These models are direct symbolic
transcriptions of MAGMA commit `705a0fb315e5054d18275a580562f6121c8e458c` from
<https://github.com/magmasource/MAGMA>. No native-curve fitting was used.

| Model source | Complete source SHA256 |
| --- | --- |
| `clinopyroxene.c` | `0a50a5171ec75f77fc439bebcd1b96f70962859eaf3c0f532d072d847fa585a0` |
| `orthopyroxene.c` | `cddc4cc48d3a829724a61589a33e96f0b909189ece1a045c6c64d775754596f7` |
| `spinel.c` | `85210623e96347735a0db09da4bc893d3702eac527042432796e980fbecf6df1` |

## Correction and source interpretation

The original prototype used `clino=False` for both the Opx site function and its
pure references. Only the site function should use that branch. Both source
files explicitly set `static const int clino = TRUE` inside **pureOrder and
purePyx**, so both phases reference the same monoclinic pure states. The source
explains this standard-state convention immediately before those routines
(`clinopyroxene.c`, lines 1499 onward; definitions at lines 1525 and 1610).
The prototype and generated JSON were corrected accordingly. Opx provenance now
names `orthopyroxene.c`; its H/S/V expressions were independently checked to be
symbolically identical to `clinopyroxene.c` with the same `clino=False` branch.
No other model correction was needed.

The ES pure entropy conversion is correct. With source `s=2*u-1`,

```
(1-s)*ln(1-s)+(1+s)*ln(1+s)-2*ln(2)
  = 2*u*ln(u)+2*(1-u)*ln(1-u).
```

Thus the source's constant `-2*R*T*ln(2)` is removed when its two unnormalized
logarithms are replaced by normalized occupations of multiplicity two. The
prototype's added `+2*R*T*ln(2)` cancels exactly that residual constant.
This entropy identity was checked symbolically for the pure references in both
source files. The corrected Cpx and Opx pure-reference arrays are identical.

For spinel, the additional source entropy is `ss4*s[2]`, and the pure magnetite
function contains the same term. The pinned defaults have `ss4=S55=0`, so its
absence from the generated nonlogarithmic polynomial is correct. Runtime
calibration resets are outside this fixed-default declaration. The HC, MT, and
SP pure entropies were checked symbolically against the four normalized sites
`u`, `1-u`, `(1-u)/2`, `(1+u)/2`, with multiplicities `1,1,2,2`; for SP the source
order parameter is `s[0]=2*u-1`.

The site-coordinate maps preserve all order variables and the source's signed
endmember domain. For example, the spinel site coordinates
`[1,0,0,0.5,0,0,0]` satisfy every declared constraint and yield native endmember
fractions `[0,-2,0,2,1]` (the Mg2TiO4 direction). The pyroxene coordinates
`[1,0,0,0,1,0,0,0]` similarly yield `[-2,1,2,0,0,0,0]` (ferrosilite). No
nonnegative-endmember simplex restriction was added. These examples illustrate
the signed directions; completeness follows from the explicit inverse site
maps and the source composition/site inequalities, not from the examples.

## Finite native compatibility

`stage2-pyx-spinel-validate.py` compares the declared equations with existing
archives under `explicit_coordinates`. It uses the native total endmember mole
count to normalize extensive Gibbs energy, then subtracts the pinned native
pure-standard chemical potentials. Pure-reference values are obtained by
bounded one-dimensional minimization including both endpoints. Remaining order
variables are minimized at each fixed archived composition.

At 2173.15 K and 267.20283416157343 bar:

| Phase | Archived comparisons | Maximum absolute mixing-G difference (J/mol) |
| --- | ---: | ---: |
| Clinopyroxene | 59 | 2.3283064365386963e-9 |
| Orthopyroxene | 59 | 2.106389729306102e-9 |
| Spinel | 62 | 4.2782630771398544e-9 |

Every final numerical optimizer reports success. Trace components near 1e-8
create very narrow order intervals; conditional bounded scalar refinement was
needed after SLSQP. Pure spinel was parameterized on its exact affine order
relation `s[1]=(1+s[0])/2`. These are verification-optimizer choices, not changes
to the declared thermodynamic models.

The detailed receipt is `stage2-pyx-spinel-validation.json`, including parameter
and archive hashes, pure-reference values, compositions, order/site coordinates,
and individual differences. No native process was launched, and no archived
result was changed. The finite comparisons do not establish a global stability
bound, native binary uniform error bound, or empirical calibration domain.
