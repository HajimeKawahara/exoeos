# Independent solid-expression transcription evidence

This archive preserves the independent Rhm, plagioclase, pyroxene, and spinel
transcriptions and their finite comparisons against previously saved native
candidates. The original scripts, parameter fragments, technical notes,
receipts, and native inputs are copied byte for byte. No native worker or new
global stability calculation was run to assemble this archive.

The production provider is documented in
[the solid-mixing example](../../../examples/m2_solid_mixing/README.md).
Its full-site extension is reviewed in
[ExoEOS PR #36](https://github.com/HajimeKawahara/exoeos/pull/36), based on
[PR #35](https://github.com/HajimeKawahara/exoeos/pull/35).
The chemical global-bound consumer is a separate responsibility, reviewed in
[ExoGibbs PR #235](https://github.com/HajimeKawahara/exogibbs/pull/235).

## What the evidence establishes

- A formal global bound applies to the declared real-arithmetic expression,
  with its serialized binary64 coefficients, physical site domain, free
  ordering coordinates, fixed standards, and enclosed pure-reference minima.
  The global-bound proof trees are not part of this finite-comparison archive.
- These finite checks test expression wiring and compatibility with specific
  saved outputs of the pinned native build. Neither local ordering searches
  nor bounded scalar minimization is a formal global-minimum proof. Finite
  agreement does not give a uniform error bound over untested compositions.
- The evidence does not establish equality with every native build, a bound
  on native floating-point/logarithm/clamping errors, an empirical calibration
  domain, or stability against omitted phases. The native Fe/Ni alloys also
  do not represent the source Fe/Si/O/H alloy.

No model was inferred by fitting the native candidate values. Rhm, pyroxene,
and spinel came from symbolic source transcription. Plagioclase came from
symbolic interpretation of the pinned binary instructions and loaded constants.

## Source provenance

The public MAGMA sources are pinned to commit
`705a0fb315e5054d18275a580562f6121c8e458c`.

| Model | Primary source | Complete source SHA256 |
| --- | --- | --- |
| Rhm oxide, five endmembers | [rhomsghiorso.c](https://github.com/magmasource/MAGMA/blob/705a0fb315e5054d18275a580562f6121c8e458c/sources/rhomsghiorso.c) | `9c06e35c78c8e522bb05d6d1933fa34806ddc927c2002980872d5fbeb00cec5e` |
| Clinopyroxene | [clinopyroxene.c](https://github.com/magmasource/MAGMA/blob/705a0fb315e5054d18275a580562f6121c8e458c/sources/clinopyroxene.c) | `0a50a5171ec75f77fc439bebcd1b96f70962859eaf3c0f532d072d847fa585a0` |
| Orthopyroxene | [orthopyroxene.c](https://github.com/magmasource/MAGMA/blob/705a0fb315e5054d18275a580562f6121c8e458c/sources/orthopyroxene.c) | `cddc4cc48d3a829724a61589a33e96f0b909189ece1a045c6c64d775754596f7` |
| Spinel | [spinel.c](https://github.com/magmasource/MAGMA/blob/705a0fb315e5054d18275a580562f6121c8e458c/sources/spinel.c) | `85210623e96347735a0db09da4bc893d3702eac527042432796e980fbecf6df1` |
| Plagioclase | `gmixPlg` in the [official alphaMELTS 2.3.2 Ubuntu 22.04 release](https://github.com/magmasource/alphaMELTS/releases/tag/v2.3.2) | binary: `c218ef6f7ba5aef4b0760f3176530d1335457d6e2716823a1d0de3a5d13301e5` |

Plagioclase has the separate provenance class
`pinned_native_binary_instruction_transcription`. The release ZIP SHA256 is
`97ec2cdb53cae69822a41b5639d8b93edf95b2c361bc15e64e53b192ea9e1425`;
the complete `gmixPlg` symbol bytes have SHA256
`ea8e10b7d7ac9a5b8bb75473a91450ef92cdb188c21cbabc547bedce2d9997f3`.
The thermodynamic C build commit inside that binary has not been established.
The 107-instruction Gibbs-output path, exact loaded binary64 constants, ELF
address checks, and symbolic algebra checks are in the
[instruction audit](plagioclase/stage2-plagioclase-instruction-audit.json) and
[disassembly](plagioclase/stage2-plagioclase-disassembly.txt).
The binary itself is not redistributed here.

## Model conventions that matter

- **Rhm:** the five-endmember `rhomsghiorso.c` expression is used, not the old
  four-endmember `rhombohedral.c`. The whole Mn-free face retains native indices
  `[0,1,2,4]` and all surviving ordering variables. The eight default SRO spline
  ordinates, at 873.15 through 1573.15 K, are exactly `0.0730205`; the natural
  spline and its endpoint extrapolation are therefore constant. This algebra
  does not establish an empirical temperature range. Parameter resets are
  outside the declaration. The historical fragment distinguishes local
  `endmember_index` from `native_endmember_index`; the production adapter uses
  the native index for reference corrections.
- **Both pyroxenes:** the site function uses the appropriate clino/ortho
  branch, while **both `pureOrder` and `purePyx` explicitly use `clino=TRUE`**.
  Opx must therefore subtract the monoclinic pure reference too. The archived
  prototype and finite receipts include this correction. Normalizing the ES
  entropy sites also requires retaining the source's `log(2)` cancellation.
- **Spinel:** the default `ss4=S55=0` is retained. All ordering variables and
  signed native endmember regions admitted by the site constraints remain in
  the declaration. Negative basis coefficients do not imply negative physical
  site occupancies.
- **Plagioclase:** the recovered enthalpy parameters differ from the public
  alkali-feldspar defaults. The declared closed simplex uses continuous
  `0*log(0)=0`; native epsilon clamps and floating-point operations are separate
  numerical behavior, not silently identified with this real-arithmetic model.

The detailed preserved notes are
[Rhm](rhm/stage2-rhm-transcription.md),
[plagioclase](plagioclase/stage2-plagioclase-transcription.md), and
[pyroxene/spinel](pyx-spinel/stage2-pyx-spinel-transcription.md).
Their `/tmp` paths describe the original run; archived counterparts keep the
same basenames. The production transcription now sorts generated entropy
terms deterministically; historical fragments preserve their original order.

## Finite comparison receipts

The saved comparisons use 2173.15 K and 267.20283416157343 bar. Native extensive
Gibbs energy is divided by the actual native endmember total, including tiny
finite-difference changes to that total, before subtracting the pure standards.
Each row below includes the archived trials and the saved final best trial.

| Phase | Compared points | Maximum absolute mixing-G difference (J/mol formula) |
| --- | ---: | ---: |
| Rhm oxide | 62 | `1.396301740896888e-9` |
| Plagioclase | 62 | `1.4897523215040565e-9` |
| Clinopyroxene | 59 | `2.3283064365386963e-9` |
| Orthopyroxene | 59 | `2.106389729306102e-9` |
| Spinel | 62 | `4.2782630771398544e-9` |

See the complete [Rhm receipt](rhm/stage2-rhm-validation.json),
[plagioclase receipt](plagioclase/stage2-plagioclase-validation.json), and
[pyroxene/spinel receipt](pyx-spinel/stage2-pyx-spinel-validation.json).
Their parameter and native-trial hashes are checked against the archived bytes.
The trial inputs are copies of the
[earlier ExoGibbs archive](https://github.com/HajimeKawahara/exogibbs/tree/2268924b0a5ea6238d1540a4e12c0afc42c92574/results/m2_stability_search/20260927/explicit_coordinates).
They are not new native evaluations at the later freshly solved source.

## Archive layout and replay

`provenance.json` records original paths, source URLs, full hashes, and roles.
`manifest.json` hashes every archived file except itself. The `inputs` directory
holds five byte-preserved trial JSONs, five pure-standard receipts, the host
receipt, and the historical native receipt producer. The latter is retained
for provenance and is not invoked by replay. Large binaries and Python/SymPy
environments are excluded.

From this directory, verify all bytes and receipt cross-links with Python's
standard library only:

```sh
python replay.py --check-only
```

To repeat a finite comparison, install NumPy and SciPy and select `rhm`,
`plagioclase`, or `pyx-spinel`. Output must be outside this immutable archive:

```sh
python replay.py --group plagioclase --output-directory /tmp/solid-finite-replay
```

`replay.py` relocates only historical absolute-path string literals in the
preserved scripts. It does not change their numerical operations, rewrite
saved results, or call native MELTS. Different SciPy versions can select
slightly different finite optimizer states; a new result remains a separate
receipt rather than a replacement for the saved one.

To regenerate an independent fragment, use NumPy and SymPy 1.14.0, the pinned
MAGMA source directory, or the pinned binary plus `objdump` for plagioclase:

```sh
python replay.py --mode transcribe --group rhm \
  --source-directory /path/to/MAGMA/sources --output-directory /tmp/solid-rhm-replay
python replay.py --mode transcribe --group pyx-spinel \
  --source-directory /path/to/MAGMA/sources --output-directory /tmp/solid-pyx-replay
python replay.py --mode transcribe --group plagioclase \
  --binary /path/to/libalphamelts.so --output-directory /tmp/solid-plg-replay
```

The original pyroxene/spinel prototype's `Macros` dependency is snapshotted
from ExoEOS `38475d56a7cbf0b00aa1e51ef6d6e61e4c4291fc`. A regenerated historical
fragment may serialize entropy terms in a different order. Production model
generation and formal chemical bounds use their separately pinned code and
are not inferred from a replay's successful exit status.

## Provider implementation checks

The full provider suite passed **469 tests** at `38475d5`. The final change
at `0515534` only makes entropy-term serialization deterministic: all 20
expressions were checked equal modulo that order, a second hash-seed run
reproduced identical ordered fragments, and all **9 solid provider tests**
passed. Original logs and exact outcomes are in [validation](validation/checks.json).
