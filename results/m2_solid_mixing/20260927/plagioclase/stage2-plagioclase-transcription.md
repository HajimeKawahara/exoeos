# Plagioclase expression from the pinned alphaMELTS binary

This expression has a different provenance class from the published MAGMA
source transcriptions. It was recovered directly from `gmixPlg` in the official
alphaMELTS 2.3.2 Ubuntu 22.04 x86_64 runtime already pinned for the parent work.
The binary SHA256 is
`c218ef6f7ba5aef4b0760f3176530d1335457d6e2716823a1d0de3a5d13301e5`.
Its local path is
`/tmp/exoeos-pr3-melts/runtime/alphamelts-py-2.3.2-ubuntu_22_04-x86_64/libalphamelts.so`.
The corresponding upstream project is <https://github.com/magmasource/alphaMELTS>.
This audit does not establish a source commit for the thermodynamic code inside
that binary. It must not be labeled as the published MAGMA `feldspar.c` model or
as the equation used by every alphaMELTS build.

## Artifacts

- `stage2-plagioclase-disassemble.py`: reproducible symbolic instruction lift.
- `stage2-plagioclase-parameters.json`: provider-shaped declared model.
- `stage2-plagioclase-instruction-audit.json`: instruction path, constants,
  binary/symbol hashes, and exact symbolic identity checks.
- `stage2-plagioclase-disassembly.txt`: complete `gmixPlg` disassembly.
- `stage2-plagioclase-validate.py`: finite comparison with archived native probes.
- `stage2-plagioclase-validation.json`: finite comparison receipt.

All artifacts were written under `/tmp`. No native worker was launched and no
finite-point polynomial fit was used.

## Instruction audit

The ELF symbol starts at virtual address `0x29f300` and occupies `0xb04` bytes.
Those complete function bytes have SHA256
`ea8e10b7d7ac9a5b8bb75473a91450ef92cdb188c21cbabc547bedce2d9997f3`.
The script checks the complete binary hash and reads its ELF section headers to
verify the file-offset mapping for the function and each rodata operand.

The prologue at `0x29f328` through `0x29f36b` loads `xab=input[0]`,
`xan=input[1]`, and `xor=1-input[0]-input[1]`, then clamps each to `DBL_EPSILON`.
The Gibbs value is produced by the FIRST-mask path, `0x29f548` through the output
store at `0x29f75b`. The script symbolically interprets every one of its 107
instructions, including stack saves/reloads, scalar SSE arithmetic, and three
calls to `log`. Other XMM registers are discarded after each log call to honor
the ABI's caller-clobber rule. The path reloads every quantity it needs.

Every loaded double is read from the binary and retained as its exact integer
ratio for the algebra audit. A second compact Margules expression is checked
against the lifted output symbolically, for independent `xab,xan,xor,T,P`.
No simplex substitution is used to make these identities pass. The checks prove
that the real-arithmetic interpretation of the complete output path equals
`H-T*S+(P_bar-1)*V` with the coefficient table below.

This is a declared real-arithmetic model based on the pinned instructions. It
models logarithms as the real logarithm, and its closed simplex uses continuous
`x*log(x)=0` at zero. It does not yet provide a uniform bound on SSE rounding,
libm approximation, or the native endpoint clamp. Numeric sparse coefficient
serialization also uses the provider's existing binary64 convention; the
rational coefficient audit is retained separately. The phase metadata leaves
`native_binary_uniform_error_bound` null.

## Recovered equation

Let `a=xab`, `b=xan`, `c=xor`, with `a,b,c>=0` and `a+b+c=1`. The provider uses
`v0=a`, `v1=b`, `c=1-v0-v1`. There is no order variable or pure-order reference.
The endmember order is albite, anorthite, sanidine, with native indices `[0,1,2]`.

```
H = whaban*a*b*(b+c/2) + whanab*a*b*(a+c/2)
  + whabor*a*c*(c+b/2) + whorab*a*c*(a+b/2)
  + whanor*b*c*(c+a/2) + whoran*b*c*(b+a/2)
  + whabanor*a*b*c

S = -R*(a*ln(a)+b*ln(b)+c*ln(c))
  + wsabor*a*c*(c+b/2) + wsorab*a*c*(a+b/2)

V = wvabor*a*c*(c+b/2) + wvorab*a*c*(a+b/2)
  + wvanor*b*c*(c+a/2) + wvabanor*a*b*c
```

| Coefficient | Pinned plagioclase value | MAGMA `feldspar.c` value |
| --- | ---: | ---: |
| `whaban` (J/mol) | 7924 | 7924 |
| `whanab` (J/mol) | 0 | 0 |
| `whabor` (J/mol) | -6498.5 | 18810 |
| `whorab` (J/mol) | -6498.5 | 27320 |
| `whanor` (J/mol) | 6498.5 | 38974 |
| `whoran` (J/mol) | 6498.5 | 40317 |
| `whabanor` (J/mol) | 9055.5 | 12545 |
| `wsabor`, `wsorab` (J/mol/K) | 10.3 | 10.3 |
| `wvabor` (J/mol/bar) | 0.4602 | 0.4602 |
| `wvorab` (J/mol/bar) | 0.3264 | 0.3264 |
| `wvanor` (J/mol/bar) | -0.1037 | -0.1037 |
| `wvabanor` (J/mol/bar) | -1.095 | -1.095 |
| `R` (J/mol/K) | 8.3143 | 8.3143 |

The distinguishing rodata addresses are `0x332088=-6498.5`,
`0x332090=9055.5`, and `0x332098=6498.5`. The source H topology is reused in the
binary, but these different enthalpy constants explain the observed mismatch:
the plagioclase-minus-published-feldspar difference is -7390.875 J/mol at the
Ab-Or midpoint and -8286.75 J/mol at the An-Or midpoint. It vanishes on the
Ab-An edge and at all three pure vertices. These statements follow from the
recovered algebra, not from a finite fit.

The binary strings contain
`$Id: plagioclase.c,v 1.3 2006/10/20 00:59:22 ghiorso Exp $`.
That embedded version label does not establish that the missing source file was
found. The complete binary and symbol hashes are the reproducible provenance.

## Finite compatibility

At 2173.15 K and 267.20283416157343 bar, the validation script compares the
61 archived trials and final best trial in `phase_14_plagioclase.json`.
Native extensive Gibbs energy is divided by the actual total native endmember
moles, and the pinned native pure standards are subtracted on the same basis.
The maximum absolute mixing-G difference is `1.4897523215040565e-9 J/mol`.

This finite check is independent evidence of correct wiring and normalization.
It does not establish a native binary uniform error bound, empirical calibration,
or any claim about other builds. The global polynomial/site-domain bound can
be applied to the declared equation, with native binary error remaining a
separate unresolved quantity.
