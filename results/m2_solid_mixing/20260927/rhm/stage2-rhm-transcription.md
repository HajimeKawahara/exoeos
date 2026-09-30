# Five-endmember rhombohedral oxide transcription

The independent fragment is in `stage2-rhm-transcription.py`, with its generated
provider data in `stage2-rhm-parameters.json`. Only `/tmp` files were written.
No native worker was launched.

The source is `rhomsghiorso.c`, MAGMA commit
`705a0fb315e5054d18275a580562f6121c8e458c`, from
<https://github.com/magmasource/MAGMA>. Its complete file SHA256 is
`9c06e35c78c8e522bb05d6d1933fa34806ddc927c2002980872d5fbeb00cec5e`.
The native component order is geikielite, hematite, ilmenite, pyrophanite,
corundum; the fragment retains native indices `[0,1,2,4]` and requires absent Mn.
The source identifies the solution parameters with Ghiorso and Evans (2008).
The older four-component `rhombohedral.c` was not used.

## Complete declared face

Let `(a,b,c,d,e)` be `(Fe2_A,Fe2_B,Mg_A,Mg_B,Al_A=Al_B)`.
The domain is the closed unit box with `1-a-b-c-d-e >= 0`. Then

```
r = [a+b, c+d, 0, e]
s = [a-b, c-d, 0]
x_native_retained = [c+d, 1-a-b-c-d-e, a+b, e]
```

Both surviving order variables can have either sign. Conversely, every source
state with nonnegative A/B occupancies and nonnegative endmember fractions on
the Mn-free face maps to these coordinates. The Ti occupancies are `b+d` on A
and `a+c` on B. Each A/B site group sums to one after including Al and Fe3.
No ordering variable is fixed to its disordered value or to a numerical native
minimum. Site entropy uses continuous `0*log(0)=0` limits. Native endpoint
clamping to machine epsilon is not reproduced as a mathematical boundary rule.

The source `H` is affine in pressure, even though it includes pressure directly.
The fragment stores its value at 1 bar as `H_polynomial` and its exact symbolic
pressure derivative as `V_polynomial`, so the provider uses `H+(P_bar-1)*V`.
The transformed polynomials contain 60 enthalpy terms and 54 volume terms and
have degree at most four. Coefficients are parsed as exact rational source
numbers before the existing numeric-provider serialization step.

## SRO and signed entropy coefficients

In the pinned defaults, `SROconst` and **all eight** `SRO600`, `SRO700`, ...,
`SRO1300` values are exactly `0.0730205`. The associated temperatures are
873.15 through 1573.15 K in 100 K steps. The source natural-spline construction
sets `u[0]=y2[0]=y2[7]=0`. Equal ordinates make all adjacent slopes zero, and
induction through its forward recurrence and backward substitution gives
`u[i]=y2[i]=0` for every entry. Evaluation then gives
`fSRO(T,0)=a*SRO+b*SRO=SRO`, because `a+b=1`, including the source's endpoint
extrapolation. Its first through third derivatives are zero.

This fixed-default algebra therefore requires `T>0` and `P_bar>0`, with no
finite temperature upper limit. It does not establish a calibration domain.
Runtime SRO parameter resets are outside the declaration.

Writing `f=0.0730205`, `h=1-a-b-c-d-e`, the nonconstant entropy terms in `G/(RT)`
use the following signed coefficients (before converting native R to common R):

| Occupation | Coefficient |
| --- | ---: |
| `a,b,c,d,b+d,a+c` | `1` each |
| `(a+b)/2,(c+d)/2,(a+b+c+d)/2` | `-2*f` each |
| `h,e` | `2*(1-f)` each |

The source entropy also has the constant `2*f*R*ln(2)`, which is retained in
`S_polynomial`. The negative ID coefficients are intentional and require signed
interval multiplication in the lower-bound consumer. The transcription checks
all source logarithmic coefficients symbolically before dropping the identically
zero Mn contributions.

## Pure reference subtraction

The source `gmixMsg` subtracts `sum(x_i*G_pure_i)` from the site Gibbs function.
The main fragment preserves the unreferenced site expression and supplies every
reference in `pure_reference_models` as one-dimensional H/S/V polynomials and
site entropy. A pure coordinate `u in [0,1]` represents source `s=2*u-1`.

The source geikielite, ilmenite, and pyrophanite pure functions coincide:

```
A(P) = 17477 + 0.010758*(P_bar-1)
B(P) = 3189 + 0.035089*(P_bar-1)
g(s) = A*(1-s*s) + B*s*s*(1-s*s)
       + R*T*((1+s)*ln(1+s)+(1-s)*ln(1-s)-2*ln(2))
```

For every positive pressure, `A,B>0`. Every admissible order state therefore
satisfies `-2*R*T*ln(2) <= g <= A+B/4`. These are safe all-state bounds and do not
assume native Newton convergence. The JSON also retains separate H(1 bar) and V
bounds; combining those requires signed multiplication by `(P_bar-1)`.

For hematite and corundum, the exact pure function is
`g=-2*f*R*T*ln(2)`. Their entropy constant must be included even though they have
no variable entropy sites. Existing consumers that only understand
`entropy_site_groups` would omit this constant; they must use the complete pure
models or the explicit entropy bounds.

There is a stronger analytic identity when `R*T >= A(P)`:

```
(1+s)*ln(1+s)+(1-s)*ln(1-s)
  = sum(n>=1, s^(2*n)/(n*(2*n-1))) >= s*s

g(s)-g(0) >= s*s*(R*T-A) + B*s*s*(1-s*s) >= 0.
```

Thus the global pure minimum is exactly `g(0)=A-2*R*T*ln(2)` in that explicit
domain. No sampling or numerical optimization enters this proof. This is an
identity for the declared pure model; a native binary error bound remains a
separate requirement.

At 2173.15 K and 267.20283416157343 bar, `R*T-A=588.3572349100868 J/mol`, so the
identity applies. The retained reference values are approximately
`[-7568.009140061324,-1829.0082067565181,-7568.009140061324,-1829.0082067565181]`
J/mol in the four-component retained order.

A provider choosing this high-temperature domain can subtract the references
exactly at the polynomial level. With `q=a+b+c+d` and `f=0.0730205`, replace

```
H_polynomial <- H_polynomial - 17477*q
V_polynomial <- V_polynomial - 0.010758*q
S_polynomial <- -2*(1-f)*R*ln(2)*q
```

and retain all eleven entropy sites. This produces mixing G directly and needs
no residual pure-reference bounds. The provider must check `R*T>=A(P)` before
using this representation. The unrestricted fragment does not silently apply
this domain restriction.

## Finite native compatibility

`stage2-rhm-validate.py` reuses the archived
`phase_26_rhm-oxide.json`; it does not invoke a native process. It compares all
61 stored trials and the final best trial at the pinned T/P, using their actual
native total endmember mole counts when normalizing extensive Gibbs energy.
The independent order minimization uses bounded local optimizers from three
initial order states. All 62 optimizations report success. The maximum absolute
mixing-G difference is `1.396301740896888e-9 J/mol`; a compared state has an order
parameter magnitude of `0.20179240617690233`, so the agreement includes ordered
states. Details and archive/parameter hashes are in
`stage2-rhm-validation.json`.

These finite comparisons support compatibility only. They do not prove a global
phase bound, native binary equivalence over the domain, or empirical validity.
The complete-domain algebra and pure-reference inequalities above supply the
independent mathematical ingredients for a subsequent certified bound.
