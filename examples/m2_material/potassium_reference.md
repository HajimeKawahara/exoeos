# Explicit finite potassium-transfer sensitivity

`potassium_reference.py` adds `K` after the eighteen species in the associated
alloy. Its required `potassium_standard_offset_rt` is a dimensionless input:

```
mu0_K,metal/(RT) = mu0_K1,actual_gas/(RT) + delta_K
```

It has no empirical default or assumed confidence interval. It preserves the
actual conserved-element reference gauge and permits finite K transfer to be
solved by the equilibrium provider. At each fixed temperature the offset is
constant across all pressure evaluations. It is not a fitted partition ratio,
a frozen trace addition, or an externally supplied potassium reservoir.

For `k=x_K`, `h=1-k` and normalized host composition `z=x[:18]/h`, the excess
scalar is `h*g_ex,18(z)`. The full ideal entropy mixes all nineteen species.
Thus the total molar scalar, apart from its linear K standard, is
`h*g_18(z)+h*log(h)+k*log(k)`. At exact zero K it recovers the eighteen-species
host. Uncalibrated K cross interactions are omitted in this declared sensitivity
model; that omission is not a measured result.

The source example uses `x_K<=0.02` and preserves the other declared particle
bounds. A rigorous curvature bound follows from the perspective Hessian. For
the seventeen non-Fe host coordinates `y`:

```
d2g = (dy + z_nonFe dk)^T H18 (dy + z_nonFe dk)/h + dk^2/[k h]
kappa19 >= min(kappa18/2,
               1/[kmax(1-kmax)] - kappa18*max(||z_nonFe||^2))
```

The host bounds are widened outward by `1/(1-kmax)` before obtaining the
positive host curvature. Young's inequality and `h<=1` yield the second line;
all remaining arithmetic is outward. The implementation requires zero solute
lower bounds and `0<kmax<0.5`, and uses a direct outward eighteen-dimensional Hessian bound if the host
curvature is nonpositive. That fallback preserves a negative lower bound; it
does not establish convexity or relax the phase-selection criteria.
At 2173.15 K the selected box gives approximately 0.3443 RT/mol species.
At `k=0` the scalar uses its continuous boundary limit; a finite Hessian is not
asserted at that entropy endpoint. Box contact remains a separate model limit.

## Primary evidence and what remains uncertain

| Primary source | Conditions and information | Use here |
| --- | --- | --- |
| [Chabot and Drake (1999)](https://doi.org/10.1016/S0012-821X(99)00208-3), Table 1 run 14 | 2173.15 K, 1.5 GPa, S-free Fe; K at or below 10 mass ppm, mass partition ratio below 1.3e-4 | Relevant same-temperature non-detection; not a bound at 269 bar in Fe-H-O-P |
| [Gessmann and Wood (2002)](https://doi.org/10.1016/S0012-821X(02)00593-9) | 2.5--24 GPa, 1773--2173 K; lithophile behavior in S-free alloys and enhanced uptake in O/S-rich alloys | Composition and pressure dependence must remain explicit |
| [Bouhifd et al. (2007)](https://doi.org/10.1016/j.pepi.2006.08.005) | 5--15 GPa, 2173 K; strong alloy-composition dependence, no resolved pressure trend within that range | Does not establish a low-pressure Henry coefficient |
| [Xiong et al. (2018)](https://doi.org/10.1029/2018JB015522) | 3000--5000 K, 20--135 GPa; first-principles free energies and strong oxygen effect | Independent mechanism evidence, not the present low-pressure standard |

These data motivate continued examination of K; their remote conditions are
not silently converted into the value of `delta_K`. A finite parameter scan can
map source and pressure sensitivity from negligible to near-complete K transfer
if those states remain inside the declared box. The direct cap on K atoms or K
mass follows from its finite input budget. Such a cap, or any sampled response,
does not by itself bound coupled pressure, phase or redox errors for every
possible physical K interaction model.
