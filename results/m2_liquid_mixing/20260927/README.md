# Published liquid expression and independent native controls

`native_controls.json` preserves 34 formal property states and every attempted
native comparison, including unavailable endpoints. The native and full-test
processes both completed with exit code 0. `manifest.json` pins the code and
saved bytes. The full EOS suite passed: **460 passed in 73.69 s**.

The central input is the accepted 16-layer H=1e24 mol source at 2173.15 K and
267.20283416157343 bar. This is a property comparison at a saved source; it is
not a new pressure closure or an independently calibrated BSE phase boundary.

| Check | Result |
| --- | --- |
| Formal property states | 34 finite |
| Native composition comparisons | 13 finite; 21 unavailable |
| Maximum finite total-G difference | 7.12e-10 RT per native mole |
| Maximum finite chemical-potential difference | 2.22e-5 RT |
| Central-source potential difference | 4.18e-11 RT |
| Published Euler residual | at most 2.85e-14 RT per native mole |
| Cached callback calls | 1000 in 0.383 s, one native standard probe |

The largest potential difference occurs in the extremely water-rich binary
with dry fraction 1e-12, where subtraction near the boundary is sensitive to
roundoff. It must not be replaced by the much smaller central-source error.
The timing measures callback work only, not a complete equilibrium solve.

Every native exception or nonfinite endpoint remains in the raw record. The
declared expression supplies continuous energies at zero components, while
absent-component potentials remain unavailable. A finite native comparison
does not certify a uniform binary error bound. The release binary's complete
thermodynamic source identity and empirical calibration domain remain separate
requirements.

The reproducible command is recorded in the raw file and uses
[`run_native_comparison.py`](../../../examples/m2_liquid_mixing/run_native_comparison.py).
Provider implementation and model-selection contracts are documented in the
[example guide](../../../examples/m2_liquid_mixing/README.md).
