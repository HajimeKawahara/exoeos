# Declared mineral site expressions

`../melts_solid_mixing.py` supplies property expressions and their complete
physical site domains for independent chemical stability calculations. It does
not select phases. The expressions are transcribed from
[MAGMA revision 705a0fb](https://github.com/magmasource/MAGMA/tree/705a0fb315e5054d18275a580562f6121c8e458c/).
Every source file SHA256 and the numeric parameter file SHA256 accompany the
runtime declaration. SymPy is only needed to regenerate the committed file.

The current 15 models are hornblende, biotite, garnet, alkali-feldspar,
kalsilite, leucite, alloy-solid, alloy-liquid, cummingtonite, olivine,
nepheline, clinoamphibole, orthoamphibole, melilite, and ortho-oxide.
Olivine covers the entire Fe/Mg/Ca face conditional on absent Mn/Ni/Co.
The native Fe/Ni alloys are different models from the source Fe/Si/O/H alloy.
Plagioclase is deliberately not aliased to alkali-feldspar: native mixture
energies disagree with that expression. The other untranscribed native
solutions are clinopyroxene, orthopyroxene, spinel, and rhm-oxide.

`solid_mixing_parameters(phase, T_K, P_Pa)` returns sparse enthalpy, entropy,
and volume polynomials combined at the requested state, site entropy terms,
native endmember maps, and explicit physical constraints. Signed endmember
coordinates are admitted wherever the site domain requires them. Ordering
coordinates remain free: a chemical consumer must bound all of them, rather
than trusting a particular local native ordering root.

At an empty site the energy uses the continuous `0 log 0 = 0` limit; it does
not invent an endpoint chemical potential. The nepheline/kalsilite source
contains a positive `sqrt(DBL_EPSILON)/r1` barrier. Its zero boundary diverges
to positive infinity and is retained as such. Melilite and ortho-oxide
subtract pure-reference ordering energies. Their declarations contain
conservative analytic intervals enclosing every admissible pure ordering
state, including the source's reference offsets. A state evaluator reports
the unreferenced energy when this subtraction remains interval valued.

`melts_liquid_evaluator.evaluate_liquid(...,
candidate_standard_states=[phase, ...])` separately obtains composition-
independent native pure standards. A positive native probe avoids undefined
endpoint activities; its requested and converted coordinates remain in the
receipt. Those coordinates are never substituted for a candidate composition.
The native molar oxide ledger is checked against canonical half-integer
stoichiometry to `1e-8 mol` when transcribing and when consuming a declaration.

Formal bounds on these declared expressions do not supply a uniform error
bound for an opaque native binary, an empirical calibration domain, or a
stability certificate for omitted phases. Native comparison and uncertainty
must be reported separately.

Regenerate with Python, NumPy, and SymPy 1.14.0:

```sh
python transcribe_parameters.py --source-directory /path/to/pinned/MAGMA/sources \
  --native-standards-directory /path/to/saved/native/standards \
  --host-properties /path/to/host_properties.json --output parameters.json
```

The transcription requires saved pure-standard receipts with a native molar
oxide matrix and a host receipt specifying the shared oxide molar masses.
These inputs determine stoichiometry only; T/P-dependent native energy values
are not embedded in the parameter file.
