# Published water calibration data

The CSV files retain the original bytes published by Maggie Thompson in
[Zenodo v1, 16418810](https://doi.org/10.5281/zenodo.16418810), licensed under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). They accompany
[Thompson et al. (2025)](https://doi.org/10.1016/j.chemgeo.2025.123048).
`provenance.json` records URLs, SHA256 and published MD5 values, coefficient
locations, the inspected notebook hash, selection rules, and limitations.

No source values are corrected. The author notebook's `XOH` means its
`OH_Conc_ppm` column divided by one million. The original Sossi water
columns and pooled OH columns require clarification before physical atom
conversion. The independent replay preserves this distinction; see
[the model documentation](../../../documents/m2_reaction_calibration.rst).

`Sossi2023_Table1.csv` is a separate transcription of all fourteen rows of
the original [Sossi et al. (2023) Table 1](https://arxiv.org/pdf/2211.13344),
attributed to Sossi, Tollan, Badro and Bower under the article's CC-BY license.
`sossi2023_provenance.json` pins the inspected PDF and transcribed table.
It includes Per-4 and the shared Per-5 background, absent from the later
twelve-row CSV. Raw and corrected rounded values are both retained; no
historical CSV or validation result is changed. The new
`sossi_water_calibration.py` reconstructs Eq.11 and propagates the common
background through its linear fit and prediction operators.
