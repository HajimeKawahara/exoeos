"""Replay the Na observations and concentration conventions of Steenstra 2018.

This diagnostic adds no component or constitutive law to a source equilibrium.
"""
from pathlib import Path
import json
import math
import re


DATA = Path(__file__).with_name("sodium_reference_sources.json")
ATOMIC_MASS = dict(Fe=55.845, O=15.999, Si=28.085, Na=22.98976928,
                   Mg=24.305, Al=26.9815385, Ca=40.078, K=39.0983,
                   Mn=54.938044, Cr=51.9961, Ti=47.867, S=32.06,
                   Ni=58.6934, Pt=195.084, C=12.011, Cu=63.546,
                   Cd=112.414, In=114.818, Sn=118.710)
OXIDES = {"MgO": {"Mg": 1, "O": 1}, "SiO2": {"Si": 1, "O": 2},
          "Al2O3": {"Al": 2, "O": 3}, "CaO": {"Ca": 1, "O": 1},
          "FeO": {"Fe": 1, "O": 1}, "K2O": {"K": 2, "O": 1},
          "Na2O": {"Na": 2, "O": 1}, "MnO": {"Mn": 1, "O": 1},
          "Cr2O3": {"Cr": 2, "O": 3}, "TiO2": {"Ti": 1, "O": 2},
          "SO2": {"S": 1, "O": 2}}


def observation(text):
    """Keep a reported censoring sign distinct from a measured central value."""
    match = re.match(r"^(<?)(\d+(?:\.\d*)?)(?:\((\d+)\))?", text.strip())
    if not match:
        return dict(raw=text, value=None, reported_two_standard_errors=None,
                    censored_upper=False)
    bound, number, error = match.groups()
    decimals = len(number.partition(".")[2])
    return dict(raw=text, value=float(number),
                reported_two_standard_errors=(None if error is None else
                                              int(error) * 10. ** -decimals),
                censored_upper=bool(bound))


def _value(text):
    return observation(text)["value"] or 0.


def _column(table, header, start, stop, name):
    column = table[header].index(name)
    return {row[0].strip(): row[column] for row in table[start:stop]}


def _mass(oxide):
    return sum(ATOMIC_MASS[e] * count for e, count in OXIDES[oxide].items())


def replay(data=None):
    """Return measured mass ratios, censored bounds and two declared KD bases.

    Major host compositions are EPMA values; Na/K use LA-ICP-MS except GGK6,
    for which the paper selects EPMA. Unquantified denominator entries are
    omitted, not asserted to be measured zeros. The reported calculated C
    mole fraction closes the alloy denominator. Its separate EPMA-total C
    mass estimate is retained as an observation, not silently substituted.
    """
    data = json.loads(DATA.read_text()) if data is None else data
    silicate, metal = (data[f"supplement_TableS{x}_cells"] for x in (3, 4))
    result = []
    for number in (1, 2, 3, 4, "5b", 6, 7, 8, 9):
        name = f"GGK-{number}"
        continued = number in (6, 7, 8, 9)
        se = _column(silicate, 23 if continued else 0,
                     25 if continued else 3, 36 if continued else 14, name)
        sl = _column(silicate, 23 if continued else 0,
                     39 if continued else 17, 45 if continued else 23, name)
        me = _column(metal, 33 if continued else 0,
                     35 if continued else 2, 49 if continued else 20, name)
        ml = _column(metal, 33 if continued else 0,
                     52 if continued else 24, 61 if continued else 33, name)
        column = metal[33 if continued else 0].index(name)
        carbon_x = _value(metal[50 if continued else 22][column])
        carbon_mass_raw = metal[49 if continued else 21][column]
        published = next(row for row in data["primary_Table1_cells"]
                         if row[0] == f"GGK{number}")
        activity = next(row for row in data["supplement_TableS2_cells"]
                        if row[0] == f"GGK{number}")
        na_s = observation(sl["Na (ppm)"])
        na_m = observation(ml["Na (ppm)"])
        oxides = {o: _value(raw) for o, raw in se.items() if o in OXIDES}
        if number == 6:
            na_s = observation(se["Na2O"])
            factor = 1e4 * 2 * ATOMIC_MASS["Na"] / _mass("Na2O")
            for key in ("value", "reported_two_standard_errors"):
                na_s[key] *= factor
            na_m = observation(me["Na"])
            potassium = _value(me["K"])
        else:
            oxides["Na2O"] = na_s["value"] * 1e-4 * _mass("Na2O") / (2 * ATOMIC_MASS["Na"])
            oxides["K2O"] = _value(sl["K2O"])
            potassium = _value(ml["K"])
        moles = {}
        for key, raw in me.items():
            element = key.split()[0]
            if element not in ATOMIC_MASS:
                continue
            scale = 1. if element in ("Fe", "O", "Si", "S", "Ni", "Cu", "Cd", "In", "Sn", "Pt") else 1e-4
            moles[element] = _value(raw) * scale / ATOMIC_MASS[element]
        moles["Na"] = na_m["value"] * 1e-4 / ATOMIC_MASS["Na"]
        moles["K"] = potassium * 1e-4 / ATOMIC_MASS["K"]
        total_moles = sum(moles.values()) / (1. - carbon_x)
        x_na, x_fe = (moles[e] / total_moles for e in ("Na", "Fe"))
        oxide_moles = {o: amount / _mass(o) for o, amount in oxides.items()}
        bases = {}
        for basis in ("one_cation", "oxide_molecules"):
            total = sum(amount * (sum(v for e, v in OXIDES[o].items() if e != "O")
                                  if basis == "one_cation" else 1.)
                        for o, amount in oxide_moles.items())
            factor = 2. if basis == "one_cation" else 1.
            x_s_na = factor * oxide_moles["Na2O"] / total
            x_s_fe = oxide_moles["FeO"] / total
            kd = x_na / x_s_na * math.sqrt(x_s_fe / x_fe)
            g_na = observation(activity[6])["value"]
            log_k = (None if g_na is None else math.log10(
                kd * g_na / math.sqrt(_value(activity[4]))
                * math.sqrt(_value(activity[3]))))
            # Parse the signed published log separately from concentrations.
            signed = re.match(r"^([−-]?\d+\.\d+)\((\d+)\)", published[6])
            published_log = float(signed[1].replace("−", "-")) if signed else None
            published_error = (int(signed[2]) * 10. ** -len(signed[1].partition(".")[2])
                               if signed else None)
            bases[basis] = dict(K_D=kd, log10_K_D=math.log10(kd),
                                x_Na_silicate=x_s_na, x_FeO_silicate=x_s_fe,
                                log10_K=log_k, published_log10_K=published_log,
                                published_two_standard_errors=published_error,
                                residual_dex=(None if log_k is None or published_log is None
                                              else log_k - published_log))
        mass_ratio = na_m["value"] / na_s["value"]
        censor = na_m["censored_upper"]
        silicate_lower = na_s["value"] - na_s["reported_two_standard_errors"]
        result.append(dict(run=f"GGK{number}", temperature_K=float(published[1]),
                           pressure_GPa=1., metal_host=published[3],
                           metal_S_wt_percent=_value(me.get("S", "-")),
                           metal_Si_wt_percent=_value(me["Si"]),
                           carbon_mole_fraction_calculated=carbon_x,
                           carbon_wt_percent_from_EPMA_total=carbon_mass_raw,
                           silicate_Na_mass_ppm=na_s, metal_Na_mass_ppm=na_m,
                           selected_measurement="EPMA" if number == 6 else "LA-ICP-MS",
                           mass_partition_ratio=mass_ratio,
                           ratio_kind="reported_censoring_upper_at_central_silicate" if censor else "central",
                           mass_partition_upper_at_reported_silicate_lower=(na_m["value"] / silicate_lower if censor else None),
                           published_Na_status=published[6],
                           excluded_from_detected_fit=(censor or number == 3),
                           status_note=("Table S4 gives 48(17) ppm but Table1 says b.d.l.; preserved, not used as a detected fit point." if number == 3 else None),
                           alloy_x_Na=x_na, alloy_x_Fe=x_fe, concentration_bases=bases))
    return result


def henry_standard_rt(log_k, log_gamma_infinite, sodium_oxide_standard_rt,
                      iron_standard_rt, iron_oxide_standard_rt):
    """Convert an explicitly supplied, consistent exchange reference to Henry.

    Inputs are natural logarithms and one-Na NaO0.5 standards. The caller must
    establish their concentration/reference convention; no native/JANAF anchor
    is silently supplied by this reference diagnostic.
    """
    values = (log_k, log_gamma_infinite, sodium_oxide_standard_rt,
              iron_standard_rt, iron_oxide_standard_rt)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Reference standards and logarithms must be finite.")
    return (sodium_oxide_standard_rt + .5 * iron_standard_rt
            - .5 * iron_oxide_standard_rt - (log_k - log_gamma_infinite))
