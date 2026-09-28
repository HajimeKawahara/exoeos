"""Conditional Na Henry constraints from declared projections of 2018 hosts.

These projections are not measurements or an uncertainty distribution. They
make the existing nonnegative MELTS basis evaluable without modifying it.
"""
import math

import numpy as np

import sodium_reference as reference


def measured_oxide_masses(data, row):
    """Return grams per original 100 g analysis on the selected analytical basis."""
    name = row["run"].replace("GGK", "GGK-")
    continued = row["run"] in ("GGK6", "GGK7", "GGK8", "GGK9")
    table = data["supplement_TableS3_cells"]
    epma = reference._column(table, 23 if continued else 0,
                             25 if continued else 3, 36 if continued else 14, name)
    laser = reference._column(table, 23 if continued else 0,
                              39 if continued else 17, 45 if continued else 23, name)
    masses = {oxide.lower(): reference._value(raw)
              for oxide, raw in epma.items() if oxide in reference.OXIDES}
    if row["run"] != "GGK6":
        masses["na2o"] = (row["silicate_Na_mass_ppm"]["value"] * 1e-4
                          * reference._mass("Na2O") / (2 * reference.ATOMIC_MASS["Na"]))
        masses["k2o"] = reference._value(laser["K2O"])
    return masses


def projected_host(native, oxide_masses, *, method, potassium_aluminum_ratio):
    """Preserve Na/Fe absolute amounts while declaring every changed oxide.

    ``remove_potassium`` lowers K only. ``add_aluminum_silica`` raises Al and,
    if needed, silica to leave 0.05 mol of free SiO2 per original 100 g analysis.
    Both explicitly remove unsupported SO2. Neither normalizes the mass total.
    """
    ratio = float(potassium_aluminum_ratio)
    if not 0 < ratio < 1 or method not in {"remove_potassium", "add_aluminum_silica"}:
        raise ValueError("Select a projection method and a finite K/Al ratio in (0, 1).")
    unknown = set(oxide_masses) - set(native.OXIDES) - {"so2"}
    if unknown or any(not math.isfinite(v) or v < 0 for v in oxide_masses.values()):
        raise ValueError("Oxide masses must be finite, nonnegative, and explicitly supported.")
    masses = dict(zip(native.OXIDES, map(float, native.OXIDE_MASSES)))
    q = np.array([oxide_masses.get(o, 0.) / masses[o] for o in native.OXIDES])
    original = q.copy()
    al, k, si = (native.OXIDES.index(o) for o in ("al2o3", "k2o", "sio2"))
    if method == "remove_potassium":
        q[k] = min(q[k], ratio * q[al])
    else:
        q[al] = max(q[al], q[k] / ratio)
        trial = np.linalg.solve(native.NU.T, q)
        q[si] += max(0., .05 - trial[native.COMPONENTS.index("sio2")])
    n = np.linalg.solve(native.NU.T, q)
    native.validate_request(1800., 1e9, n, native.COMMON_R)
    changes = {o: float((b-a)*masses[o]) for o, a, b in zip(native.OXIDES, original, q) if a != b}
    if oxide_masses.get("so2", 0.):
        changes["so2"] = -oxide_masses["so2"]
    original_components = np.linalg.solve(native.NU.T, original)
    return dict(method=method, declared_K_to_Al_atomic_ratio=ratio,
                original_oxide_masses_g=dict(oxide_masses),
                original_total_oxide_mass_g=sum(oxide_masses.values()),
                projected_total_oxide_mass_g=float(q @ native.OXIDE_MASSES),
                changed_oxide_masses_g=changes,
                projected_oxide_moles=q.tolist(), component_moles=n.tolist(),
                original_negative_component_moles={name: float(v) for name, v in
                    zip(native.COMPONENTS, original_components) if v < 0},
                preserved_Na_mol=float(2*q[native.OXIDES.index("na2o")]),
                preserved_Fe_mol=float(q[native.OXIDES.index("feo")]
                                       + 2*q[native.OXIDES.index("fe2o3")]),
                measured_host_represented=False, empirical_uncertainty_bound=False)


def virtual_reference_rt(component_order, potentials):
    """Balanced oxide contribution to R_Na; add half the Fe-metal standard.

    R_Na = 1/2 mu0_Na2SiO3 - 1/4 mu0_SiO2 - 1/4 mu0_Fe2SiO4
           + 1/2 mu0_Fe(metal). Its net elemental formula is Na.
    """
    indices = [component_order.index(x) for x in ("na2sio3", "sio2", "fe2sio4")]
    values = [float(potentials[i]) for i in indices]
    if not all(map(math.isfinite, values)):
        raise ValueError("The three virtual-oxide potentials must be finite.")
    return .5*values[0] - .25*values[1] - .25*values[2]


def henry_offset_constraint(properties, *, alloy_x_na_upper, alloy_x_fe, gamma_fe):
    """Return delta lower limit for mu0_Na,H = R_Na + delta.

    The declared S-free continuation adopts gamma_Na,H = gamma_Fe. Fe and Na
    mole fractions retain the original alloy denominator including calculated
    carbon. Native oxide activities come from actual projected-host potentials;
    no molecular-oxide concentration is substituted for a one-Na potential.
    """
    values = (alloy_x_na_upper, alloy_x_fe, gamma_fe)
    if not all(math.isfinite(v) and v > 0 for v in values) or max(values[:2]) > 1:
        raise ValueError("Positive finite alloy mole fractions and Fe activity are required.")
    order = properties["component_order"]
    oxide_actual = virtual_reference_rt(order, properties["mu_RT"])
    oxide_standard = virtual_reference_rt(order, properties["mu0_RT"])
    mixing = oxide_actual - oxide_standard
    return dict(virtual_oxide_actual_rt=oxide_actual,
                virtual_oxide_standard_rt=oxide_standard,
                virtual_oxide_mixing_rt=mixing,
                delta_lower_rt=(mixing + .5*math.log(alloy_x_fe)
                                - .5*math.log(gamma_fe) - math.log(alloy_x_na_upper)),
                alloy_x_na_upper=alloy_x_na_upper, alloy_x_fe=alloy_x_fe,
                gamma_fe=gamma_fe, gamma_na_henry_policy="equal_to_gamma_fe")


def lower_envelope(rows, *, inverse_temperature_coefficient_K=0.):
    """Smallest A with delta(T)=A+B/T satisfying all supplied one-sided rows.

    The finite A choice saturates one constraint; censoring alone supplies no
    upper limit on A. B is caller-declared, not fitted from censored data.
    """
    coefficient = float(inverse_temperature_coefficient_K)
    if not rows or not math.isfinite(coefficient):
        raise ValueError("Nonempty constraints and a finite declared slope are required.")
    transformed = []
    for row in rows:
        t, lower = float(row["temperature_K"]), float(row["delta_lower_rt"])
        if not math.isfinite(t) or t <= 0 or not math.isfinite(lower):
            raise ValueError("Temperature and constraint must be finite, with T positive.")
        transformed.append(lower-coefficient/t)
    intercept = max(transformed)
    return dict(intercept_A_rt=intercept, inverse_temperature_B_K=coefficient,
                active_runs=[row["run"] for row, v in zip(rows, transformed) if v == intercept],
                allowed_intercept_interval=[intercept, None],
                finite_intercept_selection="smallest_admissible_conditional_value",
                constraints=[dict(run=row["run"], temperature_K=row["temperature_K"],
                    delta_lower_rt=row["delta_lower_rt"],
                    declared_delta_rt=intercept+coefficient/row["temperature_K"],
                    predicted_to_reported_Na_detection_limit=math.exp(v-intercept))
                    for row, v in zip(rows, transformed)])
