"""Replay the scoped Zhang-Duan (2009) source comparison, without downloading sources.

From the repository root:
    PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python \
        results/mceg/20261021_zd2009_fluid_review/reproduce.py > /tmp/zd2009-review.json

Requires the provider environment plus NumPy and SciPy. Output goes to stdout;
the archived comparison.json is never overwritten. Table 6 values below are
published model outputs, not experimental observations. All quantities copied
from the paper are identified by their table/equation and PDF page.
"""

import hashlib
import json
import platform
from pathlib import Path

import scipy
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
from exoeos import ZhangDuanEOS, state_tp
from exoeos.zhang_duan import _EOS_COEFFICIENTS, _PUBLISHED_COMPONENT_PARAMETERS

root = Path(__file__).resolve().parents[3]
assert jax.config.x64_enabled, "Enable JAX_ENABLE_X64=1 for this comparison."
R = 8.31446261815324
# Table 3 transcribed from rendered PDF p. 3; table 4 from p. 4.
a = np.array(
    [
        0.0295177298930,
        -6337.56452413,
        -275265.428882,
        0.00129128089283,
        -145.797416153,
        76593.8947237,
        0.00000258661493537,
        0.521265321460,
        -139.839523753,
        -0.0000000236335007175,
        0.00535026383543,
        -0.271106499510,
        25038.7836486,
        0.732267260410,
        0.0154833359970,
    ]
)
assert np.array_equal(a, np.asarray(_EOS_COEFFICIENTS))
for name, expected in {"H2O": (510.0, 2.88e-10), "CO2": (235.0, 3.79e-10)}.items():
    assert _PUBLISHED_COMPONENT_PARAMETERS[name] == expected


def terms(T, q, epsilon):
    """Independent NumPy transcription of Eqs. 8, 15 and 16 (PDF pp. 3-4)."""
    tm = 154 * T / epsilon
    b, c, d, e = [a[i] + a[i + 1] / tm**2 + a[i + 2] / tm**3 for i in (0, 3, 6, 9)]
    z = (
        1
        + b * q
        + c * q * q
        + d * q**4
        + e * q**5
        + a[12] / tm**3 * q * q * (a[13] + a[14] * q * q) * np.exp(-a[14] * q * q)
    )
    integral = a[13] + 1 - (a[13] + 1 + a[14] * q * q) * np.exp(-a[14] * q * q)
    s1 = (
        b * q
        + c * q * q / 2
        + d * q**4 / 4
        + e * q**5 / 5
        + a[12] / (2 * a[14] * tm**3) * integral
    )
    s2 = (
        sum(
            (2 * a[i + 1] / tm**2 + 3 * a[i + 2] / tm**3) * q**power / power
            for i, power in zip((0, 3, 6, 9), (1, 2, 4, 5))
        )
        + 3 * a[12] / (2 * a[14] * tm**3) * integral
    )
    return z, s1, s2


# Table 6, PDF p. 5: T [K], P [MPa], V_this_study [cm3/mol].
rows = [
    (1203.15, 950.0, 22.2),
    (1293.15, 1750.0, 19.0),
    (1393.15, 1750.0, 19.52),
    (1491.15, 950.0, 24.88),
    (1493.15, 1750.0, 20.04),
    (1593.15, 1750.0, 20.56),
    (1693.15, 1750.0, 21.07),
    (1723.15, 2200.0, 19.64),
    (1873.15, 2500.0, 19.41),
    (1273.15, 1450.0, 19.96),
    (1373.15, 1450.0, 20.58),
    (1473.15, 1450.0, 21.19),
    (1573.15, 1450.0, 21.81),
    (1673.15, 1450.0, 22.43),
    (983.15, 1850.0, 17.06),
    (983.15, 1400.0, 18.25),
    (983.15, 2500.0, 15.9),
    (1173.15, 3000.0, 16.0),
    (1173.15, 3500.0, 15.41),
    (1273.15, 3000.0, 16.35),
    (1273.15, 2950.0, 16.42),
    (1273.15, 2450.0, 17.23),
    (1373.15, 3000.0, 16.68),
    (1373.15, 4000.0, 15.48),
    (1373.15, 2500.0, 17.53),
    (1373.15, 3500.0, 16.02),
    (1073.15, 850.0, 21.71),
    (1073.15, 1500.0, 18.54),
    (1073.15, 2000.0, 17.23),
    (1173.15, 850.0, 22.73),
    (1173.15, 2000.0, 17.73),
]
water = ZhangDuanEOS.from_species(("H2O",))
water_state = jax.jit(lambda t, p: state_tp(water, t, p, jnp.ones(1)))
scale = (2.88 / 3.691) ** 3 / 1000
C = 3.0636 * 3.691**3 / 154
volumes = []
for row in rows:
    T, P, published = row
    state = water_state(T, P * 1e6)
    volume = 1e6 / float(state.rho)

    # With P/Pm in MPa, V in cm3/mol and Vm in L/mol, use R=0.008314...
    # MPa L/(mol K) in Pm*Vm/(R*Tm). Eqs. 9-11 then give C*Z_physical.
    # This unit-consistent reading is an explicit reviewer interpretation.
    # Independently invert the Eq. 8 RHS for each of the two conventions.
    def solve(factor):
        q = brentq(
            lambda q: q * terms(T, q, 510.0)[0] * R * T / scale / factor - P * 1e6,
            1e-12,
            128.0,
            xtol=1e-12,
        )
        return scale * 1e6 / q

    physical = solve(1.0)
    scaled = solve(C)
    assert abs(physical - volume) < 1e-9
    volumes.append(
        {
            "T_K": T,
            "P_MPa": P,
            "paper_model_cm3_mol": published,
            "provider_cm3_mol": volume,
            "scaled_lhs_cm3_mol": scaled,
            "difference_cm3_mol": volume - published,
        }
    )

binary = ZhangDuanEOS.from_species(("H2O", "CO2"))
np.testing.assert_array_equal(
    binary.energy_interaction_parameters, [[1.0, 0.85], [0.85, 1.0]]
)
np.testing.assert_array_equal(
    binary.size_interaction_parameters, [[1.0, 1.02], [1.02, 1.0]]
)
mixture_state = jax.jit(lambda x: state_tp(binary, 1573.15, 1.45e9, x))
epsilon = np.array([510.0, 235.0])
sigma = np.array([2.88, 3.79])
k1 = np.array([[1.0, 0.85], [0.85, 1.0]])
k2 = np.array([[1.0, 1.02], [1.02, 1.0]])
eij = k1 * np.sqrt(epsilon[:, None] * epsilon[None, :])
sij = k2 * (sigma[:, None] + sigma[None, :]) / 2
printed_sij = k1 * (sigma[:, None] + sigma[None, :]) / 2
mixtures = []
for xw in (0.5, 0.7, 0.9):
    x = np.array([xw, 1 - xw])
    state = mixture_state(jnp.asarray(x))
    em = x @ eij @ x
    sm = x @ sij @ x
    q = float(state.rho) / 1000 * (sm / 3.691) ** 3
    z, s1, s2 = terms(1573.15, q, em)
    base = z - 1 - np.log(z) + s1
    # Eq. 14 as printed has +2*S2 and k1 in its size sum. Differentiating
    # N*S1 at fixed T,V gives -2*S2 and k2 instead: S2=-Tm*dS1/dTm,
    # while epsilon and sigma are the quadratic mixture rules, Eqs. 12-13.
    # This is a reviewer derivation, not an author-confirmed correction.
    printed = (
        base + 2 * s2 * (1 - eij @ x / em) + 6 * (1 - z) * (1 - printed_sij @ x / sm)
    )
    derived = base - 2 * s2 * (1 - eij @ x / em) + 6 * (1 - z) * (1 - sij @ x / sm)
    actual = np.asarray(state.lnphi)
    assert np.max(abs(derived - actual)) < 1e-11
    mixtures.append(
        {
            "x_H2O": xw,
            "provider_lnphi": actual.tolist(),
            "printed_eq14_lnphi": printed.tolist(),
            "derived_lnphi": derived.tolist(),
            "max_derived_difference": float(np.max(abs(derived - actual))),
        }
    )
report = {
    "scope": "Source equations and published model outputs, not new experimental accuracy",
    "source": {
        "doi": "10.1016/j.gca.2009.01.021",
        "url": "https://efs.idsse.ac.cn/module1/citation/2009-04-01/article.pdf",
        "sha256": "6d72a0eed45e9a83887a73f01ae08d10f77fa251720662195dfb66a1f159d9d6",
        "locations": "PDF pp. 2-6: Eqs. 1-2, 7-16 and Tables 3-7; source bytes verified during review, not downloaded by this replay",
    },
    "table3_coefficients_matched": 15,
    "table4_species": ["H2O", "CO2"],
    "table7_binary_parameters": {"k1": 0.85, "k2": 1.02},
    "eq8_scaling_factor": C,
    "table6_comparisons": volumes,
    "summary": {
        "table6_rows": len(volumes),
        "physical_within_half_printed_unit": sum(
            abs(r["difference_cm3_mol"]) <= 0.005 for r in volumes
        ),
        "scaled_within_half_printed_unit": sum(
            abs(r["scaled_lhs_cm3_mol"] - r["paper_model_cm3_mol"]) <= 0.005
            for r in volumes
        ),
        "physical_max_absolute_difference_cm3_mol": max(
            abs(r["difference_cm3_mol"]) for r in volumes
        ),
    },
    "eq14_comparisons": mixtures,
}
report["eq8_units"] = {
    "physical": "P in Pa, V in m3/mol, R=8.31446261815324 J/(mol K)",
    "scaled_lhs_reading": "P and Pm in MPa, V in cm3/mol, Vm in L/mol, R=0.00831446261815324 MPa L/(mol K)",
    "relation": "Pm*Vm/(R*Tm) = (3.0636*3.691**3/154)*Z_physical under Eqs. 9-11",
    "status": "Explicit reviewer interpretation of the printed scaling, not an author-confirmed correction",
}
report["eq14_conditions"] = {
    "T_K": 1573.15,
    "P_MPa": 1450.0,
    "component_order": ["H2O", "CO2"],
    "composition_basis": "mole fraction",
    "derived_expression": "Z-1-ln(Z)+S1-2*S2*(1-E_i/epsilon_mix)+6*(1-Z)*(1-D_i/sigma_mix)",
    "definitions": "E_i=sum_j(k1_ij*x_j*sqrt(epsilon_i*epsilon_j)); D_i=sum_j(k2_ij*x_j*(sigma_i+sigma_j)/2)",
    "status": "Derivative of N*S1 at fixed T,V; printed Eq. 14 has +2*S2 and uses k1 in D_i instead",
}
report["environment"] = {
    "python": platform.python_version(),
    "jax": jax.__version__,
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "backend": jax.default_backend(),
    "x64": jax.config.x64_enabled,
}
paths = [
    "src/exoeos/zhang_duan.py",
    "src/exoeos/helmholtz.py",
    "src/exoeos/constants.py",
    "src/exoeos/state.py",
    "results/mceg/20261021_zd2009_fluid_review/reproduce.py",
]
report["file_sha256"] = {
    p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths
}
assert report["summary"]["physical_within_half_printed_unit"] == len(rows)
print(json.dumps(report, indent=2))
