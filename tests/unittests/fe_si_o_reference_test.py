"""Scientific reference checks before implementation of the native JAX model."""

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


REFERENCE_DIRECTORY = Path(__file__).resolve().parents[1] / "reference"
REFERENCE = json.loads((REFERENCE_DIRECTORY / "fe_si_o_ma2001.json").read_text())
STATES = {state["id"]: state for state in REFERENCE["states"]}
INTERIORS = [state for name, state in STATES.items() if name.startswith("ternary_")]


def _coefficients(temperature):
    p = {key: float(value) for key, value in REFERENCE["parameters"].items()}
    scale = p["reference_temperature_K"] / temperature
    return (
        p["epsilon_SiSi_at_reference_temperature"] * scale,
        p["epsilon_OO_times_temperature_K"] / temperature,
        p["epsilon_SiO_at_reference_temperature"] * scale,
        p["ln_gamma0_Si_at_reference_temperature"] * scale,
        p["ln_gamma0_O_constant"] + p["ln_gamma0_O_inverse_temperature_K"] / temperature,
    )


def _total_gex(temperature, amounts, convention):
    """Integrate the exchange potentials independently of the fixture generator."""
    total = np.sum(amounts)
    _, x, y = amounts / total
    a, b, c, ls, lo = _coefficients(temperature)
    pair = -x * y - x * math.log1p(-y) - y * math.log1p(-x)
    pair += x**2 * y**2 * (1 / (1 - x) + 1 / (1 - y) - 1) / 2
    excess = a * (1 - x) * math.log1p(-x) + b * (1 - y) * math.log1p(-y) + c * pair
    if convention == "source":
        excess += x * (ls + a) + y * (lo + b)
    return total * excess


def test_reference_pins_the_printed_young_coefficients():
    assert REFERENCE["components"] == ["Fe", "Si", "O"]
    assert REFERENCE["parameters"] == {
        "reference_temperature_K": "1873",
        "epsilon_SiSi_at_reference_temperature": "12.41",
        "epsilon_SiO_at_reference_temperature": "-5",
        "epsilon_OO_times_temperature_K": "-16500",
        "ln_gamma0_Si_at_reference_temperature": "-6.65",
        "ln_gamma0_O_constant": "4.29",
        "ln_gamma0_O_inverse_temperature_K": "-16500",
    }
    for state in INTERIORS:
        difference = np.subtract(state["young_printed_solute_corrections"], state["gce_solute_corrections"])
        expected = (16500 - 1873) / state["T_K"] * math.log1p(-state["x"][2])
        np.testing.assert_allclose(difference, [0, expected], rtol=5e-12, atol=5e-12)


@pytest.mark.parametrize("state", INTERIORS, ids=lambda state: state["id"])
@pytest.mark.parametrize("convention", ["source", "formal"])
def test_amount_derivatives_and_euler_match_published_activities(state, convention):
    composition = np.asarray(state["x"])
    temperature = state["T_K"]
    expected = np.asarray(state[f"ln_gamma_{convention}"])
    energy = _total_gex(temperature, composition, convention)
    np.testing.assert_allclose(energy, state[f"gex_{convention}_RT"], rtol=5e-12, atol=5e-12)
    np.testing.assert_allclose(energy, composition @ expected, rtol=5e-12, atol=5e-12)
    # Unequal, unnormalized amounts exercise the extensive construction.
    amounts = 2.3 * composition
    step = 1e-4
    derivatives = []
    for direction in step * np.eye(3):
        values = [_total_gex(temperature, amounts + k * direction, convention) for k in (-2, -1, 1, 2)]
        derivatives.append((values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step))
    np.testing.assert_allclose(derivatives, expected, rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("state", INTERIORS, ids=lambda state: state["id"])
def test_standard_conversion_preserves_chemical_potentials_and_reaction(state):
    # Synthetic standard potentials and ideal silicate controls, all divided by RT.
    # Species order: Fe(metal), Si(metal), O(metal), FeO(silicate), SiO2(silicate).
    mu0 = np.asarray([-2.0, 3.0, 4.0, 5.0, -7.0])
    log_x = np.log(state["x"] + [0.2, 0.8])
    source = np.asarray(state["ln_gamma_source"] + [0.0, 0.0])
    formal = np.asarray(state["ln_gamma_formal"] + [0.0, 0.0])
    shift = np.asarray(state["mu0_formal_minus_source_over_RT"] + [0.0, 0.0])
    a, b, _, ls, lo = _coefficients(state["T_K"])
    np.testing.assert_allclose(shift, [0, ls + a, lo + b, 0, 0], rtol=5e-12, atol=5e-12)
    source_mu = mu0 + log_x + source
    formal_mu = mu0 + shift + log_x + formal
    np.testing.assert_allclose(formal_mu, source_mu, rtol=5e-12, atol=5e-12)
    # Balanced exchange: 2 FeO + Si -> SiO2 + 2 Fe.
    reaction = np.asarray([2, -1, 0, -2, 1])
    np.testing.assert_allclose(reaction @ formal_mu, reaction @ source_mu, rtol=5e-12, atol=5e-12)
    assert abs(reaction @ shift) > 1  # This conversion is consequential for the reaction.


def test_pure_fe_and_dilute_limits():
    pure = STATES["pure_fe_1873"]
    a, b, c, ls, lo = _coefficients(pure["T_K"])
    np.testing.assert_allclose(pure["ln_gamma_source"], [0, ls, lo], rtol=5e-12, atol=5e-12)
    np.testing.assert_allclose(pure["ln_gamma_formal"], [0, -a, -b], rtol=5e-12, atol=5e-12)
    for convention in ("source", "formal"):
        assert _total_gex(pure["T_K"], np.asarray(pure["x"]), convention) == 0
    dilute = STATES["dilute_1873"]
    _, x, y = dilute["x"]
    first_order = [0, ls + a * x + c * y, lo + b * y + c * x]
    np.testing.assert_allclose(dilute["ln_gamma_source"], first_order, rtol=0, atol=5e-14)


@pytest.mark.parametrize("name,resident", [("fe_si_3000", 1), ("fe_o_3000", 2)])
def test_binary_resident_and_absent_solute_limits(name, resident):
    state = STATES[name]
    a, b, c, ls, lo = _coefficients(state["T_K"])
    eps, initial = [0, a, b], [0, ls, lo]
    absent = 3 - resident
    z = state["x"][resident]
    fe = eps[resident] * (z + math.log1p(-z))
    expected = np.zeros(3)
    expected[0] = fe
    expected[resident] = initial[resident] + eps[resident] * z
    expected[absent] = fe + initial[absent] - c * math.log1p(-z)
    np.testing.assert_allclose(state["ln_gamma_source"], expected, rtol=5e-12, atol=5e-12)


def test_reference_generator_reproduces_committed_values():
    subprocess.run(
        [sys.executable, str(REFERENCE_DIRECTORY / "generate_fe_si_o_ma2001.py"), "--check"],
        check=True, capture_output=True, text=True,
    )
