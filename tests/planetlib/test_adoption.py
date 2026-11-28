"""Optional checks against the pinned, materialized Planetlib definitions."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# The ordinary provider suite does not require the optional authoring tools.
# The dedicated adoption CI installs the pinned backend and runs these checks.
sp = pytest.importorskip("sympy")
pytest.importorskip("mceg.schema_v06")
from mceg import load_contract, mathematical_model_digest, to_sympy
from mceg.check import check_contract

from exoeos import IdealSolution
from exoeos.mass_fraction_solute import mass_fraction_solute_state
from exoeos.solution_gibbs import total_solution_state
from exoeos.thermodynamics import thermodynamic_state_trho


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "mceg/planetlib/contract.yaml"
CONTRACT = load_contract(CONTRACT_PATH)
EQUATIONS = {equation.id: to_sympy(equation.sympy) for equation in CONTRACT.equations}


def rhs(family, name):
    return EQUATIONS[f"planetlib.{family}.{name}"].rhs


def evaluate(expression, values):
    return float(expression.subs({symbol: values[str(symbol)] for symbol in expression.free_symbols}))


def test_native_model_identity_and_provider_checks():
    lock = json.loads((ROOT / "mceg/planetlib/imports.lock.json").read_text())
    assert len(CONTRACT.mathematical_models) == len(lock["imports"]) == 4
    for entry in lock["imports"]:
        pin = entry["model"]
        assert mathematical_model_digest(CONTRACT, pin["id"], pin["revision"]) == pin["sha256"]
    assert len(CONTRACT.equations) == 16
    assert all(a.id.startswith("exoeos.") and a.status == "unknown" for a in CONTRACT.assessments)
    for assessment in CONTRACT.assessments:
        checks = check_contract(CONTRACT_PATH, {}, assessment.id, root=ROOT)["checks"]
        assert not [check for check in checks if check["status"] == "FAIL"], checks
        assert next(c for c in checks if c["name"] == "selected_files")["status"] == "PASS"
        assert next(c for c in checks if c["name"] == "recorded_support")["status"] == "UNKNOWN"


@pytest.mark.parametrize("amounts,standards", [([0.3, 0.7], [-2.1, 4.2]), ([1.1, 0.2], [3.7, -0.6])])
def test_binary_ideal_solution(amounts, standards):
    state = total_solution_state(IdealSolution(), 1600.0, 2e9, jnp.array(amounts), jnp.array(standards))
    values = dict(n1=amounts[0], n2=amounts[1], u01=standards[0], u02=standards[1])
    family = "thermodynamics.ideal_mixing"
    np.testing.assert_allclose(state.gibbs_RT, evaluate(rhs(family, "gibbs"), values), rtol=2e-13)
    np.testing.assert_allclose(state.mu_RT, [evaluate(rhs(family, f"potential_{i}"), values)
                                          for i in (1, 2)], rtol=2e-13)


@pytest.mark.parametrize("solute", [1e-4, 0.2, 2.0])
def test_two_host_mass_fraction_increment(solute):
    host = total_solution_state(IdealSolution(), 1600.0, 2e9,
                                jnp.array([0.3, 0.7]), jnp.array([-2.1, 4.2]))
    state = mass_fraction_solute_state(host.gibbs_RT, host.mu_RT, jnp.array([0.3, 0.7]),
                                      jnp.array([0.06, 0.14]), solute, 0.002, -3.0)
    values = dict(n1=0.3, n2=0.7, M1=0.06, M2=0.14, Ms=0.002, ns=solute, u0=-3.0)
    family = "thermodynamics.mass_fraction_solute"
    values["S"] = evaluate(rhs(family, "equivalent_host"), values)
    values["w"] = evaluate(rhs(family, "mass_fraction"), values)
    np.testing.assert_allclose(state.gibbs_RT - host.gibbs_RT,
                               evaluate(rhs(family, "gibbs_increment"), values), rtol=2e-11)
    np.testing.assert_allclose(state.solute_mu_RT, evaluate(rhs(family, "solute_potential"), values), rtol=2e-13)
    np.testing.assert_allclose(state.host_mu_RT - host.mu_RT,
                               [evaluate(rhs(family, f"host_response_{i}"), values)
                                for i in (1, 2)], rtol=2e-9, atol=1e-15)


@pytest.mark.parametrize("temperature,density", [(300.0, 20.0), (1700.0, 200.0)])
def test_molar_helmholtz_response(temperature, density):
    # An independent smooth total potential exercises mixed derivatives and
    # nonideal density dependence; this is not a second definition registry.
    class ToyPotential:
        molar_masses = jnp.array([0.002, 0.004])

        def molar_helmholtz(self, T, rho, x):
            del x
            return 8.3*T*(jnp.log(rho)-1) - 20*T*jnp.log(T) + 0.003*T*rho + 0.001*rho**2

    state = thermodynamic_state_trho(ToyPotential(), temperature, density, jnp.array([0.4, 0.6]))
    T, rho = EQUATIONS["planetlib.thermodynamics.helmholtz.pressure"].lhs.args
    a = sp.Function("a")(T, rho)
    potential = sp.Rational(83, 10)*T*(sp.log(rho)-1) - 20*T*sp.log(T) + sp.Rational(3, 1000)*T*rho + rho**2/1000
    family = "thermodynamics.helmholtz"
    pressure = rhs(family, "pressure").subs(a, potential).doit()
    cv = rhs(family, "heat_capacity_cv").subs(a, potential).doit()
    fields = dict(pressure=state.pressure, entropy=state.molar_entropy,
                  heat_capacity_cv=state.molar_heat_capacity_cv,
                  heat_capacity_cp=state.molar_heat_capacity_cp, sound_speed=state.sound_speed_squared)
    for name, actual in fields.items():
        expected = rhs(family, name).subs({a: potential, sp.Function("P")(T, rho): pressure,
                                          sp.Function("cv")(T, rho): cv}).doit()
        value = evaluate(expected, dict(T=temperature, rho=density, M=0.0032))
        np.testing.assert_allclose(actual, value, rtol=2e-13)


@pytest.mark.parametrize("pressure_gpa,fugacity_bar,water", [(0.7, 100.0, 0.0), (1.2, 500.0, 0.05), (3.0, 1000.0, 0.1)])
def test_hirschmann_endmember_equilibrium(pressure_gpa, fugacity_bar, water):
    from exoeos import magma_volatiles
    from exoeos._conditional_data import R

    expression = rhs("solubility.hirschmann2012", "endmember_regression")
    concentration = fugacity_bar * np.exp(evaluate(expression, {"P": pressure_gpa}))
    dry = jnp.full(11, 0.05)
    host = float(dry.sum()) + water
    hydrogen = host * concentration / (1.0 - concentration)
    standards = jnp.array([0.2, -1.3, 0.7])
    amounts = jnp.concatenate((dry, jnp.array([water, hydrogen, 0.0])))
    T = 1700.0

    def energy(values):
        return magma_volatiles.gibbs_j(T, pressure_gpa*1e9, values, standards,
                                       jnp.full(11, 0.1), h2_offset_rt=0.0)/(R*T)

    # Only the H2 derivative is compared; this does not assert that the full
    # Gibbs scalar implements an oxide-basis fit or a calibrated wet-host law.
    insertion = jax.grad(energy)(amounts)[12]
    np.testing.assert_allclose(insertion, standards[1]+np.log(fugacity_bar), rtol=2e-12)
