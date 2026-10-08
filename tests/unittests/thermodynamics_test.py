"""Independent caloric references and Helmholtz response identities."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from exoeos import (
    HelmholtzThermodynamicState,
    HelmholtzThermodynamics,
    IdealEOS,
    IdealGas,
    PengRobinsonEOS,
    ZhangDuanEOS,
    state_tp,
    thermodynamic_state_trho,
)


R = 8.31446261815324
AVOGADRO = 6.02214076e23


class _ThermalVirial(NamedTuple):
    """Analytic residual with nonzero TT, T-rho, and rho-rho derivatives."""

    coefficients: jax.Array

    def alphar(self, T, rho, x):
        del x
        b, c, d, third_virial = self.coefficients
        return rho * (b + c / T + d / T**2) + third_virial * rho**2


class _ClosedIsentropeEOS(NamedTuple):
    """Total molar Helmholtz model with an independently soluble isentrope."""

    molar_masses: jax.Array
    heat_capacity_cv: jax.Array
    thermal_coefficient: jax.Array
    attraction: jax.Array

    def molar_helmholtz(self, T, rho, x):
        del x
        return (
            R * T * jnp.log(rho / 1000.0)
            - self.heat_capacity_cv * T * jnp.log(T / 300.0)
            + self.thermal_coefficient * T * rho
            - self.attraction * rho
        )


@pytest.fixture
def ideal_reference() -> IdealGas:
    return IdealGas(
        jnp.asarray([2.0e-3, 28.0e-3]),
        jnp.asarray([28.0, 32.0]),
        reference_enthalpies=jnp.asarray([100.0, 400.0]),
        reference_entropies=jnp.asarray([10.0, 25.0]),
        reference_temperature=300.0,
        reference_pressure=1.0e5,
    )


@pytest.fixture
def thermal_model(ideal_reference) -> HelmholtzThermodynamics:
    residual = _ThermalVirial(jnp.asarray([2.0e-4, -0.1, 32.0, 2.0e-8]))
    return HelmholtzThermodynamics(residual, ideal_reference)


def test_ideal_reference_matches_analytic_mixture_and_existing_backend(
    ideal_reference,
) -> None:
    model = HelmholtzThermodynamics(IdealEOS(), ideal_reference)
    T, P = 600.0, 2.0e5
    x = jnp.asarray([0.25, 0.75])
    state = model.state(T, P, x)
    legacy = ideal_reference.state(T, P, x)
    rho = P / (R * T)
    mass = 0.25 * 2.0e-3 + 0.75 * 28.0e-3
    cp = 31.0
    h = 325.0 + cp * (T - 300.0)
    s = (
        21.25
        + cp * jnp.log(T / 300.0)
        - R * jnp.log(P / 1.0e5)
        - R * jnp.sum(x * jnp.log(x))
    )
    expected = {
        "temperature": T,
        "molar_density": rho,
        "mean_molar_mass": mass,
        "molar_helmholtz": h - R * T - T * s,
        "molar_internal_energy": h - R * T,
        "molar_enthalpy": h,
        "molar_entropy": s,
        "molar_gibbs": h - T * s,
        "molar_heat_capacity_cp": cp,
        "molar_heat_capacity_cv": cp - R,
        "compressibility_factor": 1.0,
        "mass_density": rho * mass,
        "number_density": rho * AVOGADRO,
        "pressure_temperature_derivative": rho * R,
        "pressure_density_derivative": R * T,
        "sound_speed_squared": cp / (cp - R) * R * T / mass,
        "adiabatic_gradient": R / cp,
        "isothermal_compressibility": 1.0 / P,
        "isentropic_compressibility": (cp - R) / (cp * P),
        "thermal_expansion": 1.0 / T,
    }
    assert isinstance(state, HelmholtzThermodynamicState)
    for name, reference in expected.items():
        assert jnp.allclose(getattr(state, name), reference, rtol=2.0e-12), name
    for name in ("molar_enthalpy", "molar_entropy", "molar_heat_capacity_cp"):
        assert jnp.allclose(getattr(state, name), getattr(legacy, name))
    direct = thermodynamic_state_trho(ideal_reference, T, rho, x)
    assert jnp.allclose(jnp.asarray(direct), jnp.asarray(state))


def test_temperature_dependent_nonideal_state_matches_analytic_derivatives(
    thermal_model,
) -> None:
    T, rho = 450.0, 1800.0
    x = jnp.asarray([0.25, 0.75])
    state = thermal_model.state_trho(T, rho, x)
    b, c, d, C = 2.0e-4, -0.1, 32.0, 2.0e-8
    B = b + c / T + d / T**2
    B_T = -c / T**2 - 2.0 * d / T**3
    B_TT = 2.0 * c / T**3 + 6.0 * d / T**4
    ideal_h = 325.0 + 31.0 * (T - 300.0)
    ideal_s = (
        21.25
        + 31.0 * jnp.log(T / 300.0)
        - R * jnp.log(rho * R * T / 1.0e5)
        - R * jnp.sum(x * jnp.log(x))
    )
    P = R * T * rho * (1.0 + B * rho + 2.0 * C * rho**2)
    P_T = R * rho * (1.0 + (B + T * B_T) * rho + 2.0 * C * rho**2)
    P_rho = R * T * (1.0 + 2.0 * B * rho + 6.0 * C * rho**2)
    cv = 31.0 - R - R * rho * (2.0 * T * B_T + T**2 * B_TT)
    cp = cv + T * P_T**2 / (rho**2 * P_rho)
    expected = {
        "pressure": P,
        "pressure_temperature_derivative": P_T,
        "pressure_density_derivative": P_rho,
        "molar_entropy": ideal_s - R * ((B + T * B_T) * rho + C * rho**2),
        "molar_enthalpy": ideal_h + R * T * ((B - T * B_T) * rho + 2 * C * rho**2),
        "molar_heat_capacity_cv": cv,
        "molar_heat_capacity_cp": cp,
        "thermal_expansion": P_T / (rho * P_rho),
        "isothermal_compressibility": 1.0 / (rho * P_rho),
        "adiabatic_gradient": P * P_T / (rho**2 * P_rho * cp),
        "sound_speed_squared": (P_rho + T * P_T**2 / (rho**2 * cv)) / 0.0215,
    }
    for name, reference in expected.items():
        assert jnp.allclose(getattr(state, name), reference, rtol=2.0e-12), name
    assert jnp.allclose(state.h, state.u + state.P / state.rho)
    assert jnp.allclose(state.a, state.u - T * state.s)
    assert jnp.allclose(state.g, state.h - T * state.s)
    assert jnp.allclose(
        state.cp / state.cv,
        state.isothermal_compressibility / state.isentropic_compressibility,
    )
    assert jnp.allclose(state.sound_speed**2, state.sound_speed_squared)
    assert jnp.allclose(state.Z, state.P / (rho * R * T))
    assert jnp.allclose(state.nabla_ad, state.adiabatic_gradient)


def test_sound_speed_and_gradient_follow_independent_constant_entropy_path() -> None:
    model = _ClosedIsentropeEOS(
        jnp.asarray([0.028]), jnp.asarray(21.0), jnp.asarray(0.003), jnp.asarray(0.25)
    )
    T, rho, step = 650.0, 1200.0, 0.012
    state = thermodynamic_state_trho(model, T, rho, jnp.asarray([1.0]))

    def path(density):
        temperature = T * jnp.exp(
            (R * jnp.log(density / rho) + 0.003 * (density - rho)) / 21.0
        )
        pressure = R * temperature * density + (0.003 * temperature - 0.25) * density**2
        return temperature, pressure

    T_plus, P_plus = path(rho + step)
    T_minus, P_minus = path(rho - step)
    speed_squared = (P_plus - P_minus) / (2.0 * step * 0.028)
    nabla = jnp.log(T_plus / T_minus) / jnp.log(P_plus / P_minus)
    assert jnp.allclose(state.sound_speed_squared, speed_squared, rtol=2.0e-9)
    assert jnp.allclose(state.nabla_ad, nabla, rtol=2.0e-9)
    assert jnp.allclose(state.cv, 21.0, rtol=2.0e-12)


def test_maxwell_relation_and_caloric_derivatives(thermal_model) -> None:
    T, rho = 450.0, 1800.0
    x = jnp.asarray([0.25, 0.75])

    def evaluate(temperature, density):
        return thermal_model.state_trho(temperature, density, x)

    state = evaluate(T, rho)
    entropy_rho = jax.grad(lambda density: evaluate(T, density).s)(rho)
    entropy_T = jax.grad(lambda temperature: evaluate(temperature, rho).s)(T)
    h_T = jax.grad(lambda temperature: evaluate(temperature, rho).h)(T)
    h_rho = jax.grad(lambda density: evaluate(T, density).h)(rho)
    drho_dT_P = (
        -state.pressure_temperature_derivative / state.pressure_density_derivative
    )
    assert jnp.allclose(entropy_rho, -state.pressure_temperature_derivative / rho**2)
    assert jnp.allclose(T * entropy_T, state.cv)
    assert jnp.allclose(h_T + h_rho * drho_dT_P, state.cp)


def test_reference_energy_and_entropy_shifts_leave_responses_unchanged(
    thermal_model,
    ideal_reference,
) -> None:
    x = jnp.asarray([0.25, 0.75])
    enthalpy_shift = jnp.asarray([1200.0, -600.0])
    entropy_shift = jnp.asarray([3.0, -5.0])
    shifted_ideal = IdealGas(
        ideal_reference.molar_masses,
        ideal_reference.molar_heat_capacities,
        reference_enthalpies=ideal_reference.reference_enthalpies + enthalpy_shift,
        reference_entropies=ideal_reference.reference_entropies + entropy_shift,
        reference_temperature=ideal_reference.reference_temperature,
        reference_pressure=ideal_reference.reference_pressure,
    )
    shifted_model = HelmholtzThermodynamics(thermal_model.residual, shifted_ideal)
    original = thermal_model.state_trho(450.0, 1800.0, x)
    shifted = shifted_model.state_trho(450.0, 1800.0, x)
    assert jnp.allclose(shifted.h - original.h, x @ enthalpy_shift)
    assert jnp.allclose(shifted.s - original.s, x @ entropy_shift)
    assert jnp.allclose(
        shifted.a - original.a, x @ enthalpy_shift - 450.0 * (x @ entropy_shift)
    )
    for name in (
        "pressure",
        "molar_heat_capacity_cv",
        "molar_heat_capacity_cp",
        "sound_speed",
        "adiabatic_gradient",
        "thermal_expansion",
        "isothermal_compressibility",
    ):
        assert jnp.allclose(getattr(shifted, name), getattr(original, name)), name


def test_wrapper_supports_jit_vmap_and_differentiable_model_parameters(
    thermal_model,
    ideal_reference,
) -> None:
    x = jnp.asarray([0.25, 0.75])
    expected = thermal_model.state_trho(450.0, 1800.0, x)
    compiled = jax.jit(lambda model, T, rho: model.state_trho(T, rho, x))(
        thermal_model, 450.0, 1800.0
    )
    assert jnp.allclose(jnp.asarray(compiled), jnp.asarray(expected))
    batched = jax.jit(jax.vmap(thermal_model.state_trho))(
        jnp.asarray([450.0, 600.0]),
        jnp.asarray([1800.0, 1200.0]),
        jnp.asarray([[0.25, 0.75], [0.5, 0.5]]),
    )
    assert batched.cp.shape == (2,)
    assert jnp.all(jnp.isfinite(batched.sound_speed))
    gradient = jax.grad(
        lambda coefficients: HelmholtzThermodynamics(
            _ThermalVirial(coefficients), ideal_reference
        )
        .state_trho(450.0, 1800.0, x)
        .cv
    )(jnp.asarray([2.0e-4, -0.1, 32.0, 2.0e-8]))
    assert jnp.allclose(
        gradient,
        jnp.asarray([0.0, 0.0, -2.0 * R * 1800.0 / 450.0**2, 0.0]),
        atol=1.0e-9,
    )
    cp_gradient = jax.grad(
        lambda capacities: HelmholtzThermodynamics(
            IdealEOS(), IdealGas(jnp.asarray([0.002, 0.028]), capacities)
        )
        .state_trho(450.0, 1800.0, x)
        .cv
    )(jnp.asarray([28.0, 32.0]))
    assert jnp.allclose(cp_gradient, x)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_explicit_model_dtype_and_zero_fraction_are_preserved(dtype) -> None:
    ideal = IdealGas(
        jnp.asarray([0.002, 0.028], dtype=dtype),
        jnp.asarray([28.0, 32.0], dtype=dtype),
    )
    model = HelmholtzThermodynamics(IdealEOS(), ideal)
    state = jax.jit(model.state_trho)(
        jnp.asarray(600.0, dtype=dtype),
        jnp.asarray(100.0, dtype=dtype),
        jnp.asarray([1.0, 0.0], dtype=dtype),
    )
    assert all(leaf.dtype == dtype for leaf in jax.tree_util.tree_leaves(state))
    assert all(jnp.isfinite(leaf) for leaf in jax.tree_util.tree_leaves(state))
    assert jnp.allclose(state.cp, 28.0, rtol=2.0e-6)
    assert jnp.isfinite(state.sound_speed)
    assert jnp.allclose(state.thermal_expansion, 1.0 / 600.0, rtol=2.0e-6)


def test_model_parameter_dtype_promotes_float32_inputs(thermal_model) -> None:
    state = jax.jit(
        lambda model: model.state_trho(
            jnp.asarray(450.0, dtype=jnp.float32),
            jnp.asarray(1800.0, dtype=jnp.float32),
            jnp.asarray([0.25, 0.75], dtype=jnp.float32),
        )
    )(thermal_model)
    assert all(leaf.dtype == jnp.float64 for leaf in jax.tree_util.tree_leaves(state))


@pytest.mark.parametrize("phase", ["vapor", "liquid"])
def test_peng_robinson_caloric_state_preserves_both_density_roots(phase) -> None:
    residual = PengRobinsonEOS([190.564], [4_599_200.0], [0.01142])
    model = HelmholtzThermodynamics(residual, IdealGas([0.01604], [35.0]))
    x = jnp.asarray([1.0])
    state = model.state_tp(150.0, 1.0e6, x, phase=phase)
    reference = state_tp(residual, 150.0, 1.0e6, x, phase=phase)
    direct = model.state_trho(150.0, reference.rho, x)
    assert jnp.allclose(state.rho, reference.rho, rtol=2.0e-12)
    assert jnp.allclose(state.P, 1.0e6, rtol=2.0e-10)
    assert jnp.allclose(state.Z, reference.Z, rtol=2.0e-12)
    assert jnp.allclose(jnp.asarray(state), jnp.asarray(direct), rtol=2.0e-12)
    assert state.cv > 0.0
    assert state.cp > state.cv
    assert jnp.isfinite(state.sound_speed)


def test_tp_entropy_derivative_matches_cp_for_real_eos() -> None:
    residual = PengRobinsonEOS([190.564], [4_599_200.0], [0.01142])
    model = HelmholtzThermodynamics(residual, IdealGas([0.01604], [35.0]))
    T, P = 270.0, 2.0e6
    x = jnp.asarray([1.0])
    state = model.state(T, P, x)
    ds_dT = jax.grad(lambda temperature: model.state(temperature, P, x).s)(T)
    drho_dP = jax.grad(lambda pressure: model.state(T, pressure, x).rho)(P)
    assert jnp.allclose(T * ds_dT, state.cp, rtol=2.0e-10)
    assert jnp.allclose(
        drho_dP / state.rho, state.isothermal_compressibility, rtol=2.0e-10
    )


def test_zhang_duan_uses_existing_pressure_inversion() -> None:
    residual = ZhangDuanEOS.from_species(("H2O",))
    model = HelmholtzThermodynamics(residual, IdealGas([0.01801528], [45.0]))
    state = model.state_tp(1203.15, 950.0e6, jnp.asarray([1.0]))
    assert jnp.allclose(state.rho, 45038.0776664, rtol=1.0e-8)
    assert jnp.allclose(state.P, 950.0e6, rtol=2.0e-10)
    assert jnp.allclose(state.Z, 2.10857862322, rtol=1.0e-9)
    assert state.cp > state.cv > 0.0
    assert jnp.isfinite(state.sound_speed)


@pytest.mark.parametrize(
    "temperature,density,composition",
    [
        (jnp.ones(2), 100.0, jnp.asarray([0.5, 0.5])),
        (600.0, jnp.ones(2), jnp.asarray([0.5, 0.5])),
        (600.0, 100.0, jnp.ones((1, 2))),
        (600.0, 100.0, jnp.asarray(1.0)),
        (600.0, 100.0, jnp.asarray([])),
        (600.0, 100.0, jnp.ones(3)),
    ],
)
def test_total_state_rejects_incompatible_shapes(
    ideal_reference,
    temperature,
    density,
    composition,
) -> None:
    with pytest.raises(ValueError):
        thermodynamic_state_trho(ideal_reference, temperature, density, composition)


def test_total_state_rejects_vector_helmholtz_output() -> None:
    class VectorEnergy(NamedTuple):
        molar_masses: jax.Array

        def molar_helmholtz(self, T, rho, x):
            return jnp.asarray([T, rho])

    with pytest.raises(ValueError, match="scalar"):
        thermodynamic_state_trho(
            VectorEnergy(jnp.asarray([0.028])), 300.0, 1000.0, jnp.asarray([1.0])
        )


def test_unstable_response_is_not_clipped() -> None:
    model = _ClosedIsentropeEOS(
        jnp.asarray([0.028]), jnp.asarray(21.0), jnp.asarray(0.0), jnp.asarray(5.0)
    )
    state = thermodynamic_state_trho(model, 300.0, 1000.0, jnp.asarray([1.0]))
    assert state.pressure_density_derivative < 0.0
    assert state.isothermal_compressibility < 0.0
    assert state.sound_speed_squared < 0.0
    assert jnp.isnan(state.sound_speed)
