"""Independent reference and transformation tests for Peng-Robinson EOS."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from exoeos import (
    PengRobinsonEOS,
    SecondVirialEOS,
    TRhoState,
    get_critical_properties,
    state_tp,
    state_trho,
)


GAS_CONSTANT = 8.31446261815324
PR_ATTRACTION_CONSTANT = 0.45723552892138218938
PR_COVOLUME_CONSTANT = 0.077796073903888455972


def _component_parameters(
    temperature,
    critical_temperatures,
    critical_pressures,
    acentric_factors,
):
    kappa = 0.37464 + 1.54226 * acentric_factors - 0.26992 * acentric_factors**2
    alpha = (1.0 + kappa * (1.0 - jnp.sqrt(temperature / critical_temperatures))) ** 2
    attraction = (
        PR_ATTRACTION_CONSTANT
        * GAS_CONSTANT**2
        * critical_temperatures**2
        * alpha
        / critical_pressures
    )
    covolume = (
        PR_COVOLUME_CONSTANT * GAS_CONSTANT * critical_temperatures / critical_pressures
    )
    return attraction, covolume


def _mixture_parameters(
    temperature,
    composition,
    critical_temperatures,
    critical_pressures,
    acentric_factors,
    binary_interaction_parameters,
):
    attraction, covolume = _component_parameters(
        temperature,
        critical_temperatures,
        critical_pressures,
        acentric_factors,
    )
    cross_attraction = jnp.sqrt(attraction[:, None] * attraction[None, :]) * (
        1.0 - binary_interaction_parameters
    )
    return (
        composition @ cross_attraction @ composition,
        composition @ covolume,
    )


def _reference_alphar(temperature, molar_density, attraction, covolume):
    sqrt_two = jnp.sqrt(jnp.asarray(2.0, dtype=molar_density.dtype))
    reduced_density = covolume * molar_density
    attraction_log = jnp.log1p((1.0 + sqrt_two) * reduced_density) - jnp.log1p(
        (1.0 - sqrt_two) * reduced_density
    )
    return -jnp.log1p(-reduced_density) - (
        attraction
        * attraction_log
        / (2.0 * sqrt_two * covolume * GAS_CONSTANT * temperature)
    )


def _reference_pressure(temperature, molar_density, attraction, covolume):
    reduced_density = covolume * molar_density
    return GAS_CONSTANT * temperature * molar_density / (
        1.0 - reduced_density
    ) - attraction * molar_density**2 / (
        1.0 + 2.0 * reduced_density - reduced_density**2
    )


@pytest.fixture
def methane_eos() -> PengRobinsonEOS:
    return PengRobinsonEOS(
        jnp.asarray([190.564]),
        jnp.asarray([4_599_200.0]),
        jnp.asarray([0.01142]),
    )


def test_mixture_helmholtz_state_matches_explicit_pr_pressure() -> None:
    critical_temperatures = jnp.asarray([190.564, 305.322])
    critical_pressures = jnp.asarray([4_599_200.0, 4_872_200.0])
    acentric_factors = jnp.asarray([0.01142, 0.0995])
    binary_interaction_parameters = jnp.asarray(
        [[0.0, 0.035], [0.035, 0.0]],
    )
    eos = PengRobinsonEOS(
        critical_temperatures,
        critical_pressures,
        acentric_factors,
        binary_interaction_parameters,
    )
    temperature = jnp.asarray(270.0)
    molar_density = jnp.asarray(3_500.0)
    composition = jnp.asarray([0.35, 0.65])
    attraction, covolume = _mixture_parameters(
        temperature,
        composition,
        critical_temperatures,
        critical_pressures,
        acentric_factors,
        binary_interaction_parameters,
    )

    state = state_trho(eos, temperature, molar_density, composition)
    expected_alphar = _reference_alphar(
        temperature,
        molar_density,
        attraction,
        covolume,
    )
    expected_pressure = _reference_pressure(
        temperature,
        molar_density,
        attraction,
        covolume,
    )

    assert jnp.allclose(
        eos.alphar(temperature, molar_density, composition),
        expected_alphar,
    )
    assert jnp.allclose(state.alphar, expected_alphar)
    assert jnp.allclose(state.P, expected_pressure)
    assert jnp.allclose(
        state.Z,
        expected_pressure / (molar_density * GAS_CONSTANT * temperature),
    )
    assert jnp.allclose(state.gres_RT, composition @ state.lnphi)


def test_cho_fixed_state_matches_frozen_teqp_reference() -> None:
    formulas = ("CO", "H2O", "CO2", "H2")
    properties = tuple(get_critical_properties(formula) for formula in formulas)
    eos = PengRobinsonEOS(
        jnp.asarray([item.critical_temperature for item in properties]),
        jnp.asarray([item.critical_pressure for item in properties]),
        jnp.asarray([item.acentric_factor for item in properties]),
        jnp.zeros((4, 4)),
    )

    # teqp v0.23.2 (5f62a6f515d517e39c3fb035c11a03524ffa3ad6): PR76
    # generalized cubic with Mathias-Copeman alpha, exact critical-condition
    # OmegaA=0.45723552892138218938, OmegaB=0.077796073903888455972,
    # R=8.31446261815324 J mol^-1 K^-1, zero kij, and the vapor root.
    state = state_tp(
        eos,
        300.0,
        4.0e6,
        jnp.asarray([0.4, 0.4, 0.1, 0.1]),
        phase="vapor",
    )

    assert jnp.allclose(
        state.rho,
        2_073.221567707672,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    assert jnp.allclose(
        state.P,
        3_999_999.9999999953,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    assert jnp.allclose(
        state.Z,
        0.7734973557808338,
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    assert jnp.allclose(
        state.lnphi,
        jnp.asarray(
            [
                0.07492890545020213,
                -0.5899275983086101,
                -0.22792636343163442,
                0.19916964221154948,
            ]
        ),
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    assert jnp.allclose(
        state.gres_RT,
        -0.2088751492653717,
        rtol=2.0e-10,
        atol=2.0e-12,
    )


def test_alpha_uses_pr76_high_acentric_factor_branch() -> None:
    eos = PengRobinsonEOS(
        jnp.asarray([500.0]),
        jnp.asarray([5.0e6]),
        jnp.asarray([0.75]),
    )
    temperature = jnp.asarray(350.0)
    molar_density = jnp.asarray(2_500.0)
    kappa_pr76 = 0.37464 + 1.54226 * 0.75 - 0.26992 * 0.75**2
    alpha_pr76 = (1.0 + kappa_pr76 * (1.0 - jnp.sqrt(350.0 / 500.0))) ** 2
    expected_attraction = (
        PR_ATTRACTION_CONSTANT
        * GAS_CONSTANT**2
        * 500.0**2
        * alpha_pr76
        / 5.0e6
    )
    expected_covolume = PR_COVOLUME_CONSTANT * GAS_CONSTANT * 500.0 / 5.0e6
    expected_alphar = _reference_alphar(
        temperature,
        molar_density,
        expected_attraction,
        expected_covolume,
    )

    assert jnp.allclose(
        eos.alphar(temperature, molar_density, jnp.asarray([1.0])),
        expected_alphar,
    )


def test_second_virial_coefficients_match_peng_robinson_low_density_limit() -> None:
    eos = PengRobinsonEOS(
        jnp.asarray([190.564, 305.322]),
        jnp.asarray([4_599_200.0, 4_872_200.0]),
        jnp.asarray([0.01142, 0.0995]),
        jnp.asarray([[0.0, 0.035], [0.035, 0.0]]),
    )
    temperature = 700.0
    composition = jnp.asarray([0.35, 0.65])
    coefficients = eos.second_virial_coefficients(temperature)
    virial_eos = SecondVirialEOS(coefficients)
    molar_density = 1.0e-3

    pr_state = state_trho(eos, temperature, molar_density, composition)
    virial_state = state_trho(
        virial_eos,
        temperature,
        molar_density,
        composition,
    )

    assert coefficients.shape == (2, 2)
    assert jnp.allclose(coefficients, coefficients.T)
    assert jnp.allclose(pr_state.Z, virial_state.Z, rtol=1.0e-12, atol=1.0e-14)
    assert jnp.allclose(
        pr_state.gres_RT,
        virial_state.gres_RT,
        rtol=1.0e-7,
        atol=1.0e-14,
    )


def test_state_tp_selects_reference_vapor_and_liquid_roots(
    methane_eos: PengRobinsonEOS,
) -> None:
    temperature = 150.0
    pressure = 1.0e6
    composition = jnp.asarray([1.0])

    vapor = state_tp(
        methane_eos,
        temperature,
        pressure,
        composition,
        phase="vapor",
    )
    liquid = state_tp(
        methane_eos,
        temperature,
        pressure,
        composition,
        phase="liquid",
    )

    assert isinstance(vapor, TRhoState)
    assert isinstance(liquid, TRhoState)
    assert jnp.allclose(vapor.Z, 0.8250427593763904, rtol=2.0e-10)
    assert jnp.allclose(liquid.Z, 0.03311547801117998, rtol=2.0e-10)
    assert jnp.allclose(vapor.rho, 971.8474481139544, rtol=2.0e-10)
    assert jnp.allclose(liquid.rho, 24_212.71708698018, rtol=2.0e-10)
    assert jnp.allclose(vapor.P, pressure, rtol=2.0e-10)
    assert jnp.allclose(liquid.P, pressure, rtol=2.0e-10)
    assert jnp.allclose(
        vapor.lnphi,
        jnp.asarray([-0.16302147255901314]),
        rtol=2.0e-10,
    )
    assert jnp.allclose(
        liquid.lnphi,
        jnp.asarray([-0.12695799083968184]),
        rtol=2.0e-10,
    )
    assert liquid.rho > vapor.rho


def test_state_tp_supports_jit_vmap_and_pressure_gradient(
    methane_eos: PengRobinsonEOS,
) -> None:
    composition = jnp.asarray([1.0])
    compiled = jax.jit(state_tp, static_argnames=("phase",))(
        methane_eos,
        300.0,
        5.0e6,
        composition,
        phase="vapor",
    )
    batched = jax.vmap(lambda T, P: state_tp(methane_eos, T, P, composition))(
        jnp.asarray([280.0, 300.0]),
        jnp.asarray([1.0e6, 5.0e6]),
    )
    density_pressure_gradient = jax.grad(
        lambda P: state_tp(methane_eos, 300.0, P, composition).rho
    )(5.0e6)
    unique_liquid = state_tp(
        methane_eos,
        300.0,
        5.0e6,
        composition,
        phase="liquid",
    )
    attraction, covolume = _component_parameters(
        jnp.asarray(300.0),
        jnp.asarray([190.564]),
        jnp.asarray([4_599_200.0]),
        jnp.asarray([0.01142]),
    )
    reduced_density = covolume[0] * compiled.rho
    pressure_density_gradient = (
        GAS_CONSTANT * 300.0 / (1.0 - reduced_density) ** 2
        - 2.0
        * attraction[0]
        * compiled.rho
        * (1.0 + reduced_density)
        / (1.0 + 2.0 * reduced_density - reduced_density**2) ** 2
    )

    assert isinstance(compiled, TRhoState)
    assert jnp.allclose(compiled.Z, 0.9018278227398956, rtol=2.0e-10)
    assert jnp.allclose(unique_liquid.rho, compiled.rho)
    assert batched.rho.shape == (2,)
    assert batched.lnphi.shape == (2, 1)
    assert jnp.all(jnp.isfinite(batched.rho))
    assert jnp.allclose(
        density_pressure_gradient,
        1.0 / pressure_density_gradient,
        rtol=2.0e-6,
    )


def test_low_pressure_float32_gradient_recovers_ideal_limit() -> None:
    dtype = jnp.float32
    eos = PengRobinsonEOS(
        jnp.asarray([190.564], dtype=dtype),
        jnp.asarray([4_599_200.0], dtype=dtype),
        jnp.asarray([0.01142], dtype=dtype),
    )
    temperature = jnp.asarray(300.0, dtype=dtype)
    pressure = jnp.asarray(10.0, dtype=dtype)
    composition = jnp.asarray([1.0], dtype=dtype)

    density_pressure_gradient = jax.jit(
        jax.grad(lambda value: eos.molar_density(temperature, value, composition))
    )(pressure)
    vapor_density = eos.molar_density(temperature, pressure, composition)
    liquid_density = eos.molar_density(
        temperature,
        pressure,
        composition,
        phase="liquid",
    )

    assert jnp.isfinite(density_pressure_gradient)
    assert jnp.allclose(liquid_density, vapor_density)
    assert jnp.allclose(
        density_pressure_gradient,
        1.0 / (GAS_CONSTANT * temperature),
        rtol=1.0e-5,
    )


@pytest.mark.parametrize(
    "pressure,expected_density",
    [
        (1.0e3, 23_948.37675939591),
        (1.0e5, 23_975.648053861143),
    ],
)
def test_low_pressure_float32_liquid_root_round_trips_pressure(
    pressure,
    expected_density,
) -> None:
    dtype = jnp.float32
    eos = PengRobinsonEOS(
        jnp.asarray([190.564], dtype=dtype),
        jnp.asarray([4_599_200.0], dtype=dtype),
        jnp.asarray([0.01142], dtype=dtype),
    )

    state = state_tp(
        eos,
        jnp.asarray(150.0, dtype=dtype),
        jnp.asarray(pressure, dtype=dtype),
        jnp.asarray([1.0], dtype=dtype),
        phase="liquid",
    )

    assert jnp.allclose(state.rho, expected_density, rtol=2.0e-6)
    assert jnp.allclose(state.P, pressure, rtol=3.0e-2, atol=3.0)
    assert jnp.all(jnp.isfinite(state.lnphi))


def test_peng_robinson_rejects_invalid_shapes_and_phase(
    methane_eos: PengRobinsonEOS,
) -> None:
    with pytest.raises(ValueError):
        PengRobinsonEOS([190.564, 305.322], [4_599_200.0], [0.01142, 0.0995])
    with pytest.raises(ValueError):
        PengRobinsonEOS(
            [190.564, 305.322],
            [4_599_200.0, 4_872_200.0],
            [0.01142, 0.0995],
            jnp.zeros((2, 3)),
        )
    with pytest.raises(ValueError):
        methane_eos.alphar(300.0, 1_000.0, jnp.asarray([0.5, 0.5]))
    with pytest.raises(ValueError, match="phase"):
        state_tp(
            methane_eos,
            300.0,
            1.0e5,
            jnp.asarray([1.0]),
            phase="solid",
        )


@pytest.mark.parametrize(
    "temperature,pressure,composition",
    [
        (jnp.ones(2), 1.0e5, jnp.asarray([1.0])),
        (300.0, jnp.ones(2), jnp.asarray([1.0])),
        (300.0, 1.0e5, jnp.ones((1, 1))),
    ],
)
def test_state_tp_rejects_batched_input_shapes(
    methane_eos: PengRobinsonEOS,
    temperature,
    pressure,
    composition,
) -> None:
    with pytest.raises(ValueError):
        state_tp(methane_eos, temperature, pressure, composition)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_peng_robinson_is_a_pytree_and_preserves_dtype(dtype) -> None:
    eos = PengRobinsonEOS(
        jnp.asarray([190.564], dtype=dtype),
        jnp.asarray([4_599_200.0], dtype=dtype),
        jnp.asarray([0.01142], dtype=dtype),
        jnp.zeros((1, 1), dtype=dtype),
    )
    state = jax.jit(
        lambda model: state_tp(
            model,
            jnp.asarray(300.0, dtype=dtype),
            jnp.asarray(5.0e6, dtype=dtype),
            jnp.asarray([1.0], dtype=dtype),
        )
    )(eos)

    leaves = jax.tree_util.tree_leaves(eos)
    assert len(leaves) == 4
    assert all(leaf.dtype == dtype for leaf in leaves)
    assert all(leaf.dtype == dtype for leaf in jax.tree_util.tree_leaves(state))


def test_state_trho_differentiates_all_peng_robinson_parameters() -> None:
    critical_temperatures = jnp.asarray([190.564, 305.322])
    critical_pressures = jnp.asarray([4_599_200.0, 4_872_200.0])
    acentric_factors = jnp.asarray([0.01142, 0.0995])
    binary_interaction_parameters = jnp.asarray(
        [[0.0, 0.035], [0.035, 0.0]],
    )
    composition = jnp.asarray([0.35, 0.65])

    def residual_gibbs(Tc, Pc, omega, kij):
        eos = PengRobinsonEOS(Tc, Pc, omega, kij)
        return state_trho(eos, 270.0, 3_500.0, composition).gres_RT

    gradients = jax.jit(jax.grad(residual_gibbs, argnums=(0, 1, 2, 3)))(
        critical_temperatures,
        critical_pressures,
        acentric_factors,
        binary_interaction_parameters,
    )

    for gradient, parameter in zip(
        gradients,
        (
            critical_temperatures,
            critical_pressures,
            acentric_factors,
            binary_interaction_parameters,
        ),
    ):
        assert gradient.shape == parameter.shape
        assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradients[-1], gradients[-1].T)
