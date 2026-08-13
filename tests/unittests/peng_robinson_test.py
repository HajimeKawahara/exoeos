"""Independent reference and transformation tests for Peng-Robinson EOS."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from exoeos import PengRobinsonEOS, TRhoState, state_tp, state_trho


GAS_CONSTANT = 8.31446261815324
PR_ATTRACTION_CONSTANT = 0.45724
PR_COVOLUME_CONSTANT = 0.07780


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
    assert jnp.allclose(vapor.Z, 0.8250426402060995, rtol=2.0e-10)
    assert jnp.allclose(liquid.Z, 0.0331183299264132, rtol=2.0e-10)
    assert jnp.allclose(vapor.rho, 971.847588488933, rtol=2.0e-10)
    assert jnp.allclose(liquid.rho, 24_210.6320598419, rtol=2.0e-10)
    assert jnp.allclose(vapor.P, pressure, rtol=2.0e-10)
    assert jnp.allclose(liquid.P, pressure, rtol=2.0e-10)
    assert jnp.allclose(
        vapor.lnphi,
        jnp.asarray([-0.163021888735602]),
        rtol=2.0e-10,
    )
    assert jnp.allclose(
        liquid.lnphi,
        jnp.asarray([-0.126859724193990]),
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
    assert jnp.allclose(compiled.Z, 0.901830710952881, rtol=2.0e-10)
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
        (1.0, 23_945.952706161217),
        (1.0e3, 23_946.22925603225),
        (1.0e5, 23_973.507321528363),
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
