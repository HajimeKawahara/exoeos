"""Independent reference and transformation tests for Zhang-Duan 2009."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from exoeos import TRhoState, ZhangDuanEOS, state_tp, state_trho


GAS_CONSTANT = 8.31446261815324
REFERENCE_DIAMETER = 3.691e-10
COEFFICIENTS = jnp.asarray(
    [
        2.95177298930e-2,
        -6.33756452413e3,
        -2.75265428882e5,
        1.29128089283e-3,
        -1.45797416153e2,
        7.65938947237e4,
        2.58661493537e-6,
        5.21265321460e-1,
        -1.39839523753e2,
        -2.36335007175e-8,
        5.35026383543e-3,
        -2.71106499510e-1,
        2.50387836486e4,
        7.32267260410e-1,
        1.54833359970e-2,
    ]
)


def _reference_compressibility(
    temperature,
    molar_density,
    composition,
    energies,
    diameters,
    energy_interactions,
    size_interactions,
):
    pair_energies = energy_interactions * jnp.sqrt(
        energies[:, None] * energies[None, :]
    )
    pair_diameters = size_interactions * 0.5 * (
        diameters[:, None] + diameters[None, :]
    )
    mixture_energy = composition @ pair_energies @ composition
    mixture_diameter = composition @ pair_diameters @ composition
    reduced_temperature = 154.0 * temperature / mixture_energy
    reduced_density = (
        molar_density
        / 1000.0
        * (mixture_diameter / REFERENCE_DIAMETER) ** 3
    )

    def parameter(index):
        return (
            COEFFICIENTS[index]
            + COEFFICIENTS[index + 1] / reduced_temperature**2
            + COEFFICIENTS[index + 2] / reduced_temperature**3
        )

    b, c, d, e = (parameter(index) for index in (0, 3, 6, 9))
    squared_density = reduced_density**2
    return (
        1.0
        + b * reduced_density
        + c * squared_density
        + d * reduced_density**4
        + e * reduced_density**5
        + COEFFICIENTS[12]
        / reduced_temperature**3
        * squared_density
        * (COEFFICIENTS[13] + COEFFICIENTS[14] * squared_density)
        * jnp.exp(-COEFFICIENTS[14] * squared_density)
    )


@pytest.fixture
def water_eos() -> ZhangDuanEOS:
    return ZhangDuanEOS.from_species(("H2O",))


@pytest.mark.parametrize(
    "temperature,pressure,expected",
    [
        (
            1203.15,
            950.0e6,
            (45038.0776664, 2.10857862322, 0.118536482599, 0.481101023611),
        ),
        (
            1873.15,
            2500.0e6,
            (51517.4469783, 3.11586716164, 0.879145739790, 1.85850540538),
        ),
        (
            1373.15,
            3500.0e6,
            (62410.6651631, 4.91198539466, 1.07776757606, 3.39807475328),
        ),
    ],
)
def test_water_matches_independent_zd09_references(
    water_eos: ZhangDuanEOS,
    temperature,
    pressure,
    expected,
) -> None:
    # Volumes agree with Zhang and Duan (2009), Table 6, before its rounding,
    # and with Atmodeller commit 77e81aac5147bad7eec727fba5b299e055db2d4d.
    state = state_tp(water_eos, temperature, pressure, jnp.asarray([1.0]))
    expected_density, expected_z, expected_alphar, expected_lnphi = expected

    assert isinstance(state, TRhoState)
    assert jnp.allclose(state.rho, expected_density, rtol=1.0e-8, atol=0.0)
    assert jnp.allclose(state.P, pressure, rtol=1.0e-10, atol=0.0)
    assert jnp.allclose(state.Z, expected_z, rtol=1.0e-9, atol=1.0e-11)
    assert jnp.allclose(
        state.alphar,
        expected_alphar,
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    assert jnp.allclose(
        state.lnphi,
        expected_lnphi,
        rtol=1.0e-9,
        atol=1.0e-11,
    )


def test_h2o_co2_mixture_matches_independent_reference() -> None:
    # These high-precision values come from an independent amount derivative
    # of n*alphar. They intentionally do not transcribe Equation 14: its
    # printed energy term has the opposite sign from Tm=154*T/epsilon and its
    # size term uses k1 where the Equation 13 mixing rule requires k2.
    eos = ZhangDuanEOS.from_species(("H2O", "CO2"))
    composition = jnp.asarray([0.5, 0.5])
    state = state_tp(eos, 1573.15, 1450.0e6, composition)

    assert jnp.allclose(state.rho, 33153.0695179, rtol=1.0e-8, atol=0.0)
    assert jnp.allclose(
        1.0e6 / state.rho,
        30.1631195705603,
        rtol=1.0e-8,
        atol=0.0,
    )
    assert jnp.allclose(
        state.Z,
        3.34379726534063,
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    assert jnp.allclose(
        state.alphar,
        1.10707806686274,
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    assert jnp.allclose(
        state.lnphi,
        jnp.asarray([1.08459102044986, 3.40294550997438]),
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    assert jnp.allclose(
        state.gres_RT,
        2.24376826521212,
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    assert jnp.allclose(
        state.gres_RT,
        composition @ state.lnphi,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert jnp.allclose(
        state.gres_RT,
        state.alphar + state.Z - 1.0 - jnp.log(state.Z),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    "species,composition,temperature,molar_density",
    [
        (("H2O", "CO2"), [0.35, 0.65], 1573.15, 30000.0),
        (("H2O", "CH4"), [0.7, 0.3], 1200.0, 35000.0),
    ],
)
def test_mixture_chemical_potentials_match_amount_finite_differences(
    species,
    composition,
    temperature,
    molar_density,
) -> None:
    eos = ZhangDuanEOS.from_species(species)
    amounts = jnp.asarray(composition)
    volume = jnp.sum(amounts) / molar_density
    state = state_trho(eos, temperature, molar_density, amounts)

    def extensive_alphar(values):
        total = jnp.sum(values)
        return total * eos.alphar(
            temperature,
            total / volume,
            values / total,
        )

    step = 1.0e-5
    basis = jnp.eye(len(species))
    finite_difference_mu = jax.vmap(
        lambda direction: (
            extensive_alphar(amounts + step * direction)
            - extensive_alphar(amounts - step * direction)
        )
        / (2.0 * step)
    )(basis)

    assert jnp.allclose(
        state.mu_res_RT,
        finite_difference_mu,
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_alphar_density_derivative_recovers_explicit_zd09_equation() -> None:
    eos = ZhangDuanEOS.from_species(("CO", "H2O", "CO2", "H2"))
    temperature = 1000.0
    molar_density = 12000.0
    composition = jnp.asarray([0.4, 0.4, 0.1, 0.1])
    expected_z = _reference_compressibility(
        temperature,
        molar_density,
        composition,
        eos.epsilon_over_k,
        eos.molecular_diameters,
        eos.energy_interaction_parameters,
        eos.size_interaction_parameters,
    )
    alphar_density_derivative = jax.grad(
        lambda rho: eos.alphar(temperature, rho, composition)
    )(molar_density)
    state = state_trho(eos, temperature, molar_density, composition)

    assert jnp.allclose(
        1.0 + molar_density * alphar_density_derivative,
        expected_z,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert jnp.allclose(state.Z, expected_z, rtol=1.0e-10, atol=1.0e-12)
    assert jnp.allclose(
        state.P,
        molar_density * GAS_CONSTANT * temperature * expected_z,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_published_species_factory_inserts_table_parameters() -> None:
    species = ("CH4", "H2O", "CO2", "H2", "CO", "O2", "C2H6")
    eos = ZhangDuanEOS.from_species(species)

    assert jnp.allclose(
        eos.epsilon_over_k,
        jnp.asarray([154.0, 510.0, 235.0, 31.2, 105.6, 124.5, 246.1]),
    )
    assert jnp.allclose(
        eos.molecular_diameters / 1.0e-10,
        jnp.asarray([3.691, 2.88, 3.79, 2.93, 3.66, 3.36, 4.35]),
    )
    assert eos.energy_interaction_parameters[0, 1] == 0.8
    assert eos.energy_interaction_parameters[1, 2] == 0.85
    assert eos.size_interaction_parameters[0, 1] == 1.0
    assert eos.size_interaction_parameters[1, 2] == 1.02
    assert jnp.allclose(
        eos.energy_interaction_parameters,
        eos.energy_interaction_parameters.T,
    )
    assert jnp.allclose(
        eos.size_interaction_parameters,
        eos.size_interaction_parameters.T,
    )
    assert jnp.allclose(jnp.diag(eos.energy_interaction_parameters), 1.0)
    assert jnp.allclose(jnp.diag(eos.size_interaction_parameters), 1.0)


def test_low_density_limit_is_ideal(water_eos: ZhangDuanEOS) -> None:
    state = state_trho(water_eos, 1200.0, 1.0e-8, jnp.asarray([1.0]))

    assert jnp.allclose(state.alphar, 0.0, rtol=0.0, atol=1.0e-12)
    assert jnp.allclose(state.Z - 1.0, 0.0, rtol=0.0, atol=1.0e-12)
    assert jnp.allclose(state.lnphi, 0.0, rtol=0.0, atol=1.0e-12)


def test_state_tp_selects_a_mechanically_stable_root(
    water_eos: ZhangDuanEOS,
) -> None:
    temperature = 1873.15
    pressure = 2500.0e6
    composition = jnp.asarray([1.0])
    state = state_tp(water_eos, temperature, pressure, composition)
    pressure_density_gradient = jax.grad(
        lambda rho: state_trho(water_eos, temperature, rho, composition).P
    )(state.rho)

    assert jnp.allclose(
        1.0e6 / state.rho,
        19.4108997757749,
        rtol=1.0e-8,
        atol=0.0,
    )
    assert pressure_density_gradient > 0.0


def test_state_tp_supports_jit_vmap_and_pressure_gradient() -> None:
    eos = ZhangDuanEOS.from_species(("H2O", "CO2"))
    composition = jnp.asarray([0.5, 0.5])
    compiled = jax.jit(state_tp, static_argnames=("phase",))(
        eos,
        1573.15,
        1450.0e6,
        composition,
        phase="vapor",
    )
    batched = jax.vmap(lambda T, P: state_tp(eos, T, P, composition))(
        jnp.asarray([1200.0, 1573.15]),
        jnp.asarray([500.0e6, 1450.0e6]),
    )
    density_pressure_gradient = jax.grad(
        lambda P: state_tp(eos, 1573.15, P, composition).rho
    )(1450.0e6)
    pressure_density_gradient = jax.grad(
        lambda rho: state_trho(eos, 1573.15, rho, composition).P
    )(compiled.rho)

    assert isinstance(compiled, TRhoState)
    assert jnp.allclose(compiled.P, 1450.0e6, rtol=1.0e-10, atol=0.0)
    assert batched.rho.shape == (2,)
    assert batched.lnphi.shape == (2, 2)
    assert jnp.all(jnp.isfinite(batched.rho))
    assert jnp.allclose(
        density_pressure_gradient,
        1.0 / pressure_density_gradient,
        rtol=2.0e-8,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "temperature,pressure,composition",
    [
        (jnp.ones(2), 1.0e9, jnp.asarray([1.0])),
        (1200.0, jnp.ones(2), jnp.asarray([1.0])),
        (1200.0, 1.0e9, jnp.ones((1, 1))),
    ],
)
def test_state_tp_rejects_batched_input_shapes(
    water_eos: ZhangDuanEOS,
    temperature,
    pressure,
    composition,
) -> None:
    with pytest.raises(ValueError):
        state_tp(water_eos, temperature, pressure, composition)


def test_zhang_duan_rejects_invalid_shapes_species_and_phase(
    water_eos: ZhangDuanEOS,
) -> None:
    with pytest.raises(ValueError):
        ZhangDuanEOS([510.0, 235.0], [2.88e-10])
    with pytest.raises(ValueError):
        ZhangDuanEOS(
            [510.0, 235.0],
            [2.88e-10, 3.79e-10],
            jnp.ones((2, 3)),
        )
    with pytest.raises(ValueError):
        ZhangDuanEOS(
            [510.0, 235.0],
            [2.88e-10, 3.79e-10],
            size_interaction_parameters=jnp.ones((2, 3)),
        )
    with pytest.raises(ValueError):
        ZhangDuanEOS.from_species(())
    with pytest.raises(KeyError):
        ZhangDuanEOS.from_species(("N2",))
    with pytest.raises(ValueError):
        water_eos.alphar(1200.0, 1000.0, jnp.asarray([0.5, 0.5]))
    with pytest.raises(ValueError, match="phase"):
        state_tp(
            water_eos,
            1200.0,
            1.0e9,
            jnp.asarray([1.0]),
            phase="liquid",
        )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_zhang_duan_is_a_pytree_and_preserves_dtype(dtype) -> None:
    eos = ZhangDuanEOS(
        jnp.asarray([510.0], dtype=dtype),
        jnp.asarray([2.88e-10], dtype=dtype),
        jnp.ones((1, 1), dtype=dtype),
        jnp.ones((1, 1), dtype=dtype),
    )
    state = jax.jit(
        lambda model: state_tp(
            model,
            jnp.asarray(1203.15, dtype=dtype),
            jnp.asarray(950.0e6, dtype=dtype),
            jnp.asarray([1.0], dtype=dtype),
        )
    )(eos)

    leaves = jax.tree_util.tree_leaves(eos)
    assert len(leaves) == 4
    assert all(leaf.dtype == dtype for leaf in leaves)
    assert all(leaf.dtype == dtype for leaf in jax.tree_util.tree_leaves(state))
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(state))


def test_float32_pressure_round_trip_at_high_pressure() -> None:
    dtype = jnp.float32
    eos = ZhangDuanEOS(
        jnp.asarray([510.0, 235.0], dtype=dtype),
        jnp.asarray([2.88e-10, 3.79e-10], dtype=dtype),
        jnp.asarray([[1.0, 0.85], [0.85, 1.0]], dtype=dtype),
        jnp.asarray([[1.0, 1.02], [1.02, 1.0]], dtype=dtype),
    )
    target_pressure = jnp.asarray(10.0e9, dtype=dtype)
    state = jax.jit(state_tp)(
        eos,
        jnp.asarray(673.0, dtype=dtype),
        target_pressure,
        jnp.asarray([0.5, 0.5], dtype=dtype),
    )

    assert jnp.allclose(state.P, target_pressure, rtol=2.0e-5, atol=0.0)


def test_state_tp_differentiates_all_zhang_duan_parameters() -> None:
    energies = jnp.asarray([510.0, 235.0])
    diameters = jnp.asarray([2.88e-10, 3.79e-10])
    energy_interactions = jnp.asarray([[1.0, 0.85], [0.85, 1.0]])
    size_interactions = jnp.asarray([[1.0, 1.02], [1.02, 1.0]])
    composition = jnp.asarray([0.5, 0.5])

    def residual_gibbs(epsilon, sigma, k1, k2):
        eos = ZhangDuanEOS(epsilon, sigma, k1, k2)
        return state_tp(eos, 1573.15, 1450.0e6, composition).gres_RT

    gradients = jax.jit(jax.grad(residual_gibbs, argnums=(0, 1, 2, 3)))(
        energies,
        diameters,
        energy_interactions,
        size_interactions,
    )

    for gradient, parameter in zip(
        gradients,
        (energies, diameters, energy_interactions, size_interactions),
    ):
        assert gradient.shape == parameter.shape
        assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradients[2], gradients[2].T)
    assert jnp.allclose(gradients[3], gradients[3].T)
