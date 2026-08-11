"""Residual Helmholtz-energy kernel contracts."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from exoeos import IdealEOS, TRhoState, psir, state_trho


GAS_CONSTANT = 8.31446261815324


class _SecondVirialEOS(NamedTuple):
    """Symmetric second-virial model used as an analytic test double."""

    coefficients: jax.Array

    def alphar(self, T, rho, x):
        """Return the reduced residual Helmholtz energy."""

        del T
        return rho * jnp.einsum("i,ij,j->", x, self.coefficients, x)


@pytest.fixture
def virial_model() -> _SecondVirialEOS:
    return _SecondVirialEOS(
        jnp.asarray([[1.0e-4, 2.0e-5], [2.0e-5, 8.0e-5]]),
    )


def test_ideal_eos_has_zero_residual_helmholtz_state() -> None:
    model = IdealEOS()
    temperature = 600.0
    molar_density = 40.0
    composition = jnp.asarray([0.25, 0.75])
    partial_densities = molar_density * composition

    state = state_trho(model, temperature, molar_density, composition)
    chemical_potentials = jax.grad(
        lambda values: psir(model, temperature, values)
    )(partial_densities)

    assert isinstance(state, TRhoState)
    assert jnp.allclose(model.alphar(temperature, molar_density, composition), 0.0)
    assert jnp.allclose(psir(model, temperature, partial_densities), 0.0)
    assert jnp.allclose(state.molar_density, molar_density)
    assert jnp.allclose(
        state.pressure,
        molar_density * GAS_CONSTANT * temperature,
    )
    assert jnp.allclose(state.compressibility_factor, 1.0)
    assert jnp.allclose(state.reduced_residual_helmholtz, 0.0)
    assert jnp.allclose(chemical_potentials, jnp.zeros(2))
    assert jnp.allclose(state.reduced_residual_chemical_potentials, jnp.zeros(2))
    assert jnp.allclose(state.log_fugacity_coefficients, jnp.zeros(2))
    assert jnp.allclose(state.reduced_residual_gibbs, 0.0)


def test_ideal_eos_is_a_pytree_and_supports_state_derivatives() -> None:
    model = IdealEOS()
    temperature = 600.0
    molar_density = 40.0
    composition = jnp.asarray([1.0, 0.0])

    compiled_pressure = jax.jit(
        lambda eos, T, rho: state_trho(eos, T, rho, composition).P
    )(model, temperature, molar_density)
    pressure_density_gradient = jax.grad(
        lambda rho: state_trho(model, temperature, rho, composition).P
    )(molar_density)
    pressure_temperature_gradient = jax.grad(
        lambda T: state_trho(model, T, molar_density, composition).P
    )(temperature)

    assert jax.tree_util.tree_leaves(model) == []
    assert jnp.allclose(
        compiled_pressure,
        molar_density * GAS_CONSTANT * temperature,
    )
    assert jnp.allclose(pressure_density_gradient, GAS_CONSTANT * temperature)
    assert jnp.allclose(pressure_temperature_gradient, GAS_CONSTANT * molar_density)
    assert all(
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree_util.tree_leaves(
            state_trho(model, temperature, molar_density, composition)
        )
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_ideal_eos_preserves_floating_dtype(dtype) -> None:
    state = state_trho(
        IdealEOS(),
        jnp.asarray(600.0, dtype=dtype),
        jnp.asarray(40.0, dtype=dtype),
        jnp.asarray([0.25, 0.75], dtype=dtype),
    )

    assert all(leaf.dtype == dtype for leaf in jax.tree_util.tree_leaves(state))


def test_state_trho_promotes_to_model_parameter_dtype(
    virial_model: _SecondVirialEOS,
) -> None:
    state = jax.jit(
        lambda model: state_trho(
            model,
            jnp.asarray(700.0, dtype=jnp.float32),
            jnp.asarray(50.0, dtype=jnp.float32),
            jnp.asarray([0.4, 0.6], dtype=jnp.float32),
        )
    )(virial_model)

    assert virial_model.coefficients.dtype == jnp.float64
    assert all(
        leaf.dtype == jnp.float64 for leaf in jax.tree_util.tree_leaves(state)
    )


def test_second_virial_state_matches_helmholtz_identities(
    virial_model: _SecondVirialEOS,
) -> None:
    temperature = 700.0
    molar_density = 50.0
    composition = jnp.asarray([0.4, 0.6])
    partial_densities = molar_density * composition
    coefficients = virial_model.coefficients

    state = state_trho(
        virial_model,
        temperature,
        molar_density,
        composition,
    )
    expected_psi = partial_densities @ coefficients @ partial_densities
    expected_alphar = expected_psi / molar_density
    expected_mu_res_RT = 2.0 * coefficients @ partial_densities
    expected_Z = 1.0 + expected_psi / molar_density
    expected_lnphi = expected_mu_res_RT - jnp.log(expected_Z)
    expected_gres_RT = composition @ expected_lnphi

    assert jnp.allclose(psir(virial_model, temperature, partial_densities), expected_psi)
    assert jnp.allclose(state.reduced_residual_helmholtz, expected_alphar)
    assert jnp.allclose(
        state.reduced_residual_chemical_potentials,
        expected_mu_res_RT,
    )
    assert jnp.allclose(state.compressibility_factor, expected_Z)
    assert jnp.allclose(
        state.pressure,
        molar_density * GAS_CONSTANT * temperature * expected_Z,
    )
    assert jnp.allclose(state.log_fugacity_coefficients, expected_lnphi)
    assert jnp.allclose(state.reduced_residual_gibbs, expected_gres_RT)
    assert jnp.allclose(
        state.reduced_residual_gibbs,
        state.reduced_residual_helmholtz
        + state.compressibility_factor
        - 1.0
        - jnp.log(state.compressibility_factor),
    )

    assert jnp.allclose(state.rho, state.molar_density)
    assert jnp.allclose(state.P, state.pressure)
    assert jnp.allclose(state.Z, state.compressibility_factor)
    assert jnp.allclose(state.alphar, state.reduced_residual_helmholtz)
    assert jnp.allclose(
        state.mu_res_RT,
        state.reduced_residual_chemical_potentials,
    )
    assert jnp.allclose(state.lnphi, state.log_fugacity_coefficients)
    assert jnp.allclose(state.gres_RT, state.reduced_residual_gibbs)


def test_partial_density_derivatives_match_second_virial_model(
    virial_model: _SecondVirialEOS,
) -> None:
    temperature = 700.0
    partial_densities = jnp.asarray([20.0, 30.0])
    energy_density = lambda values: psir(virial_model, temperature, values)

    chemical_potentials = jax.grad(energy_density)(partial_densities)
    hessian = jax.jacfwd(jax.grad(energy_density))(partial_densities)

    assert jnp.allclose(
        chemical_potentials,
        2.0 * virial_model.coefficients @ partial_densities,
    )
    assert jnp.allclose(hessian, 2.0 * virial_model.coefficients)
    assert jnp.allclose(hessian, hessian.T)


def test_state_trho_differentiates_model_parameters(
    virial_model: _SecondVirialEOS,
) -> None:
    gradient = jax.grad(
        lambda coefficients: state_trho(
            _SecondVirialEOS(coefficients),
            700.0,
            50.0,
            jnp.asarray([0.4, 0.6]),
        ).gres_RT
    )(virial_model.coefficients)

    assert gradient.shape == virial_model.coefficients.shape
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradient, gradient.T)


def test_state_trho_supports_jit_and_vmap(
    virial_model: _SecondVirialEOS,
) -> None:
    evaluate = lambda T, rho, x: state_trho(virial_model, T, rho, x)
    temperature = 700.0
    molar_density = 50.0
    composition = jnp.asarray([0.4, 0.6])

    expected = evaluate(temperature, molar_density, composition)
    compiled = jax.jit(evaluate)(temperature, molar_density, composition)
    assert all(
        jnp.allclose(actual, reference)
        for actual, reference in zip(
            jax.tree_util.tree_leaves(compiled),
            jax.tree_util.tree_leaves(expected),
        )
    )

    batched = jax.vmap(evaluate)(
        jnp.asarray([600.0, 700.0]),
        jnp.asarray([20.0, 50.0]),
        jnp.asarray([[0.3, 0.7], [0.4, 0.6]]),
    )
    assert batched.molar_density.shape == (2,)
    assert batched.pressure.shape == (2,)
    assert batched.compressibility_factor.shape == (2,)
    assert batched.reduced_residual_helmholtz.shape == (2,)
    assert batched.reduced_residual_chemical_potentials.shape == (2, 2)
    assert batched.log_fugacity_coefficients.shape == (2, 2)
    assert batched.reduced_residual_gibbs.shape == (2,)


@pytest.mark.parametrize(
    "temperature,molar_density,composition",
    [
        (jnp.ones(2), 40.0, jnp.asarray([0.25, 0.75])),
        (600.0, jnp.ones(2), jnp.asarray([0.25, 0.75])),
        (600.0, 40.0, jnp.ones((1, 2))),
        (600.0, 40.0, jnp.asarray(1.0)),
    ],
)
def test_state_trho_rejects_batched_input_shapes(
    virial_model: _SecondVirialEOS,
    temperature,
    molar_density,
    composition,
) -> None:
    with pytest.raises(ValueError):
        state_trho(virial_model, temperature, molar_density, composition)


@pytest.mark.parametrize(
    "temperature,partial_densities",
    [
        (jnp.ones(2), jnp.asarray([10.0, 30.0])),
        (600.0, jnp.ones((1, 2))),
        (600.0, jnp.asarray(40.0)),
    ],
)
def test_psir_rejects_non_single_state_input_shapes(
    virial_model: _SecondVirialEOS,
    temperature,
    partial_densities,
) -> None:
    with pytest.raises(ValueError):
        psir(virial_model, temperature, partial_densities)
