"""Mass-density closure contracts."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from exoeos import (
    IdealEOS,
    additive_volume_mass_density,
    mass_density_tp,
)


GAS_CONSTANT = 8.31446261815324


class _PhaseEOS(NamedTuple):
    vapor_molar_density: jax.Array
    liquid_molar_density: jax.Array

    def alphar(self, T, rho, x):
        del T, x
        return jnp.zeros_like(rho)

    def molar_density(self, T, P, x, phase="vapor"):
        del T, P, x
        if phase == "vapor":
            return self.vapor_molar_density
        if phase == "liquid":
            return self.liquid_molar_density
        raise ValueError("unsupported phase")


def test_mass_density_tp_matches_molar_density_times_mixture_molar_mass() -> None:
    temperature = 600.0
    pressure = 2.0e5
    mole_fractions = jnp.asarray([0.25, 0.75])
    molar_masses = jnp.asarray([2.0e-3, 28.0e-3])

    density = mass_density_tp(
        IdealEOS(),
        temperature,
        pressure,
        mole_fractions,
        molar_masses,
    )
    mean_molar_mass = jnp.sum(mole_fractions * molar_masses)

    assert jnp.allclose(
        density,
        pressure * mean_molar_mass / (GAS_CONSTANT * temperature),
    )


def test_mass_density_tp_forwards_phase_selection() -> None:
    eos = _PhaseEOS(jnp.asarray(10.0), jnp.asarray(1000.0))
    composition = jnp.asarray([1.0])
    molar_masses = jnp.asarray([18.0e-3])

    vapor = mass_density_tp(eos, 300.0, 1.0e5, composition, molar_masses)
    liquid = mass_density_tp(
        eos,
        300.0,
        1.0e5,
        composition,
        molar_masses,
        phase="liquid",
    )

    assert jnp.allclose(vapor, 0.18)
    assert jnp.allclose(liquid, 18.0)


def test_mass_density_tp_promotes_state_inputs_to_molar_mass_dtype() -> None:
    dtype = jnp.float64
    temperature = jnp.asarray(600.0, dtype=jnp.float32)
    pressure = jnp.asarray(2.0e5, dtype=jnp.float32)
    composition = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
    molar_masses = jnp.asarray([2.0e-3, 28.0e-3], dtype=dtype)

    density = mass_density_tp(
        IdealEOS(),
        temperature,
        pressure,
        composition,
        molar_masses,
    )
    expected = pressure.astype(dtype) * jnp.sum(
        composition.astype(dtype) * molar_masses
    ) / (GAS_CONSTANT * temperature.astype(dtype))

    assert density.dtype == dtype
    assert jnp.allclose(density, expected, rtol=1.0e-14)


def test_additive_volume_mass_density_matches_specific_volume_sum() -> None:
    mass_fractions = jnp.asarray([0.25, 0.75])
    component_densities = jnp.asarray([1000.0, 500.0])

    density = additive_volume_mass_density(mass_fractions, component_densities)

    assert jnp.allclose(
        density,
        1.0 / jnp.sum(mass_fractions / component_densities),
    )


def test_density_helpers_support_jax_transformations() -> None:
    composition = jnp.asarray([0.25, 0.75])
    molar_masses = jnp.asarray([2.0e-3, 28.0e-3])
    pressure = 2.0e5

    evaluate = jax.jit(
        lambda value: mass_density_tp(
            IdealEOS(),
            600.0,
            value,
            composition,
            molar_masses,
        )
    )
    density = evaluate(pressure)
    batched = jax.vmap(evaluate)(jnp.asarray([1.0e5, 2.0e5]))
    pressure_gradient = jax.grad(evaluate)(pressure)
    additive_gradient = jax.grad(
        lambda values: additive_volume_mass_density(
            jnp.asarray([0.25, 0.75]),
            values,
        )
    )(jnp.asarray([1000.0, 500.0]))

    assert jnp.allclose(pressure_gradient, density / pressure)
    assert batched.shape == (2,)
    assert jnp.allclose(batched[1], 2.0 * batched[0])
    assert jnp.all(jnp.isfinite(additive_gradient))


@pytest.mark.parametrize(
    "mole_fractions,molar_masses",
    [
        (jnp.asarray(1.0), jnp.asarray([18.0e-3])),
        (jnp.asarray([1.0]), jnp.asarray([18.0e-3, 44.0e-3])),
    ],
)
def test_mass_density_tp_rejects_invalid_component_shapes(
    mole_fractions,
    molar_masses,
) -> None:
    with pytest.raises(ValueError):
        mass_density_tp(
            IdealEOS(),
            300.0,
            1.0e5,
            mole_fractions,
            molar_masses,
        )


def test_mass_density_tp_rejects_batched_state_inputs() -> None:
    with pytest.raises(ValueError, match="temperature must be a scalar"):
        mass_density_tp(
            IdealEOS(),
            jnp.asarray([300.0, 400.0]),
            1.0e5,
            jnp.asarray([1.0]),
            jnp.asarray([18.0e-3]),
        )


def test_additive_volume_mass_density_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        additive_volume_mass_density(
            jnp.asarray([0.5, 0.5]),
            jnp.asarray([1000.0]),
        )
