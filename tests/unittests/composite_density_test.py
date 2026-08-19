"""Composite mass-density provider contracts."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from exoeos import (
    AdditiveVolumeCompositeDensityProvider,
    DensityComponent,
    FixedCompositionDensityProvider,
    MassDensityProvider,
    TPHelmholtzDensityProvider,
)


class _MassState(NamedTuple):
    mass_density: jax.Array


class _FixedDensityEOS(NamedTuple):
    density_scale: jax.Array

    def state_tp(self, temperature, pressure):
        return _MassState(self.density_scale * pressure / temperature)


class _ScaledMolarDensityEOS(NamedTuple):
    molar_density_scale: jax.Array

    def alphar(self, temperature, molar_density, mole_fractions):
        del temperature, mole_fractions
        return jnp.zeros_like(molar_density)

    def molar_density(
        self,
        temperature,
        pressure,
        mole_fractions,
        phase="vapor",
    ):
        del mole_fractions
        phase_scale = 1.0 if phase == "vapor" else 2.0
        return phase_scale * self.molar_density_scale * pressure / temperature


class _VectorMolarDensityEOS(NamedTuple):
    def alphar(self, temperature, molar_density, mole_fractions):
        del temperature, mole_fractions
        return jnp.zeros_like(molar_density)

    def molar_density(
        self,
        temperature,
        pressure,
        mole_fractions,
        phase="vapor",
    ):
        del temperature, pressure, mole_fractions, phase
        return jnp.ones(2)


def _providers(dtype=jnp.float64):
    fixed = FixedCompositionDensityProvider(
        eos=_FixedDensityEOS(jnp.asarray(2.0, dtype=dtype)),
        molar_masses=jnp.asarray([2.0e-3, 4.0e-3], dtype=dtype),
        expected_mass_fractions=jnp.asarray([0.75, 0.25], dtype=dtype),
        composition_rtol=1.0e-6,
    )
    water = TPHelmholtzDensityProvider(
        eos=_ScaledMolarDensityEOS(jnp.asarray(0.25, dtype=dtype)),
        molar_masses=jnp.asarray([18.0e-3], dtype=dtype),
    )
    return fixed, water


def _model(
    dtype=jnp.float64,
    *,
    species=("H2", "He", "H2O"),
    reverse_components=False,
):
    fixed, water = _providers(dtype)
    components = (
        DensityComponent(("H2", "He"), fixed),
        DensityComponent(("H2O",), water),
    )
    if reverse_components:
        components = components[::-1]
    return AdditiveVolumeCompositeDensityProvider(
        species=species,
        components=components,
    )


def test_composite_density_matches_independent_additive_volume_oracle() -> None:
    model = _model()
    temperature = jnp.asarray(500.0)
    pressure = jnp.asarray(1.0e5)
    mole_fractions = jnp.asarray([0.6, 0.1, 0.3])

    density = model.mass_density_tp(temperature, pressure, mole_fractions)

    species_masses = mole_fractions * jnp.asarray([2.0e-3, 4.0e-3, 18.0e-3])
    total_mass = jnp.sum(species_masses)
    hhe_mass_fraction = jnp.sum(species_masses[:2]) / total_mass
    water_mass_fraction = species_masses[2] / total_mass
    hhe_density = 2.0 * pressure / temperature
    water_density = (0.25 * pressure / temperature) * 18.0e-3
    expected = 1.0 / (
        hhe_mass_fraction / hhe_density + water_mass_fraction / water_density
    )

    assert jnp.allclose(density, expected, rtol=1.0e-13)


def test_species_labels_map_permuted_input_and_component_orders() -> None:
    canonical = _model()
    permuted = _model(
        species=("H2O", "He", "H2"),
        reverse_components=True,
    )

    canonical_density = canonical.mass_density_tp(
        500.0,
        1.0e5,
        jnp.asarray([0.6, 0.1, 0.3]),
    )
    permuted_density = permuted.mass_density_tp(
        500.0,
        1.0e5,
        jnp.asarray([0.3, 0.1, 0.6]),
    )

    assert jnp.allclose(canonical_density, permuted_density, rtol=1.0e-13)
    assert jnp.allclose(
        canonical.molar_masses,
        jnp.asarray([2.0e-3, 4.0e-3, 18.0e-3]),
    )
    assert jnp.allclose(
        permuted.molar_masses,
        jnp.asarray([18.0e-3, 4.0e-3, 2.0e-3]),
    )


def test_validate_species_requires_the_declared_order() -> None:
    model = _model()

    assert model.validate_species(("H2", "He", "H2O")) is None
    with pytest.raises(ValueError):
        model.validate_species(("He", "H2", "H2O"))


def test_fixed_composition_provider_accepts_matching_group_mass_fractions() -> None:
    fixed, _ = _providers()
    local_mole_fractions = jnp.asarray([6.0 / 7.0, 1.0 / 7.0])

    density = fixed.mass_density_tp(500.0, 1.0e5, local_mole_fractions)

    assert jnp.allclose(density, 400.0)


def test_fixed_composition_provider_returns_nan_for_a_mismatch() -> None:
    fixed, _ = _providers()

    density = jax.jit(fixed.mass_density_tp)(
        500.0,
        1.0e5,
        jnp.asarray([0.5, 0.5]),
    )

    assert jnp.isnan(density)


def test_zero_abundance_fixed_group_is_ignored_without_nan() -> None:
    model = _model()

    density = jax.jit(model.mass_density_tp)(
        500.0,
        1.0e5,
        jnp.asarray([0.0, 0.0, 1.0]),
    )

    expected_water_density = (0.25 * 1.0e5 / 500.0) * 18.0e-3
    assert jnp.isfinite(density)
    assert jnp.allclose(density, expected_water_density)


def test_zero_abundance_pure_component_recovers_fixed_group_density() -> None:
    model = _model()

    density = model.mass_density_tp(
        500.0,
        1.0e5,
        jnp.asarray([6.0 / 7.0, 1.0 / 7.0, 0.0]),
    )

    assert jnp.isfinite(density)
    assert jnp.allclose(density, 400.0)


def test_positive_fixed_group_mismatch_propagates_nan() -> None:
    model = _model()

    density = model.mass_density_tp(500.0, 1.0e5, jnp.asarray([0.5, 0.2, 0.3]))

    assert jnp.isnan(density)


@pytest.mark.parametrize(
    ("species", "component_species"),
    [
        ((), (("H2", "He"), ("H2O",))),
        (("H2", "H2", "H2O"), (("H2", "He"), ("H2O",))),
        (("H2", "He", "H2O"), ()),
        (("H2", "He", "H2O"), ((), ("H2O",))),
        (("H2", "He", "H2O"), (("H2", "H2"), ("H2O",))),
        (("H2", "He", "H2O"), (("H2", "Xe"), ("H2O",))),
        (("H2", "He", "H2O"), (("H2", "He"), ("H2",))),
        (("H2", "He", "H2O"), (("H2", "He"),)),
    ],
)
def test_composite_constructor_rejects_invalid_species_topology(
    species,
    component_species,
) -> None:
    fixed, water = _providers()
    providers = (fixed, water)

    with pytest.raises(ValueError):
        components = tuple(
            DensityComponent(labels, provider)
            for labels, provider in zip(component_species, providers)
        )
        AdditiveVolumeCompositeDensityProvider(species, components)


def test_composite_constructor_rejects_provider_molar_mass_arity() -> None:
    _, water = _providers()
    wrong_arity = TPHelmholtzDensityProvider(
        _ScaledMolarDensityEOS(jnp.asarray(0.25)),
        jnp.asarray([2.0e-3]),
    )

    with pytest.raises(ValueError):
        AdditiveVolumeCompositeDensityProvider(
            species=("H2", "He", "H2O"),
            components=(
                DensityComponent(("H2", "He"), wrong_arity),
                DensityComponent(("H2O",), water),
            ),
        )


def test_fixed_provider_rejects_expected_mass_fraction_shape() -> None:
    with pytest.raises(ValueError):
        FixedCompositionDensityProvider(
            eos=_FixedDensityEOS(jnp.asarray(2.0)),
            molar_masses=jnp.asarray([2.0e-3, 4.0e-3]),
            expected_mass_fractions=jnp.asarray([1.0]),
            composition_rtol=1.0e-6,
        )


def test_fixed_provider_rejects_negative_composition_tolerance() -> None:
    with pytest.raises(ValueError):
        FixedCompositionDensityProvider(
            eos=_FixedDensityEOS(jnp.asarray(2.0)),
            molar_masses=jnp.asarray([2.0e-3, 4.0e-3]),
            expected_mass_fractions=jnp.asarray([0.75, 0.25]),
            composition_rtol=-1.0,
        )


@pytest.mark.parametrize(
    ("temperature", "pressure", "mole_fractions"),
    [
        (jnp.asarray([500.0]), 1.0e5, jnp.asarray([0.6, 0.1, 0.3])),
        (500.0, jnp.asarray([1.0e5]), jnp.asarray([0.6, 0.1, 0.3])),
        (500.0, 1.0e5, jnp.asarray(1.0)),
        (500.0, 1.0e5, jnp.asarray([0.6, 0.4])),
    ],
)
def test_composite_call_rejects_invalid_state_shapes(
    temperature,
    pressure,
    mole_fractions,
) -> None:
    with pytest.raises(ValueError):
        _model().mass_density_tp(temperature, pressure, mole_fractions)


def test_component_provider_must_return_scalar_density() -> None:
    provider = TPHelmholtzDensityProvider(
        eos=_VectorMolarDensityEOS(),
        molar_masses=jnp.asarray([18.0e-3]),
    )
    model = AdditiveVolumeCompositeDensityProvider(
        species=("H2O",),
        components=(DensityComponent(("H2O",), provider),),
    )

    with pytest.raises(ValueError):
        model.mass_density_tp(500.0, 1.0e5, jnp.asarray([1.0]))


def test_composite_supports_jit_vmap_grad_and_pytree_round_trip() -> None:
    model: MassDensityProvider = _model()
    temperature = 500.0
    pressure = 1.0e5
    composition = jnp.asarray([0.6, 0.1, 0.3])

    evaluate = jax.jit(
        lambda provider, value_t, value_p, values_x: provider.mass_density_tp(
            value_t,
            value_p,
            values_x,
        )
    )
    density = evaluate(model, temperature, pressure, composition)
    batched = jax.vmap(model.mass_density_tp)(
        jnp.asarray([400.0, 500.0]),
        jnp.asarray([0.8e5, 1.0e5]),
        jnp.stack([composition, composition]),
    )
    temperature_gradient = jax.grad(model.mass_density_tp, argnums=0)(
        temperature,
        pressure,
        composition,
    )
    pressure_gradient = jax.grad(model.mass_density_tp, argnums=1)(
        temperature,
        pressure,
        composition,
    )
    leaves, tree_definition = jax.tree_util.tree_flatten(model)
    reconstructed = jax.tree_util.tree_unflatten(tree_definition, leaves)

    assert leaves
    assert isinstance(reconstructed, AdditiveVolumeCompositeDensityProvider)
    assert jnp.allclose(
        reconstructed.mass_density_tp(temperature, pressure, composition),
        density,
    )
    assert batched.shape == (2,)
    assert jnp.allclose(batched[0], batched[1])
    assert jnp.allclose(temperature_gradient, -density / temperature)
    assert jnp.allclose(pressure_gradient, density / pressure)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_composite_preserves_explicit_inexact_dtype(dtype) -> None:
    model = _model(dtype)

    density = model.mass_density_tp(
        jnp.asarray(500.0, dtype=dtype),
        jnp.asarray(1.0e5, dtype=dtype),
        jnp.asarray([0.6, 0.1, 0.3], dtype=dtype),
    )

    assert density.dtype == dtype
