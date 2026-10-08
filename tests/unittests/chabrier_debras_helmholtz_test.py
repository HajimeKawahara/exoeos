"""Potential reconstruction remains separate from source table evaluation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import ChabrierDebrasEOS, HelmholtzTable, MassThermodynamicState


@pytest.fixture(scope="module")
def original():
    temperature = 10.0 ** (2 + 0.05 * np.arange(121))[:, None]
    density = 10.0 ** (-3 + 0.05 * np.arange(241))[None, :]
    gas_constant, cv = 4000.0, 6000.0
    entropy = 2e5 + cv * np.log(temperature / 100) - gas_constant * np.log(density / 1e-3)
    fields = np.zeros((121, 241, 8))
    fields[..., 0] = np.log10(gas_constant * temperature * density / 1e9)
    fields[..., 1] = np.log10(cv * temperature / 1e6)
    fields[..., 2] = np.log10(entropy / 1e6)
    fields[..., 3] = -1
    fields[..., 4] = 1
    fields[..., 5] = (cv + gas_constant) / entropy
    # Deliberately inconsistent independent columns must remain observable.
    fields[..., 6] = -gas_constant / entropy + 0.02
    fields[..., 7] = gas_constant / (cv + gas_constant) + 0.03
    return ChabrierDebrasEOS(np.zeros((121, 441, 8)), fields, variant="Y0275")


def test_conversion_preserves_original_and_exposes_source_inconsistency(original):
    before = np.asarray(original.trho_fields).copy()
    potential = original.to_helmholtz()
    assert isinstance(potential, HelmholtzTable)
    temperature, density = 1e4, 100.0
    source = original.state_trho(temperature, density)
    reconstructed = potential.state_trho(temperature, density)
    for name in ("pressure", "specific_internal_energy", "specific_entropy",
                 "dlnrho_dlnT_P", "dlnrho_dlnP_T", "dlns_dlnT_P"):
        np.testing.assert_allclose(getattr(reconstructed, name), getattr(source, name), rtol=1e-9)
    residuals = jax.jit(original.helmholtz_residuals)(potential, temperature, density)
    assert isinstance(residuals, MassThermodynamicState)
    np.testing.assert_allclose(residuals.dlns_dlnP_T, -0.02, atol=1e-9)
    np.testing.assert_allclose(residuals.adiabatic_gradient, -0.03, atol=1e-9)
    assert source.adiabatic_gradient == pytest.approx(0.43)
    np.testing.assert_array_equal(original.trho_fields, before)


def test_reconstructed_off_grid_ideal_gas_and_residual_batches(original):
    potential = original.to_helmholtz()
    temperature = jnp.array([1234.0, 5345.0, 23010.0])
    density = jnp.array([0.52, 23.5, 321.0])
    states = jax.jit(jax.vmap(potential.state_trho))(temperature, density)
    np.testing.assert_allclose(states.P, 4000 * temperature * density, rtol=1e-10)
    np.testing.assert_allclose(states.cv, 6000, rtol=1e-9)
    residuals = jax.jit(jax.vmap(lambda t, r: original.helmholtz_residuals(potential, t, r)))(temperature, density)
    np.testing.assert_allclose(residuals.adiabatic_gradient, -0.03, atol=1e-9)
    assert np.all(np.isfinite(np.asarray(residuals)))


def test_conversion_requires_float64(original):
    with jax.experimental.enable_x64(False):
        with pytest.raises(ValueError, match="float64"):
            original.to_helmholtz()
