"""Full-catalog thermodynamic and domain contracts for the M2 gas scenarios."""

import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos.constants import MOLAR_GAS_CONSTANT as R


PATH = Path(__file__).resolve().parents[2] / "examples/m2_material/major_gas_eos.py"
SPEC = importlib.util.spec_from_file_location("m2_major_gas_test_provider", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
SPECIES = ("H2", "He1", "H2O1", "O2", "Mg1")


def model(cross=10., policy="extrapolate", species=SPECIES):
    return M.make_major_gas_eos(species, {
        "h2_he_cm3_mol": cross, "water_cross_temperature_policy": policy,
        "trace_pair_policy": "zero"})


def test_scalar_derivatives_recover_fugacity_volume_and_extensivity():
    eos, t, p = model(), 2173.15, 267.425178265e5
    n = jnp.asarray([.75, .17, .06, .012, .008])
    scalar = lambda n, p: eos.gibbs_residual_rt(t, p, n)
    state = eos.state(t, p, n / n.sum())
    np.testing.assert_allclose(jax.grad(scalar)(n, p), state.lnphi, rtol=2e-13, atol=1e-14)
    assert float(scalar(1e24*n, p)/1e24) == pytest.approx(float(scalar(n, p)), rel=1e-13)
    assert float(n @ state.lnphi) == pytest.approx(float(scalar(n, p)), rel=1e-13)
    hessian = np.asarray(jax.hessian(scalar)(n, p))
    np.testing.assert_allclose(hessian, hessian.T, rtol=0., atol=1e-13)
    np.testing.assert_allclose(hessian @ np.asarray(n), 0., atol=1e-13)
    # The pressure derivative of total G/(RT), including N*ln(P), gives V/(RT).
    derivative = jax.grad(lambda pressure: scalar(n, pressure))(p) + n.sum()/p
    assert float(derivative) == pytest.approx(float(n.sum()/(state.rho*R*t)), rel=2e-13)
    masses = jnp.asarray([.002016, .004003, .018015, .031998, .024305])
    assert float(eos.mass_density(t, p, n, masses)) == pytest.approx(float(state.rho*(n @ masses)))


def test_trace_dilution_uses_the_full_basis_in_scalar_and_every_chemical_potential():
    eos, t, p = model(), 2173.15, 270e5
    x = jnp.asarray([.64, .12, .04, .1, .1])
    state = eos.state(t, p, x)
    matrix = eos.coefficients(t)
    assert np.count_nonzero(np.asarray(matrix)[3:]) == 0
    np.testing.assert_allclose(matrix, matrix.T, atol=0.)
    np.testing.assert_allclose(state.lnphi[3:], -jnp.log(state.Z), rtol=1e-13)
    major_only = model(species=SPECIES[:3]).state(t, p, x[:3]/x[:3].sum())
    assert float(state.Z) < float(major_only.Z)
    np.testing.assert_allclose(jax.grad(lambda n: eos.gibbs_residual_rt(t, p, n))(x),
                               state.lnphi, rtol=1e-13, atol=1e-14)
    # Finite zero faces remain in the catalog and in the residual gradient.
    face = jnp.asarray([.75, .18, .07, 0., 0.])
    np.testing.assert_allclose(jax.grad(lambda n: eos.gibbs_residual_rt(t, p, n))(face),
                               eos.state(t, p, face).lnphi, rtol=1e-13, atol=1e-14)


def test_species_permutation_and_zero_major_face_recover_the_same_eos_and_ideal_limit():
    eos, t, p = model(), 1500., 2e5
    x = jnp.asarray([.7, .19, .08, .02, .01])
    permutation = np.array([3, 2, 0, 4, 1])
    other = model(species=tuple(SPECIES[i] for i in permutation))
    original, changed = eos.state(t, p, x), other.state(t, p, x[permutation])
    assert float(changed.rho) == pytest.approx(float(original.rho), rel=1e-14)
    np.testing.assert_allclose(changed.lnphi, original.lnphi[permutation], atol=1e-14)
    ideal = eos.state(t, p, jnp.asarray([0., 0., 0., .4, .6]))
    assert float(ideal.Z) == 1.
    assert float(ideal.rho) == pytest.approx(p/(R*t))
    assert float(ideal.gres_RT) == 0.
    np.testing.assert_array_equal(ideal.lnphi, jnp.zeros(len(SPECIES)))


@pytest.mark.parametrize("cross", [0., 10., 20.])
@pytest.mark.parametrize("policy", ["extrapolate", "hold_2000"])
def test_all_six_scenarios_have_replayable_source_receipts_and_curvature(cross, policy):
    eos, t, p = model(cross, policy), 2173.15, 320e5
    receipt = eos.parameters(t, p)
    assert receipt["species"] == list(SPECIES)
    assert receipt["options"] == eos.options
    assert receipt["convexity_diagnostic"]["certified"] is False
    alpha = receipt["convexity_diagnostic"]["entropy_curvature_lower"]
    assert 0. < alpha < 1.
    np.testing.assert_array_equal(receipt["coefficients_m3_mol"], eos.coefficients(t))
    assert receipt["temperature_assumptions"]["water_cross_high_temperature_assumption_active"]
    for name, digest in receipt["original_source_checkout"]["file_sha256"].items():
        assert receipt["provider_recipe_file_sha256"]["examples/m2_material/"+name] == digest
    # A numerical check supports the analytic bound; it is not the global proof.
    x = jnp.asarray([.2, .1, .5, .15, .05])
    energy = lambda y: jnp.sum(y*jnp.log(y)) + eos.gibbs_residual_rt(t, p, y)
    hessian = np.asarray(jax.hessian(energy)(x))
    tangent = np.eye(len(x))[:, :-1] - np.eye(len(x))[:, -1:]
    reduced = tangent.T @ (hessian - alpha*np.diag(1/np.asarray(x))) @ tangent
    assert np.linalg.eigvalsh(reduced).min() > 0.


def test_high_temperature_policy_only_changes_water_cross_terms_and_preserves_reference():
    extrapolated, held = model(), model(policy="hold_2000")
    np.testing.assert_array_equal(extrapolated.coefficients(2000.), held.coefficients(2000.))
    t = 2173.15
    a, b, endpoint = (np.asarray(extrapolated.coefficients(t)),
                      np.asarray(held.coefficients(t)), np.asarray(held.coefficients(2000.)))
    changed = np.zeros(a.shape, dtype=bool)
    changed[0, 2] = changed[2, 0] = changed[1, 2] = changed[2, 1] = True
    np.testing.assert_array_equal(a[~changed], b[~changed])
    np.testing.assert_array_equal(b[changed], endpoint[changed])
    reference = M._REFERENCE.pair_coefficients_cm3_mol(t)
    assert a[0, 0]*1e6 == pytest.approx(reference["H2-H2"], abs=1e-12)
    assert a[1, 1]*1e6 == pytest.approx(reference["He-He"], abs=1e-12)
    assert a[2, 2]*1e6 == pytest.approx(reference["H2O-H2O"], abs=1e-12)
    assert a[0, 2]*1e6 == pytest.approx(reference["H2-H2O"], abs=1e-12)
    assert a[1, 2]*1e6 == pytest.approx(reference["He-H2O"], abs=1e-12)


def test_jit_and_temperature_derivatives_work_away_from_policy_corners():
    eos = model()
    x = jnp.asarray([.75, .17, .06, .012, .008])
    value = jax.jit(lambda t, p, n: eos.gibbs_residual_rt(t, p, n))(2173.15, 270e5, x)
    assert np.isfinite(float(value))
    assert np.isfinite(float(jax.grad(lambda t: eos.gibbs_residual_rt(t, 270e5, x))(2173.15)))
    derivative = jax.grad(lambda t: model(policy="hold_2000").coefficients(t)[0, 2])(2173.15)
    assert float(derivative) == 0.


@pytest.mark.parametrize("options", [None, {}, {
    "h2_he_cm3_mol": True, "water_cross_temperature_policy": "extrapolate", "trace_pair_policy": "zero"
}, {"h2_he_cm3_mol": np.nan, "water_cross_temperature_policy": "extrapolate", "trace_pair_policy": "zero"}, {
    "h2_he_cm3_mol": 10., "water_cross_temperature_policy": "guessed", "trace_pair_policy": "zero"
}, {"h2_he_cm3_mol": 10., "water_cross_temperature_policy": "extrapolate", "trace_pair_policy": "ideal"}])
def test_undeclared_or_ambiguous_options_fail_closed(options):
    with pytest.raises(ValueError):
        M.make_major_gas_eos(SPECIES, options)


def test_invalid_catalog_composition_amounts_and_domains_fail_closed():
    for species in (SPECIES[:-3], SPECIES+("H2",), "H2", ()):
        with pytest.raises(ValueError):
            model(species=species)
    eos = model()
    for t in (999., 3001., np.nan):
        with pytest.raises(ValueError):
            eos.parameters(t, 270e5)
    for p in (0., -1., np.nan):
        with pytest.raises(ValueError):
            eos.parameters(2173.15, p)
    for x in (jnp.ones(3), jnp.ones(5), jnp.asarray([1.1, -.1, 0., 0., 0.])):
        with pytest.raises(ValueError):
            eos.state(2173.15, 270e5, x)
    with pytest.raises(ValueError, match="positive total"):
        eos.gibbs_residual_rt(2173.15, 270e5, jnp.zeros(5))
    with pytest.raises(ValueError, match="stable density bound"):
        model(-1000.).parameters(2173.15, 270e5)
    with pytest.raises(ValueError, match="curvature"):
        model(10000.).parameters(2173.15, 270e5)
    with pytest.raises(ValueError, match="stable low-density root"):
        eos.state(1000., 2e9, jnp.asarray([0., 0., 1., 0., 0.]))
