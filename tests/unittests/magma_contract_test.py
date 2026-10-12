"""Atom ledgers and reference conventions across native magma amount bases."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos.magma import (
    CLINOPYROXENE_COMPONENTS, CMAS_COMPONENTS, CMFAS_COMPONENTS, COMPONENTS,
    ELEMENTS, OLIVINE_COMPONENTS, PYROXENE_COMPONENTS, TERNARY_COMPONENTS,
    anorthite_gibbs, element_matrix, liquid_gibbs,
)


@pytest.mark.parametrize("components,fe,ca,al", [
    (COMPONENTS, 0., 0., 0.), (TERNARY_COMPONENTS, .2, 0., 0.),
    (CMAS_COMPONENTS, 0., .3, .1), (CMFAS_COMPONENTS, .2, .3, .1),
])
def test_liquid_ledger_matches_independent_oxide_atom_counts(components, fe, ca, al):
    # One mol SiO2 and 0.6 mol MgO, plus the listed FeO/CaO/Al2O3 amounts.
    amounts = {"SiO2": 1. - .3 - fe/2 - ca, "Mg2SiO4": .3,
               "Fe2SiO4": fe/2, "CaSiO3": ca, "Al2O3": al}
    n = jnp.array([amounts[name] for name in components])
    expected = {"O": 2. + .6 + fe + ca + 3*al, "Mg": .6, "Si": 1.,
                "Fe": fe, "Al": 2*al, "Ca": ca}
    assert ELEMENTS == ("O", "Mg", "Si", "Fe", "Al", "Ca", "Na", "K", "Ti", "Cr", "P")
    np.testing.assert_allclose(element_matrix(components) @ n,
                               [expected.get(name, 0) for name in ELEMENTS], rtol=0, atol=1e-15)


@pytest.mark.parametrize("components,liquid_columns", [
    (OLIVINE_COMPONENTS, [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]),
    (CLINOPYROXENE_COMPONENTS, [[.5, .5, 0, 1, 0], [.5, 0, .5, 1, 0]]),
    (PYROXENE_COMPONENTS, [[0, 0, 0, 1, 0], [.5, .5, 0, 0, 0], [.5, 0, .5, 0, 0]]),
    (("CaAl2Si2O8",), [[1, 0, 0, 1, 1]]),
    (("SiO2", "SiO2", "SiO2"), [[1, 0, 0, 0, 0]] * 3),
])
def test_solid_transfers_preserve_every_liquid_element(components, liquid_columns):
    # These independent stoichiometric reactions use formula mol for Di/Hd/An,
    # but mol Si for the three physical pyroxene coordinates.
    liquid_atoms = element_matrix(CMFAS_COMPONENTS)
    solid_atoms = element_matrix(components)
    transfer = jnp.asarray(liquid_columns).T
    np.testing.assert_array_equal(liquid_atoms @ transfer, solid_atoms)
    assert solid_atoms.shape == (len(ELEMENTS), len(components))
    assert jnp.issubdtype(solid_atoms.dtype, jnp.integer)


def test_common_element_reference_and_rt_conversion_preserve_reaction_energy():
    T, P = 1700., 5e7
    n = jnp.array([.4, .25, .075, .25, .12])
    liquid_atoms = element_matrix(CMFAS_COMPONENTS)
    anorthite_atoms = element_matrix(("CaAl2Si2O8",))[:, 0]
    # SiO2 + CaSiO3 + Al2O3 -> CaAl2Si2O8; all other amounts stay fixed.
    transfer = jnp.array([1., 0., 0., 1., 1.])
    reference = jnp.array([1300., -200., 500., 700., -1100., 300., 40., 50., 60., 70., 80.])
    shifted = lambda amounts: liquid_gibbs(T, P, amounts) + reference @ liquid_atoms @ amounts
    shifted_mu = jax.jit(jax.grad(shifted))(n)
    shifted_anorthite = anorthite_gibbs(T, P) + reference @ anorthite_atoms
    mu = jax.grad(liquid_gibbs, 2)(T, P, n)
    reaction = anorthite_gibbs(T, P) - transfer @ mu
    np.testing.assert_allclose(shifted_anorthite - transfer @ shifted_mu, reaction,
                               rtol=0, atol=3e-9)
    # The consumer's R converts output units without replacing the MELTS R.
    rt = 8.31446261815324 * T
    reduced_mu = jax.grad(lambda amounts: liquid_gibbs(T, P, amounts) / rt)(n)
    np.testing.assert_allclose(anorthite_gibbs(T, P) / rt - transfer @ reduced_mu,
                               reaction / rt, rtol=0, atol=3e-13)


@pytest.mark.parametrize("components", [(), ("H2O",)])
def test_element_lookup_rejects_empty_or_unimplemented_bases(components):
    with pytest.raises(ValueError):
        element_matrix(components)
