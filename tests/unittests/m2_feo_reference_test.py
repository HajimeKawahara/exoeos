"""Associated species must conserve Fe/O and differentiate as one scalar."""

import numpy as np
import pytest

from examples.m2_material.feo_reference import (
    binary_coexistence, interactions_rt, mixing_state_rt, oxygen_activity_shape, reference_report,
)


def test_associated_scalar_and_atomic_potentials_agree():
    t, p = 2173.15, 270.
    atoms = np.array([.97, .03])
    associated = np.array([atoms[0]-atoms[1], atoms[1]])
    g, mu = mixing_state_rt(t,p,associated)
    atomic_mu = np.array([mu[0],mu[1]-mu[0]])
    assert g == pytest.approx(atoms @ atomic_mu,abs=1e-14)
    for i in range(2):
        step = np.eye(2)[i]*1e-6
        plus, minus = atoms+step,atoms-step
        numeric = (mixing_state_rt(t,p,[plus[0]-plus[1],plus[1]])[0]
                   -mixing_state_rt(t,p,[minus[0]-minus[1],minus[1]])[0])/2e-6
        assert numeric == pytest.approx(atomic_mu[i],abs=2e-9)
    scaled_g,scaled_mu = mixing_state_rt(t,p,associated*1e24)
    assert scaled_g == pytest.approx(g*1e24)
    np.testing.assert_allclose(scaled_mu,mu,atol=1e-14)


def test_two_liquids_match_both_potentials_and_preserve_observed_discrepancies():
    state=binary_coexistence(2173.15)
    left,right=state['FeO_associated_mole_fractions']
    assert left < .03 < .9 < right
    np.testing.assert_allclose(mixing_state_rt(2173.15,1,[1-left,left])[1],
                               mixing_state_rt(2173.15,1,[1-right,right])[1],atol=1e-9)
    report=reference_report()
    assert report['accepted_coupled_material_domain'] is None
    assert max(abs(r['Frost_minus_measured_mass_percent']) for r in report['observations']) > .02
    assert report['target_published_ambient_saturation_fit_O_mass_percent'] > .6
    # Compare the exact atomic dilute derivative to the independently
    # derived coefficient; do not differentiate in associated fraction.
    first=oxygen_activity_shape(2173.15,270.,1e-6)
    assert first['Frost_ln_gamma_O_minus_infinite_dilution']/first['O_atomic_mole_fraction'] == pytest.approx(
        first['Frost_dilute_atomic_epsilon_OO'],abs=5e-7)
    a1=interactions_rt(2173.15,1)[0];a270=interactions_rt(2173.15,270)[0]
    assert a270 < a1
