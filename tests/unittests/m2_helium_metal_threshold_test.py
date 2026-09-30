"""An inverse partition threshold is not an empirical bound on partitioning."""

import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/m2_material"))
try:
    threshold = importlib.import_module("helium_metal_threshold").metal_partition_threshold
    forward = importlib.import_module("helium_metal_box").redistribute_helium
finally:
    sys.path.pop(0)


def inputs():
    return dict(total_helium_mol=2., other_gas_mol=3., pressure_bar=5.,
                dry_silicate_mass_kg=4., silicate_nonhelium_mass_kg=6.,
                metal_nonhelium_mass_kg=2., helium_molar_mass_kg=.004,
                henry_mol_per_kg_dry_bar=.1)


def test_small_threshold_replays_through_the_independent_forward_equation():
    result = threshold(inputs(), .001)
    d = result['critical_mass_fraction_partition_D']
    assert 0 < d < 1
    replay = forward(**inputs(), metal_silicate_mass_fraction_D=d)
    assert replay['fraction_of_total_helium']['metal'] == pytest.approx(.001, rel=2e-15)
    assert replay['He_partial_pressure_bar'] == pytest.approx(result['He_partial_pressure_bar'], rel=2e-15)


def test_superunity_D_is_a_valid_inverse_state_with_finite_mass_fractions():
    result = threshold(inputs(), .5)
    assert result['critical_mass_fraction_partition_D'] > 1
    assert result['He_mass_fraction_total_metal'] < 1
    assert sum(result['helium_mol'].values()) == pytest.approx(2., rel=2e-15)
    gas, silicate, metal = (result['helium_mol'][key] for key in ('gas','silicate','metal'))
    ws, wm = .004*silicate/(6+.004*silicate), .004*metal/(2+.004*metal)
    assert result['critical_mass_fraction_partition_D'] == pytest.approx(wm/ws, rel=2e-15)
    assert silicate == pytest.approx(4*.1*5*gas/(3+gas), rel=2e-15)
    assert not result['critical_D_is_adopted_material_parameter']
    assert not result['supports_material_admission'] and not result['is_BSE_error_bound']


@pytest.mark.parametrize('target', [0., 1., -.1, float('nan')])
def test_invalid_target_is_rejected(target):
    with pytest.raises(ValueError):
        threshold(inputs(), target)


@pytest.mark.parametrize('key', ['total_helium_mol','metal_nonhelium_mass_kg','dry_silicate_mass_kg','henry_mol_per_kg_dry_bar'])
def test_no_finite_metal_threshold_without_the_required_host_or_solute(key):
    with pytest.raises(ValueError):
        threshold({**inputs(), key: 0.}, .01)


def test_prescribed_D_cannot_silently_be_overwritten_by_a_threshold():
    with pytest.raises(ValueError, match='unknown'):
        threshold({**inputs(), 'metal_silicate_mass_fraction_D': .017}, .01)
