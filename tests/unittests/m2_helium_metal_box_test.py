"""Finite-He conservation, true mass-fraction partition and endpoint checks."""

import importlib.util
import math
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "examples/m2_material/helium_metal_box.py"
SPEC = importlib.util.spec_from_file_location("helium_metal_box", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def inputs():
    return dict(total_helium_mol=2., other_gas_mol=3., pressure_bar=5.,
                dry_silicate_mass_kg=4., silicate_nonhelium_mass_kg=6.,
                metal_nonhelium_mass_kg=2., helium_molar_mass_kg=.004,
                henry_mol_per_kg_dry_bar=.1, metal_silicate_mass_fraction_D=.017)


def test_D_zero_matches_independent_quadratic_and_complete_gas_denominator():
    p = inputs()
    p['metal_silicate_mass_fraction_D'] = 0.
    result = MODULE.redistribute_helium(**p)
    capacity = p['dry_silicate_mass_kg'] * p['henry_mol_per_kg_dry_bar'] * p['pressure_bar']
    c = p['other_gas_mol'] + capacity - p['total_helium_mol']
    gas = 2 * p['total_helium_mol'] * p['other_gas_mol'] / (math.sqrt(c*c + 4*p['total_helium_mol']*p['other_gas_mol']) + c)
    assert result['helium_mol']['gas'] == pytest.approx(gas, rel=2e-15)
    assert result['helium_mol']['metal'] == 0.
    assert result['He_partial_pressure_bar'] == pytest.approx(p['pressure_bar']*gas/(p['other_gas_mol']+gas))
    assert abs(result['helium_budget_relative_residual']) < 5e-16


def test_mass_fraction_partition_uses_full_wet_and_He_inclusive_denominators():
    p = inputs()
    p.update(total_helium_mol=50., helium_molar_mass_kg=1., henry_mol_per_kg_dry_bar=10.,
             metal_silicate_mass_fraction_D=.2)
    result = MODULE.redistribute_helium(**p)
    gas, silicate, metal = (result['helium_mol'][name] for name in ('gas','silicate','metal'))
    ws = silicate / (6 + silicate)
    wm = metal / (2 + metal)
    assert wm / ws == pytest.approx(.2, rel=2e-15)
    assert silicate == pytest.approx(4*10*5*gas/(3+gas), rel=2e-15)
    assert gas + silicate + metal == pytest.approx(50, rel=2e-15)
    # Replacing total phase mass by either dry mass or He-free phase mass
    # would give a detectably different finite partition in this control.
    assert metal != pytest.approx(.2*2/4*silicate, rel=.1)
    assert metal != pytest.approx(.2*2/6*silicate, rel=.1)


def test_partition_monotonicity_and_extensive_scaling():
    p = inputs()
    low = MODULE.redistribute_helium(**p)
    high = MODULE.redistribute_helium(**{**p, 'metal_silicate_mass_fraction_D': .5})
    assert high['helium_mol']['metal'] > low['helium_mol']['metal']
    assert high['He_partial_pressure_bar'] < low['He_partial_pressure_bar']
    for key in ('total_helium_mol','other_gas_mol','dry_silicate_mass_kg',
                'silicate_nonhelium_mass_kg','metal_nonhelium_mass_kg'):
        p[key] *= 1e23
    scaled = MODULE.redistribute_helium(**p)
    for phase, value in low['helium_mol'].items():
        assert scaled['helium_mol'][phase]/1e23 == pytest.approx(value, rel=2e-15)
    assert scaled['He_partial_pressure_bar'] == pytest.approx(low['He_partial_pressure_bar'], rel=2e-15)
    assert not scaled['supports_material_admission'] and not scaled['is_BSE_error_bound']


@pytest.mark.parametrize('key', ['total_helium_mol','dry_silicate_mass_kg','henry_mol_per_kg_dry_bar','metal_nonhelium_mass_kg'])
def test_zero_inventory_capacity_or_metal_are_explicit(key):
    p = inputs(); p[key] = 0.
    result = MODULE.redistribute_helium(**p)
    assert abs(result['helium_budget_relative_residual']) < 5e-16
    if key != 'metal_nonhelium_mass_kg':
        assert result['helium_mol']['silicate'] == result['helium_mol']['metal'] == 0.
    else:
        assert result['helium_mol']['metal'] == 0.
        assert result['He_mass_fraction_total_metal'] is None


@pytest.mark.parametrize('change', [dict(other_gas_mol=0), dict(total_helium_mol=-1),
    dict(dry_silicate_mass_kg=7), dict(metal_silicate_mass_fraction_D=1),
    dict(metal_silicate_mass_fraction_D=-.1), dict(pressure_bar=float('nan')),
    dict(helium_molar_mass_kg=0), dict(henry_mol_per_kg_dry_bar=float('inf'))])
def test_invalid_or_unsupported_bases_are_rejected(change):
    with pytest.raises(ValueError):
        MODULE.redistribute_helium(**{**inputs(), **change})
