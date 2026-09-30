"""Finite Na and phase-mass normalization in conditional fixed boxes."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).parents[2] / "examples/m2_material/sodium_fixed_box.py"
spec = importlib.util.spec_from_file_location("sodium_fixed_box", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def args():
    return dict(active_sodium_mol=10., silicate_non_sodium_mass_kg=4.,
                metal_non_sodium_mass_kg=2., sodium_molar_mass_kg=.023)


@pytest.mark.parametrize("d", [0., .03, 1., 2., 1e5])
def test_finite_mass_fraction_ratio_and_conservation(d):
    result = module.partition_sodium(**args(), metal_silicate_mass_fraction_D=d)
    silicate, metal = (result["sodium_mol"][name] for name in ("silicate", "metal"))
    assert silicate + metal == pytest.approx(10.)
    # Independent ratio includes redistributed Na in both phase masses.
    ratio = (metal / (2. + .023 * metal)) / (silicate / (4. + .023 * silicate))
    assert ratio == pytest.approx(d, rel=1e-10)


def test_unit_partition_means_equal_concentrations_not_equal_moles():
    result = module.partition_sodium(**args(), metal_silicate_mass_fraction_D=1.)
    assert result["sodium_mol"]["metal"] == pytest.approx(10. / 3.)


def test_critical_D_reaches_absolute_inventory_target():
    threshold = module.critical_partition_coefficient(**args(), target_metal_sodium_mol=1.)
    result = module.partition_sodium(**args(), metal_silicate_mass_fraction_D=threshold)
    assert result["sodium_mol"]["metal"] == pytest.approx(1., abs=2e-15)


def test_scaling_all_amounts_and_host_masses_preserves_partition():
    original = module.partition_sodium(**args(), metal_silicate_mass_fraction_D=.3)
    scaled = args()
    for key in ("active_sodium_mol", "silicate_non_sodium_mass_kg", "metal_non_sodium_mass_kg"):
        scaled[key] *= 1e22
    result = module.partition_sodium(**scaled, metal_silicate_mass_fraction_D=.3)
    assert result["fraction_of_active_Na_in_metal"] == pytest.approx(original["fraction_of_active_Na_in_metal"])


def test_empty_inventory_and_invalid_target():
    result = module.partition_sodium(**{**args(), "active_sodium_mol": 0.},
                                     metal_silicate_mass_fraction_D=.03)
    assert result["sodium_mol"] == {"silicate": 0., "metal": 0.}
    with pytest.raises(ValueError, match="target"):
        module.critical_partition_coefficient(**args(), target_metal_sodium_mol=10.)
