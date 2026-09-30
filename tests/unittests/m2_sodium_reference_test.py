"""Primary Na censoring and standard-state conversion regressions."""
import importlib.util
from pathlib import Path
import math

import pytest


path = Path(__file__).parents[2] / "examples/m2_material/sodium_reference.py"
spec = importlib.util.spec_from_file_location("sodium_reference", path)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def rows():
    return {row["run"]: row for row in reference.replay()}


def test_reported_limits_are_not_measured_central_values():
    assert reference.observation("<29")["censored_upper"]
    observed = reference.observation("48(17)")
    assert not observed["censored_upper"]
    assert observed["reported_two_standard_errors"] == 17.
    assert reference.observation("b.d.l.")["value"] is None


def test_five_independent_phase_mass_ratios():
    expected = {"GGK1": (29, 974), "GGK2": (35, 1034),
                "GGK7": (44, 1768), "GGK8": (53, 2882), "GGK9": (30, 2053)}
    data = rows()
    for run, (limit, silicate) in expected.items():
        assert data[run]["mass_partition_ratio"] == pytest.approx(limit / silicate)
        assert data[run]["ratio_kind"].startswith("reported_censoring_upper")
        assert data[run]["pressure_GPa"] == 1.
        # Unknown Na activity is not replaced by the K activity coefficient.
        assert data[run]["concentration_bases"]["one_cation"]["log10_K"] is None


def test_silicate_error_and_censoring_have_distinct_roles():
    row = rows()["GGK1"]
    assert row["mass_partition_upper_at_reported_silicate_lower"] == pytest.approx(29 / (974 - 66))
    assert row["mass_partition_upper_at_reported_silicate_lower"] > row["mass_partition_ratio"]
    assert row["metal_Na_mass_ppm"]["reported_two_standard_errors"] is None


def test_author_selected_EPMA_and_excluded_conflicting_GGK3():
    data = rows()
    assert data["GGK6"]["selected_measurement"] == "EPMA"
    silicate_ppm = .12 * 1e4 * (2 * 22.98976928) / (2 * 22.98976928 + 15.999)
    assert data["GGK6"]["mass_partition_ratio"] == pytest.approx(293 / silicate_ppm)
    assert data["GGK6"]["mass_partition_ratio"] != pytest.approx(188 / 910)
    assert data["GGK3"]["excluded_from_detected_fit"]
    assert data["GGK3"]["metal_Na_mass_ppm"]["raw"] == "48(17)"


def test_independent_published_exchange_values_expose_the_basis_difference():
    data = rows()
    for run in ("GGK4", "GGK5b", "GGK6"):
        oxide = data[run]["concentration_bases"]["oxide_molecules"]
        cation = data[run]["concentration_bases"]["one_cation"]
        assert abs(oxide["residual_dex"]) < oxide["published_two_standard_errors"]
        assert abs(cation["residual_dex"]) > cation["published_two_standard_errors"]
    assert data["GGK6"]["concentration_bases"]["oxide_molecules"]["log10_K"] == pytest.approx(-2.2207359155)


def test_henry_rebasing_cancels_an_arbitrary_activity_reference():
    original = reference.henry_standard_rt(-4., 1., -30., -4., -20.)
    rebased = reference.henry_standard_rt(-4. + 3., 1. + 3., -30., -4., -20.)
    assert original == rebased
    assert original == -17.


def test_element_gauge_covariance_follows_the_balanced_reaction():
    original = reference.henry_standard_rt(-4., 1., -30., -4., -20.)
    sodium, oxygen, iron = 2., 3., 7.
    shifted = reference.henry_standard_rt(-4., 1., -30. + sodium + .5 * oxygen,
                                          -4. + iron, -20. + iron + oxygen)
    assert shifted - original == sodium


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_standard_is_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        reference.henry_standard_rt(bad, 1., -30., -4., -20.)
