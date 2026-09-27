"""Closed mineral site domains and native-standard receipt contracts."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MIXING = load("melts_solid_mixing")
NATIVE = load("melts_liquid_evaluator")


def parameters(phase):
    return MIXING.solid_mixing_parameters(phase, 2173.15, 267.20283416157343e5)


def test_hornblende_includes_the_signed_endmember_corner_and_continuous_limits():
    model = parameters("hornblende")
    state = MIXING.solid_mixing_state(model, [1., 1.])
    assert state["native_endmember_fractions"] == [-1., 1., 1.]
    assert state["mixing_gibbs_rt"] == pytest.approx(0., abs=1e-14)
    oxide = np.array(model["endmember_oxide_moles"]).T @ [-1., 1., 1.]
    assert np.all(oxide >= 0)
    for corner in ([0., 0.], [1., 0.], [0., 1.]):
        assert MIXING.solid_mixing_state(model, corner)["mixing_gibbs_rt"] == pytest.approx(0., abs=1e-14)


@pytest.mark.parametrize("phase,point", [("kalsilite", [0.5, 0., 0.]), ("nepheline", [0.5, 0., 0.5, 0.])])
def test_zero_vacancy_endpoints_diverge_instead_of_receiving_a_finite_fill(phase, point):
    model = parameters(phase)
    result = MIXING.solid_mixing_state(model, point)
    assert result["boundary_limit"] == "+infinity"
    assert result["mixing_gibbs_rt"] is None
    near = list(point)
    near[1] = 1e-16
    assert MIXING.solid_mixing_state(model, near)["mixing_gibbs_rt"] > 1000


def test_olivine_keeps_the_full_physical_fe_mg_ca_site_face():
    model = parameters("olivine")
    assert model["required_absent_elements"] == ["Mn", "Ni", "Co"]
    corner = MIXING.solid_mixing_state(model, [1., 0., 1.])
    assert corner["native_endmember_fractions"] == [0.5, 1., -0.5]
    with pytest.raises(ValueError, match="site constraint"):
        MIXING.solid_mixing_state(model, [0., 0.8, 0.8])


def test_pure_reference_uncertainty_is_not_reported_as_evaluated_mixing_energy():
    model = parameters("melilite")
    result = MIXING.solid_mixing_state(model, [0.3, 0.2, 0.1, 0.2])
    assert result["status"] == "ok_unreferenced_site_properties"
    assert "mixing_gibbs_rt" not in result
    assert result["pure_reference_bounds"][0]["enthalpy_upper_J_mol"] == 21588.5


def test_binary_feldspar_variant_retains_its_distinct_expression_and_provenance():
    plagioclase = parameters("plagioclase")
    feldspar = parameters("alkali-feldspar")
    assert plagioclase["source_kind"] == "pinned_native_binary_instruction_transcription"
    assert plagioclase["source_commit"] is None
    difference = (MIXING.solid_mixing_state(plagioclase, [0.5, 0.])["mixing_gibbs_rt"]
                  - MIXING.solid_mixing_state(feldspar, [0.5, 0.])["mixing_gibbs_rt"])
    assert difference * plagioclase["T_K"] * MIXING.COMMON_R == pytest.approx(-7390.875, abs=1e-8)


def test_orthopyroxene_pure_references_keep_the_monoclinic_source_convention():
    cpx = parameters("clinopyroxene")
    opx = parameters("orthopyroxene")
    assert cpx["pure_reference_models"] == opx["pure_reference_models"]
    assert cpx["polynomial_rt"] != opx["polynomial_rt"]


def test_rhm_retains_the_complete_manganese_free_site_domain():
    model = parameters("rhm-oxide")
    assert model["required_absent_elements"] == ["Mn"]
    assert model["native_endmember_indices"] == [0, 1, 2, 4]
    assert [row["endmember_index"] for row in model["pure_reference_models"]] == [0, 1, 2, 4]
    assert any(site["coefficient_rt"] < 0 for site in model["entropy_sites"])


def test_standards_keep_the_actual_probe_without_reusing_its_composition(monkeypatch):
    matrix = np.array([[2., 0.], [0., 3.]])
    basis = {"phase": "solid", "native_endmember_oxide_mass_g_per_mol": matrix.tolist()}
    monkeypatch.setattr(NATIVE, "molar_candidate_properties", lambda model, name: basis)

    class Engine:
        status = SimpleNamespace(failed=False)
        temperature, pressure = 1900., 200.

        def calcEndMemberProperties(self, name, oxide):
            np.testing.assert_array_equal(oxide, [1., 1.5])
            self.mu0 = {name: [-100., -200.]}
            self.X = {name: [0.8, 0.2]}

    result = NATIVE.candidate_standard_state_properties(SimpleNamespace(engine=Engine()), "solid")
    assert result["mu0_J_mol"] == [-100., -200.]
    assert result["probe"]["requested_native_fractions"] == [0.5, 0.5]
    assert result["probe"]["returned_native_fractions"] == [0.8, 0.2]
    assert result["T_K"] == 2173.15 and result["P_Pa"] == 2e7
    assert "gibbs_J" not in result
