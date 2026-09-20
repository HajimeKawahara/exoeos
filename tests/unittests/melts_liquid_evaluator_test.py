"""Offline boundary checks; the optional CLI validates actual MELTS derivatives."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


EXAMPLE = Path(__file__).resolve().parents[2] / "examples/melts_liquid_evaluator.py"
spec = importlib.util.spec_from_file_location("melts_liquid_evaluator", EXAMPLE)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
N = np.asarray(evaluator.REFERENCE["states"][0]["liquid"]["component_moles"])


@pytest.mark.parametrize("temperature,pressure,amounts,common_R", [
    (0, 1e5, N, 8.3), (np.nan, 1e5, N, 8.3), (273.15, 1e5, N, 8.3),
    (1500, 0, N, 8.3), (1500, np.inf, N, 8.3), (1500, 1e5, N, 0),
    (1500, 1e5, N, np.nan), (1500, 1e5, N[:-1], 8.3),
    (1500, 1e5, np.zeros(19), 8.3), (1500, 1e5, -N, 8.3),
    (1500, 1e5, np.full(19, np.nan), 8.3), (1500, 1e5, N[:, None], 8.3),
])
def test_invalid_requests_are_rejected_before_external_import(temperature, pressure, amounts, common_R):
    with pytest.raises(ValueError):
        evaluator.validate_request(temperature, pressure, amounts, common_R)


@pytest.mark.parametrize("mode,component", [(1, "co2"), (1, "so3"), (4, "so3"), (1, "cl2o-1"), (4, "cl2o-1"), (1, "f2o-1"), (4, "f2o-1")])
def test_placeholder_components_are_rejected_before_runtime_access(tmp_path, mode, component):
    amounts = N.copy()
    amounts[evaluator.COMPONENTS.index(component)] = 0.001
    with pytest.raises(ValueError, match="Unsupported positive component " + component):
        evaluator.evaluate_liquid(1473.15, 5e7, amounts, runtime=tmp_path, calculation_mode=mode)


@pytest.mark.parametrize("mode", [0, 2, 3, 5, True, "4", 4.0, None, [], {}])
def test_unsupported_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="calculation_mode must be 1 or 4"):
        evaluator.validate_request(1473.15, 5e7, N, evaluator.COMMON_R, mode)


def test_formula_mapping_preserves_full_background_and_redox_budgets():
    n, grams = evaluator.validate_request(1473.15, 5e7, N, evaluator.COMMON_R)
    liquid = evaluator.REFERENCE["states"][0]["phases"][0]
    np.testing.assert_allclose(grams, liquid["mass_g"] * np.asarray(liquid["oxide_wt_percent"]) / 100)
    np.testing.assert_allclose(np.linalg.solve(evaluator.NU.T, grams / evaluator.OXIDE_MASSES), n)
    formulas = {name: dict(zip(evaluator.ELEMENTS, row)) for name, row in zip(evaluator.COMPONENTS, evaluator.FORMULA_MATRIX)}
    assert {key: value for key, value in formulas["fe2sio4"].items() if value} == {"Fe": 2, "Si": 1, "O": 4}
    assert {key: value for key, value in formulas["kalsio4"].items() if value} == {"K": 1, "Al": 1, "Si": 1, "O": 4}
    assert formulas["h2o"]["H"] == 2 and formulas["cl2o-1"]["O"] == -1
    assert np.linalg.matrix_rank(evaluator.NU) == 19
    elements = dict(zip(evaluator.ELEMENTS, evaluator.FORMULA_MATRIX.T @ N))
    for element in ("Mg", "Al", "Ca", "Na", "K", "Cr", "P", "Ti", "H", "Fe", "O", "Si"):
        assert elements[element] > 0


def test_modified_runtime_is_rejected_before_import(tmp_path):
    name = next(iter(evaluator.REFERENCE["backend"]["runtime_sha256"]))
    (tmp_path / name).write_text("raise RuntimeError('must not be imported')\n")
    with pytest.raises(ValueError, match="Runtime hash mismatch"):
        evaluator.check_runtime(tmp_path)


def mock_engine(monkeypatch, drift=None):
    calls = []

    class Engine:
        def __init__(self):
            self.status = SimpleNamespace(failed=False, message="mock properties", molwts={"bulk": evaluator.OXIDE_MASSES.tolist()})
            self.calculation_mode = 1

        def setSystemProperties(self, key, value):
            calls.append((key, value))

        def calcPhaseProperties(self, phase, grams):
            calls.append("phase")
            grams = np.asarray(grams)
            n = np.linalg.solve(evaluator.NU.T, grams / evaluator.OXIDE_MASSES)
            x = n / n.sum()
            present = n > 0
            mu = -1e5 + evaluator.BACKEND_R * (self.temperature + 273.15) * np.log(x[present])
            self.mass = {phase: grams.sum()}
            self.g = {phase: n[present] @ mu}
            self.dispComposition = {phase: 100 * grams / grams.sum()}
            if drift == "phase":
                self.mass[phase] *= 1.001

        def calcEndMemberProperties(self, phase, grams):
            calls.append("endmember")
            n = np.linalg.solve(evaluator.NU.T, np.asarray(grams) / evaluator.OXIDE_MASSES)
            x = n / n.sum()
            present = n > 0
            self.X = {phase: x.copy()}
            self.mu0 = {phase: np.full(19, -1e5)}
            mu = np.full(19, np.nan)
            mu[present] = -1e5 + evaluator.BACKEND_R * (self.temperature + 273.15) * np.log(x[present])
            self.mu, self.activity = {phase: mu}, {phase: x}
            if self.calculation_mode == 4:
                carbonate = x[14] * 0.4
                self.X[phase] = np.append(x, carbonate)
                self.X[phase][0] += carbonate
                self.X[phase][10] -= carbonate
                self.X[phase][14] -= carbonate
                carbonate_mu = mu[10] + mu[14] - mu[0]
                self.mu[phase] = np.append(mu, carbonate_mu)
                self.mu0[phase] = np.append(self.mu0[phase], -1e5)
                self.activity[phase] = np.append(x, np.exp((carbonate_mu + 1e5) / (evaluator.BACKEND_R * (self.temperature + 273.15))))
            if drift == "redox":
                self.dispComposition[phase][evaluator.OXIDES.index("fe2o3")] += 0.001
            elif drift == "fraction":
                self.X[phase][0] += 0.001
            elif drift == "potential":
                self.mu[phase][0] = np.nan
            elif drift == "carbonate":
                self.mu[phase][-1] += 100

    engine = Engine()

    def model(mode):
        engine.calculation_mode = mode
        species_order = evaluator.COMPONENTS + (["caco3"] if mode == 4 else [])
        return SimpleNamespace(engine=engine, endMemberFormulas={"bulk": evaluator.OXIDES, "liquid": species_order}, version="mock")

    monkeypatch.setitem(sys.modules, "meltsdynamic", SimpleNamespace(MELTSdynamic=model))
    monkeypatch.setattr(evaluator, "check_runtime", lambda runtime: None)
    monkeypatch.setattr(evaluator, "source_provenance", lambda: {})
    monkeypatch.setattr(evaluator.importlib.metadata, "version", lambda name: "mock")
    return engine, calls


def request():
    return {"T_K": 1473.15, "P_Pa": 5e7, "component_moles": N.tolist(), "common_R_J_mol_K": evaluator.COMMON_R}


def test_worker_preserves_amounts_and_reports_two_gas_constant_conventions(monkeypatch, tmp_path):
    engine, calls = mock_engine(monkeypatch)
    result = evaluator._worker(tmp_path, request())
    assert calls == [("Log fO2 Path", "None"), "phase", "endmember"]
    assert engine.temperature == 1200 and engine.pressure == 500
    assert not result["phase_policy"]["equilibrated"] and not result["phase_policy"]["stability_checked"]
    present = N > 0
    for field in ("mu_J_mol", "mu0_J_mol", "mu_RT", "mu0_RT", "ln_activity", "ln_gamma_common_R"):
        assert [value is not None for value in result[field]] == present.tolist()
    mu, mu0 = [np.asarray(result[field], float)[present] for field in ("mu_J_mol", "mu0_J_mol")]
    mu_rt, mu0_rt = [np.asarray(result[field], float)[present] for field in ("mu_RT", "mu0_RT")]
    np.testing.assert_allclose(mu_rt, mu / (evaluator.COMMON_R * result["T_K"]))
    np.testing.assert_allclose(result["gibbs_RT"], result["gibbs_J"] / (evaluator.COMMON_R * result["T_K"]))
    np.testing.assert_allclose(mu_rt - mu0_rt, np.asarray(result["ln_activity_common_R"], float)[present])
    np.testing.assert_allclose(np.asarray(result["ln_activity"], float)[present], (mu - mu0) / (evaluator.BACKEND_R * result["T_K"]))
    expected_gamma = (mu - mu0) / (evaluator.COMMON_R * result["T_K"]) - np.log(np.asarray(result["x"])[present])
    np.testing.assert_allclose(np.asarray(result["ln_gamma_common_R"], float)[present], expected_gamma, atol=1e-14)
    assert np.max(np.abs(expected_gamma)) > 1e-5  # Backend-ideal does not mean common-R gamma=1.
    json.dumps(result, allow_nan=False)


def test_carbonate_species_are_mapped_to_independent_components(monkeypatch, tmp_path):
    engine, _ = mock_engine(monkeypatch)
    carbon_request = request()
    carbon_request["calculation_mode"] = 4
    carbon_request["component_moles"][14] = 0.001
    result = evaluator._worker(tmp_path, carbon_request)
    n = np.asarray(carbon_request["component_moles"])
    assert engine.calculation_mode == 4
    assert result["model_id"] == evaluator.MODEL_IDS[4]
    assert result["provenance"]["backend"]["calculation_mode"] == 4
    assert result["provenance"]["backend"]["model"] == "rhyolite-MELTS 1.2.0"
    assert result["backend_species"]["order"] == evaluator.COMPONENTS + ["caco3"]
    species_x = np.asarray(result["backend_species"]["x"])
    assert species_x[-1] > 0 and species_x[14] < result["x"][14]
    np.testing.assert_allclose(result["x"], n / n.sum())
    # An ideal mock in the independent basis must stay ideal after speciation.
    np.testing.assert_allclose(np.asarray(result["ln_gamma"], float)[n > 0], 0, atol=1e-14)
    # CaCO3 has Ca=1, C=1, O=3; the species ledger preserves every element.
    carbonate_formula = np.array([{"Ca": 1, "C": 1, "O": 3}.get(e, 0) for e in evaluator.ELEMENTS])
    species_elements = np.vstack([evaluator.FORMULA_MATRIX, carbonate_formula])
    np.testing.assert_allclose(species_elements.T @ species_x * n.sum(), evaluator.FORMULA_MATRIX.T @ n)
    assert "unsupported" in result["capabilities"]["nitrogen"]
    assert "unsupported" in result["capabilities"]["sulfur"]
    json.dumps(result, allow_nan=False)


def test_carbonate_reaction_potential_is_checked(monkeypatch, tmp_path):
    mock_engine(monkeypatch, "carbonate")
    carbon_request = request()
    carbon_request["calculation_mode"] = 4
    carbon_request["component_moles"][14] = 0.001
    with pytest.raises(ValueError, match="carbonate reaction potential"):
        evaluator._worker(tmp_path, carbon_request)


@pytest.mark.parametrize("drift", ["phase", "redox", "fraction", "potential"])
def test_worker_rejects_changed_composition_or_undefined_present_potential(monkeypatch, tmp_path, drift):
    mock_engine(monkeypatch, drift)
    with pytest.raises(ValueError):
        evaluator._worker(tmp_path, request())


def test_every_evaluation_uses_a_fresh_process_and_working_directory(monkeypatch, tmp_path):
    directories = []
    monkeypatch.setattr(evaluator, "check_runtime", lambda runtime: None)

    def run(command, **kwargs):
        directory = Path(kwargs["cwd"])
        assert directory.exists() and directory not in directories
        directories.append(directory)
        assert command[0] == "/custom/python" and "--worker" in command
        assert kwargs["timeout"] == 60
        assert kwargs["env"]["LD_LIBRARY_PATH"].split(":")[0] == str(tmp_path)
        input_path = Path(command[command.index("--input") + 1])
        np.testing.assert_array_equal(json.loads(input_path.read_text())["component_moles"], N)
        assert json.loads(input_path.read_text())["calculation_mode"] == (1, 4)[len(directories) - 1]
        Path(command[command.index("--output") + 1]).write_text(json.dumps({"provenance": {}}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(evaluator.subprocess, "run", run)
    for mode in (1, 4):
        evaluator.evaluate_liquid(1473.15, 5e7, N, runtime=tmp_path, python_executable="/custom/python", calculation_mode=mode)
    assert len(directories) == 2 and all(not path.exists() for path in directories)


def test_help_works_without_an_external_runtime():
    result = subprocess.run([sys.executable, str(EXAMPLE), "--help"], check=True, capture_output=True, text=True)
    assert "--runtime" in result.stdout and "--validate" in result.stdout
