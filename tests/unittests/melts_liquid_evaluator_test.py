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
            if drift == "redox":
                self.dispComposition[phase][evaluator.OXIDES.index("fe2o3")] += 0.001
            elif drift == "fraction":
                self.X[phase][0] += 0.001
            elif drift == "potential":
                self.mu[phase][0] = np.nan

    engine = Engine()
    model = SimpleNamespace(engine=engine, endMemberFormulas={"bulk": evaluator.OXIDES, "liquid": evaluator.COMPONENTS}, version="mock")
    monkeypatch.setitem(sys.modules, "meltsdynamic", SimpleNamespace(MELTSdynamic=lambda mode: model))
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
    np.testing.assert_allclose(mu_rt - mu0_rt, np.asarray(result["ln_activity_common_R"], float)[present])
    np.testing.assert_allclose(np.asarray(result["ln_activity"], float)[present], (mu - mu0) / (evaluator.BACKEND_R * result["T_K"]))
    expected_gamma = (mu - mu0) / (evaluator.COMMON_R * result["T_K"]) - np.log(np.asarray(result["x"])[present])
    np.testing.assert_allclose(np.asarray(result["ln_gamma_common_R"], float)[present], expected_gamma, atol=1e-14)
    assert np.max(np.abs(expected_gamma)) > 1e-5  # Backend-ideal does not mean common-R gamma=1.
    json.dumps(result, allow_nan=False)


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
        Path(command[command.index("--output") + 1]).write_text(json.dumps({"provenance": {}}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(evaluator.subprocess, "run", run)
    for _ in range(2):
        evaluator.evaluate_liquid(1473.15, 5e7, N, runtime=tmp_path, python_executable="/custom/python")
    assert len(directories) == 2 and all(not path.exists() for path in directories)


def test_help_works_without_an_external_runtime():
    result = subprocess.run([sys.executable, str(EXAMPLE), "--help"], check=True, capture_output=True, text=True)
    assert "--runtime" in result.stdout and "--validate" in result.stdout
