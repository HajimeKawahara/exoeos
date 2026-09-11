"""Run a finite-H numerical control; all gas/H standards are synthetic."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

# Select this provider checkout even if an older ExoEOS is installed.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import exoeos
import jax
import jax.numpy as jnp
import numpy as np

from exoeos import MaFeSiOHLiquid, solution_state


def run_reference(hydrogen_budget: float = 0.4) -> dict:
    """Close finite atomic H between alloy and an ideal H2/He gas control.

    Synthetic standards are zero for atomic H in alloy and molecular H2 in
    gas, independently of the model's conversion shift. The sole exchange
    is 2 H(alloy) = H2(gas), with ideal gas fugacity x_H2 * P. This fixture
    tests dilution and accounting; it is not a physical partition law or an
    ExoGibbs equilibrium implementation.
    """
    if not math.isfinite(hydrogen_budget) or hydrogen_budget <= 0:
        raise ValueError("hydrogen_budget must be finite and positive.")
    if not jax.config.x64_enabled:
        raise RuntimeError("Run this numerical reference with JAX_ENABLE_X64=1.")
    assert Path(exoeos.__file__).resolve().parent == ROOT / "src" / "exoeos"
    model = MaFeSiOHLiquid()
    temperature, pressure = 1873.0, 1.0e5
    dry_amounts = np.asarray([0.85, 0.10, 0.05])
    helium_amount = 0.3

    def exchange_residual(hydrogen_in_alloy):
        gas_h2 = (hydrogen_budget - hydrogen_in_alloy) / 2
        alloy_x_h = hydrogen_in_alloy / (dry_amounts.sum() + hydrogen_in_alloy)
        gas_x_h2 = gas_h2 / (gas_h2 + helium_amount)
        return 2 * math.log(alloy_x_h) - math.log(gas_x_h2 * pressure / 1e5)

    lower, upper = 0.0, hydrogen_budget
    for _ in range(70):
        midpoint = (lower + upper) / 2
        if exchange_residual(midpoint) > 0:
            upper = midpoint
        else:
            lower = midpoint
    hydrogen_in_alloy = (lower + upper) / 2
    alloy_amounts = np.append(dry_amounts, hydrogen_in_alloy)
    gas_amounts = np.asarray([(hydrogen_budget - hydrogen_in_alloy) / 2, helium_amount])
    x = alloy_amounts / alloy_amounts.sum()
    model.validate_state(temperature, pressure, x)
    state = jax.jit(solution_state)(model, temperature, pressure, jnp.asarray(x))
    # Rows are Fe, Si, O, H, He; columns retain their phase component bases.
    alloy_formula = np.vstack((np.eye(4, dtype=int), np.zeros(4, dtype=int)))
    gas_formula = np.asarray([[0, 0], [0, 0], [0, 0], [2, 0], [0, 1]])
    budget = np.append(dry_amounts, [hydrogen_budget, helium_amount])
    closure = alloy_formula @ alloy_amounts + gas_formula @ gas_amounts - budget
    chemical_residual = exchange_residual(hydrogen_in_alloy) + 2 * float(state.lngamma[3])
    assert np.max(np.abs(closure)) < 1e-12
    assert abs(chemical_residual) < 1e-12
    source_paths = [
        Path(__file__).resolve(),
        *sorted((ROOT / "src" / "exoeos").glob("*.py")),
        ROOT / "tests" / "reference" / "fe_si_o_ma2001.json",
    ]
    return {
        "model_id": model.reference_model_id,
        "evidence_level": "consistent_mechanism_control_with_synthetic_standards",
        "provenance": {
            "exoeos_import_path": exoeos.__file__,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
            "file_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
            "dry_model_id": model.dry_model.reference_model_id,
            "interaction_K": np.asarray(model.dry_model.interaction_K).tolist(),
            "python": sys.version,
            "jax": jax.__version__,
            "numpy": np.__version__,
            "dtype": str(state.lngamma.dtype),
            "command": [sys.executable, *sys.argv],
            "environment": {name: os.environ.get(name) for name in ("JAX_ENABLE_X64", "JAX_PLATFORMS")},
            "reproduction_command": "JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python examples/fe_si_o_h_control.py",
        },
        "domain": {
            "mathematical": "T/P > 0; normalized nonnegative fractions; positive dry amount; dry Fe > 0; dry solutes < 1",
            "calibration": "No finite-H calibration; dry source has no established calibration box",
            "extrapolation": "Conditional numerical control; pressure dependence is absent",
            "stable_phase_evidence": "Not assessed; T=1873 K and P=1 bar are numerical inputs",
            "exclusions": ["physical H partition", "Okuchi H standards", "melt", "phase competition", "S/C/N", "H excess interactions"],
        },
        "ledger": {
            "elements": ["Fe", "Si", "O", "H", "He"],
            "amount_unit": "mol of the listed component",
            "standard_potentials_unit": "mu0/(R*T)",
            "reference_pressure_Pa": 1e5,
            "alloy": {
                "components": list(model.components),
                "formula_matrix": alloy_formula.tolist(),
                "activity_basis": "four-component atomic mole fractions",
                "standard_convention": "formal dry standards; independently prescribed synthetic mu0_H/(R*T)=0",
                "standard_state_shift_RT": np.asarray(model.standard_state_shift_RT(temperature)).tolist(),
                "excess_equation": "n_d * psi_d(T, P, n_dry/n_d); no H excess interactions",
            },
            "gas": {
                "components": ["H2", "He"],
                "formula_matrix": gas_formula.tolist(),
                "activity_basis": "ideal molecular mole fractions times P/P0",
                "standard_convention": "synthetic mu0_H2/(R*T)=0; He inert with finite inventory",
            },
        },
        "state": {
            "T_K": temperature,
            "P_Pa": pressure,
            "element_budget_mol": budget.tolist(),
            "alloy_amounts_mol": alloy_amounts.tolist(),
            "gas_amounts_mol": gas_amounts.tolist(),
            "alloy_x": x.tolist(),
            "alloy_gex_RT": float(state.gex_RT),
            "alloy_lngamma": np.asarray(state.lngamma).tolist(),
        },
        "acceptance": {
            "max_element_residual_mol": float(np.max(np.abs(closure))),
            "exchange_residual_RT": chemical_residual,
            "tolerance": 1e-12,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_reference(), indent=2))
