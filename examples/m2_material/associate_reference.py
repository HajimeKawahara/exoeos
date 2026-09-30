"""Finite metal/oxygen associates under an explicit primary-data continuation.

Components are chemical species, not atomic components. Equilibrium and
element conservation remain caller responsibilities; this module supplies G.
"""

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import ClassVar

import jax.numpy as jnp
from jax import tree_util
import numpy as np
from scipy.interpolate import CubicHermiteSpline

from exoeos.ma_interval import _Interval, _SecondOrder, _interval, _ma_excess


R = 8.31446261815324
DATA_PATH = Path(__file__).with_name("associate_sources.json")
HO_DATA_PATH = Path(__file__).with_name("hydrogen_oxygen_sources.json")
COMPONENTS = ("Fe", "Si", "O", "H", "P", "Mg", "Ca", "Al", "Cr", "Ti",
              "MgO", "CaO", "AlO", "CrO", "TiO", "Al2O", "Cr2O", "Ti2O")
FORMULAS = tuple([{name: 1.} for name in COMPONENTS[:10]] +
                 [{name: 1., "O": 1.} for name in ("Mg", "Ca", "Al", "Cr", "Ti")] +
                 [{name: 2., "O": 1.} for name in ("Al", "Cr", "Ti")])


def _phosphorus():
    """Load this example's own immutable phosphorus recipe once."""
    name = "_exoeos_associate_phosphorus"
    path = Path(__file__).with_name("phosphorus_reference.py")
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise ValueError("Use one associated-alloy provider checkout per process.")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def associated_interactions(temperature_k, *, temperature_policy="constant",
                            hydrogen_oxygen_model="omitted"):
    """Return a symmetric matrix for additional, explicitly declared terms."""
    p = _phosphorus()
    p_inputs = p.phosphorus_interactions(temperature_k, temperature_policy=temperature_policy)
    data = json.loads(DATA_PATH.read_text())
    matrix = np.zeros((18, 18))
    for element, (a, b) in data["self_epsilon"].items():
        i = COMPONENTS.index(element)
        matrix[i, i] = a / temperature_k + b
    for pair, (a, b) in data["cross_epsilon"].items():
        i, j = [COMPONENTS.index(element) for element in pair.split(",")]
        matrix[i, j] = matrix[j, i] = a / temperature_k + b
    factor = 1. if temperature_policy == "constant" else 1873.15 / temperature_k
    for pair, row in data["extra_cross"].items():
        i, j = [COMPONENTS.index(element) for element in pair.split(",")]
        value = (row["epsilon"] if "epsilon" in row else
                 p.mass_percent_to_mole_interaction(row["mass_percent_e_H_Cr"],
                                                   row["solute_molar_mass_g_mol"]))
        matrix[i, j] = matrix[j, i] = value * factor
    if hydrogen_oxygen_model not in ("omitted", "schenck1961_abstract"):
        raise ValueError("Select omitted or schenck1961_abstract hydrogen/oxygen interaction.")
    ho_receipt = {"model": hydrogen_oxygen_model, "epsilon_natural_log": 0.}
    if hydrogen_oxygen_model == "schenck1961_abstract":
        ho = json.loads(HO_DATA_PATH.read_text())
        reference = ho["reported_log10_gamma_H_derivative_wrt_O_mole_fraction"] * np.log(10.)
        factor_ho = (1. if temperature_policy == "constant" else
                     ho["reported_temperature_K"] / temperature_k)
        coefficient = float(reference * factor_ho)
        matrix[2, 3] = matrix[3, 2] = coefficient
        example = ho["reported_numerical_example"]
        ho_receipt.update(epsilon_natural_log=coefficient,
                          epsilon_reference_natural_log=float(reference),
                          reference_temperature_K=ho["reported_temperature_K"],
                          temperature_policy=temperature_policy,
                          example_predicted_gamma_H=float(np.exp(reference * example["O_mole_fraction"])),
                          example_reported_gamma_H=example["gamma_H_due_to_O"],
                          source_sha256=hashlib.sha256(HO_DATA_PATH.read_bytes()).hexdigest(),
                          scope=ho["scope"], original_full_text_verified=False)
    return matrix, {"phosphorus": p_inputs, "additional_matrix": matrix.tolist(),
                    "hydrogen_oxygen": ho_receipt,
                    "temperature_policy_for_extra_P_H_cross_terms": temperature_policy,
                    "zero_terms": "Unlisted additional terms are omitted by the declared continuation, not measured to be zero. The original Jung model used no further terms for its evaluated subsystems; H/P extensions have separately listed coefficients."}


def make_associated_model(temperature_k, *, temperature_policy="constant",
                          hydrogen_oxygen_model="omitted"):
    p = _phosphorus()
    matrix, receipt = associated_interactions(temperature_k, temperature_policy=temperature_policy,
                                             hydrogen_oxygen_model=hydrogen_oxygen_model)
    model = MaAssociatedLiquid(p.MaPhosphorusLiquid(np.asarray(receipt["phosphorus"]["epsilon"])), matrix)
    return model, receipt


def _janaf_gibbs_rt(temperature, rows, hf298):
    t, s, dh = np.asarray(rows, dtype=float).T
    g = hf298 * 1000. + dh * 1000. - t * s
    return float(CubicHermiteSpline(t, g, -s, extrapolate=False)(temperature)) / (R * temperature)


def associated_standards_rt(temperature_k, host_standards_rt, gas_standards_rt):
    """Anchor associates to actual gas standards without fitting a gauge.

    Jung's free-O Henry reference remains explicit. Its difference from the
    supplied Ma free-O reference is not silently absorbed into association G.
    The existing host standards (including P) are returned unchanged.
    """
    t, host = float(temperature_k), np.asarray(host_standards_rt)
    if (not 2100. <= t <= 2400. or host.shape != (5,) or not np.isrealobj(host)
            or np.any(~np.isfinite(host))):
        raise ValueError("Require five finite host standards and 2100 <= T/K <= 2400.")
    required = ("O2", "Mg1", "Ca1", "Al1", "Cr1", "Ti1")
    if any(name not in gas_standards_rt or not np.isfinite(gas_standards_rt[name]) for name in required):
        raise ValueError("Require actual O2 and all five atomic gas standards.")
    data = json.loads(DATA_PATH.read_text())
    free = {}
    conversions = {}
    for element in COMPONENTS[5:10]:
        correction = 0.
        if element in data["pure_phase_janaf"]:
            pure, atom = data["pure_phase_janaf"][element], data["atomic_janaf"][element]
            correction = (_janaf_gibbs_rt(t, pure["rows"], pure["hf298"]) -
                          _janaf_gibbs_rt(t, [[row[0], row[1], row[3]] for row in atom["data"]],
                                         atom["formation_enthalpy_298_kJ_mol"]))
        a, b = data["henry"][element]["ln_gamma"]
        free[element] = float(gas_standards_rt[element + "1"] + correction + a / t + b)
        conversions[element] = {"pure_minus_atomic_gas_rt": correction,
                                "ln_henry_gamma": a / t + b,
                                "pure_phase": data["henry"][element]["pure_phase"]}
    a, b = data["henry"]["O"]["ln_gamma"]
    oxygen = .5 * gas_standards_rt["O2"] + a / t + b
    standards = list(host) + [free[element] for element in COMPONENTS[5:10]]
    for species, formula in zip(COMPONENTS[10:], FORMULAS[10:]):
        metal = next(element for element in formula if element != "O")
        standards.append(formula[metal] * free[metal] + oxygen +
                         data["association_delta_g_J_mol"][species] / (R * t))
    return np.asarray(standards), {
        "temperature_K": t, "component_order": list(COMPONENTS), "component_formulas": list(FORMULAS),
        "standard_potentials_rt": [float(value) for value in standards],
        "actual_gas_standard_potentials_rt": {key: float(gas_standards_rt[key]) for key in required},
        "gas_standard_pressure_Pa": data["gas_standard_pressure_Pa"],
        "gas_standard_pressure_evidence": data["gas_standard_pressure_evidence"],
        "pure_to_gas_conversions": conversions, "jung_free_oxygen_standard_rt": float(oxygen),
        "jung_minus_supplied_host_oxygen_standard_rt": float(oxygen - host[2]),
        "scope": "Gas-anchored Jung associate standards with JANAF pure-to-gas differences; original Ma/P host retained. Association standards are not recalibrated to the host O law. No pressure correction or empirical coupled error bound is supplied."}


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MaAssociatedLiquid:
    """Species-mole extensive G for the retained host and finite associates."""
    host: object
    additional_matrix: object
    components: ClassVar[tuple] = COMPONENTS
    activity_basis: ClassVar[str] = "mole_fraction"
    standard_state_convention: ClassVar[str] = "symmetric"
    reference_model_id: ClassVar[str] = "ma_p_jung_associated_metal_continuation_v1"

    def gex_RT(self, T, P, x):
        x = jnp.asarray(x)
        if x.shape != (18,):
            raise ValueError("Require the eighteen declared chemical species.")
        fraction = jnp.sum(x[:5])
        return (fraction * self.host.gex_RT(T, P, x[:5] / fraction) +
                .5 * x @ jnp.asarray(self.additional_matrix) @ x)

    def standard_state_shift_RT(self, T):
        shift = self.host.standard_state_shift_RT(T)
        return jnp.concatenate((shift, jnp.zeros(13, dtype=shift.dtype)))

    def validate_state(self, T, P, x):
        x, matrix = np.asarray(x), np.asarray(self.additional_matrix)
        if (x.shape != (18,) or matrix.shape != (18, 18) or not np.isrealobj(x)
                or not np.isrealobj(matrix) or np.any(~np.isfinite(x)) or np.any(~np.isfinite(matrix))
                or np.any(x < 0) or x[:5].sum() <= 0
                or not np.isclose(x.sum(), 1., rtol=0., atol=8*np.finfo(float).eps)
                or not np.array_equal(matrix, matrix.T)):
            raise ValueError("Require normalized species fractions and a finite symmetric interaction matrix.")
        self.host.validate_state(T, P, x[:5] / x[:5].sum())

    def tree_flatten(self):
        return (self.host, self.additional_matrix), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


def associated_excess(temperature_k, dry_coefficients, phosphorus_epsilon, matrix, solutes):
    """Pure arithmetic expression on seventeen non-Fe species fractions."""
    silicon, oxygen, hydrogen, phosphorus = solutes[:4]
    added = sum(solutes[4:])
    host_fraction = 1 - added
    value = _ma_excess(temperature_k, dry_coefficients,
                       [silicon, oxygen, hydrogen + phosphorus + added])
    quadratic = phosphorus_epsilon[0] * phosphorus * phosphorus / 2
    for coefficient, fraction in zip(phosphorus_epsilon[1:], (silicon, oxygen, hydrogen)):
        quadratic = quadratic + coefficient * phosphorus * fraction
    value = value + quadratic / host_fraction
    # Fe entries are zero by construction; avoid a redundant dependent variable.
    for i, left in enumerate(solutes, 1):
        if matrix[i][i] != 0:
            value = value + matrix[i][i] * left * left / 2
        for j in range(i + 1, 18):
            if matrix[i][j] != 0:
                value = value + matrix[i][j] * left * solutes[j - 1]
    return value


def associated_curvature_lower_bound(model, temperature_k, lower, upper):
    """Outward reduced-Hessian bound per mole of chemical species."""
    from exoeos import MaFeSiOHLiquid, MaFeSiOLiquid

    p = _phosphorus()
    if (type(model) is not MaAssociatedLiquid or type(model.host) is not p.MaPhosphorusLiquid
            or type(model.host.host) is not MaFeSiOHLiquid
            or type(model.host.host.dry_model) is not MaFeSiOLiquid):
        raise TypeError("Require the exact associated scalar and unchanged Ma/P host.")
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    matrix = np.asarray(model.additional_matrix)
    if (not np.isfinite(temperature_k) or temperature_k <= 0
            or np.any(~np.isfinite(np.asarray(model.host.epsilon)))
            or lo.shape != (18,) or hi.shape != (18,) or matrix.shape != (18, 18)
            or any(np.any(~np.isfinite(v)) for v in (lo, hi, matrix))
            or np.any(lo < 0) or np.any(lo > hi) or np.any(hi > 1)
            or lo.sum() > 1 or hi.sum() < 1 or lo[0] <= 0
            or not np.array_equal(matrix, matrix.T) or np.any(matrix[0] != 0)):
        raise ValueError("Require a finite feasible Fe-rich species box and symmetric non-Fe matrix.")
    # Tightening each upper bound by the other declared lower bounds preserves
    # every feasible point. Reject a rectangular enclosure with a singular dry
    # denominator; never infer a finite interval through a zero crossing.
    hi = np.minimum(hi, 1 - (lo.sum() - lo))
    variables = [_SecondOrder(_Interval(lo[i+1], hi[i+1]), i, dimension=17) for i in range(17)]
    value = associated_excess(temperature_k, np.asarray(model.host.host.dry_model.interaction_K),
                              np.asarray(model.host.epsilon), matrix, variables)
    free = [i for i in range(17) if lo[i+1] < hi[i+1]]
    if not free:
        return 0.
    diagonal = {i: value.hessian[i][i] + _interval(hi[i+1]).reciprocal() for i in free}
    cross = {(i, j): max(abs(value.hessian[i][j].lo), abs(value.hessian[i][j].hi))
             for i in free for j in free if i != j}
    comparison = np.array([[diagonal[i].lo if i == j else -cross[i, j] for j in free] for i in free])
    weights = np.abs(np.linalg.eigh(comparison)[1][:, 0])
    if np.any(weights <= 0) or np.any(~np.isfinite(weights)):
        weights = np.ones(len(free))
    rows = []
    for row, i in enumerate(free):
        bound = diagonal[i]
        for column, j in enumerate(free):
            if i != j:
                bound = bound - cross[i, j] * _interval(weights[column]) / weights[row]
        rows.append(bound.lo)
    result = min(rows)
    if not np.isfinite(result):
        raise ValueError("The species box does not give a finite curvature lower bound.")
    return float(result)
