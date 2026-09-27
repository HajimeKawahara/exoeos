"""Primary-data phosphorus standards and a finite Fe-Si-O-H-P scalar.

This example combines separately measured dilute coefficients in an explicit
integrable continuation. It does not turn temperature continuation, pressure
independence, or the reported measurement errors into a material error bound.
Local equilibrium and elemental budgets belong to ExoGibbs and ExoInventory.
"""

from dataclasses import dataclass, field
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from exoeos import MaFeSiOHLiquid


R = 8.31446261815324
ATOMIC_MASSES = {"Fe": 55.845, "P": 30.973761998, "H": 1.00794, "O": 15.9994}
REFERENCE_TEMPERATURE_K = 1873.15
SOURCES = {
    "standard": "https://doi.org/10.2355/tetsutohagane1955.66.14_2032",
    "self": "https://doi.org/10.2355/tetsutohagane1955.65.2_264",
    "silicon": "https://doi.org/10.2355/isijinternational1966.23.51",
    "oxygen": "https://doi.org/10.2355/tetsutohagane1955.48.14_1729",
    "hydrogen": "https://doi.org/10.2355/tetsutohagane1955.52.13_1823",
}


def mass_percent_to_mole_interaction(e_i_j, solute_mass):
    """Convert e_i^j=d log10 f_i/d wt%j to epsilon_i^j=d ln gamma_i/d x_j.

    The mass-to-mole normalization term is essential; reciprocal mole-basis
    coefficients are equal, whereas reciprocal mass-percent coefficients are
    generally different. The coefficient refers to infinite dilution in Fe.
    """
    return (100 * np.log(10.) * solute_mass / ATOMIC_MASSES["Fe"] * e_i_j
            + 1 - solute_mass / ATOMIC_MASSES["Fe"])


def phosphorus_standard_rt(temperature_k, gas_standard_rt, *, gas_species,
                           allow_temperature_continuation=False,
                           measurement_shift_kcal_mol=0.):
    """Convert an actual P(g) or P2(g) 1-bar standard to dissolved mole P.

    Yamamoto et al. (1980) report P gas dissolution to the 1 wt% Henry state
    with gas pressures in atm. The 1.01325 correction converts the caller's
    gas standard; the mole-fraction conversion is separate. No independent
    reference gauge is fitted, and the supplied gas potential is preserved.
    """
    if (gas_species not in ("P1", "P2") or isinstance(temperature_k, bool)
            or not np.isfinite(temperature_k) or temperature_k <= 0
            or not np.isfinite(gas_standard_rt)
            or not np.isfinite(measurement_shift_kcal_mol)):
        raise ValueError("Require positive T, finite standards, and gas_species P1 or P2.")
    t = float(temperature_k)
    in_temperature_range = 1863.15 <= t <= 1923.15
    if not in_temperature_range and not allow_temperature_continuation:
        raise ValueError("P dissolution measurements span 1863.15--1923.15 K; opt in to continuation.")
    fraction, intercept, slope, error = ((1., -95.3, .0155, 2.2) if gas_species == "P1"
                                       else (.5, -37.7, .0013, 1.1))
    delta_g = (intercept + slope * t + measurement_shift_kcal_mol) * 4184.
    conversion = np.log(100 * ATOMIC_MASSES["P"] / ATOMIC_MASSES["Fe"])
    standard = fraction * (float(gas_standard_rt) + np.log(1.01325)) + delta_g / (R * t) + conversion
    return {"standard_rt": float(standard), "temperature_K": t,
            "gas_species": gas_species, "gas_standard_rt": float(gas_standard_rt),
            "gas_pressure_standard_bar": 1., "original_gas_pressure_standard_atm": 1.,
            "dissolution_delta_g_J_mol": float(delta_g),
            "mole_fraction_minus_mass_percent_standard_rt": float(conversion),
            "measurement_shift_kcal_mol": float(measurement_shift_kcal_mol),
            "reported_measurement_error_kcal_mol": error,
            "reported_error_definition": "Plus/minus two standard deviations in Table 3; not a continuation error bound.",
            "measured_temperature_interval_K": [1863.15, 1923.15],
            "measured_P_mass_percent_interval": [1., 3.],
            "temperature_continued": not in_temperature_range,
            "source": SOURCES["standard"], "empirical_coupled_error_bound": None,
            "pressure_response_calibrated": False,
            "scope": "Measured dilute standard with explicit temperature continuation. Reported errors do not bound continuation, common-reference disagreement, or coupled finite response."}


def phosphorus_interactions(temperature_k, *, temperature_policy):
    """Return epsilon_PP, epsilon_PSi, epsilon_PO, epsilon_PH and provenance.

    Constant and inverse-temperature continuations are explicit alternatives,
    not an uncertainty envelope. Each positive cross coefficient appears once
    in the scalar and affects both chemical potentials through differentiation.
    """
    if (temperature_policy not in ("constant", "enthalpic")
            or isinstance(temperature_k, bool) or not np.isfinite(temperature_k)
            or temperature_k <= 0):
        raise ValueError("Require positive T and explicit constant or enthalpic continuation.")
    epsilon = np.array([7.3, 11.9,
        mass_percent_to_mole_interaction(.06, ATOMIC_MASSES["P"]),
        mass_percent_to_mole_interaction(.015, ATOMIC_MASSES["P"])])
    factor = 1. if temperature_policy == "constant" else REFERENCE_TEMPERATURE_K / temperature_k
    return {"epsilon": (factor * epsilon).tolist(),
            "order": ["P-P", "P-Si", "P-O", "P-H"],
            "reference_temperature_K": REFERENCE_TEMPERATURE_K,
            "temperature_policy": temperature_policy,
            "sources": {key: SOURCES[key] for key in ("self", "silicon", "oxygen", "hydrogen")},
            "reported_errors_at_reference": {"P-P": .1, "P-Si": .6},
            "mass_percent_coefficients": {"e_O_P": .06, "e_H_P": .015},
            "reported_O_P_values_across_temperatures": [.05, .06, .07],
            "measured_temperature_intervals_K": {
                "P-P": [1873.15, 1873.15], "P-Si": [1873.15, 1873.15],
                "P-O": [1813.15, 1898.15], "P-H": [1723.15, 1943.15]},
            "scope": "Integrable quadratic continuation of independently measured dilute coefficients; no validated simultaneous five-component error bound or pressure term."}


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MaPhosphorusLiquid:
    """Finite five-component scalar preserving the original host at x_P=0.

    Standards remain external. The ideal term uses all five atomic fractions.
    The original extensive host excess energy is preserved under P dilution.
    Quadratic terms reproduce the supplied reciprocal dilute coefficients.
    This defines a continuation, not a unique finite-concentration inference
    from the dilute data. There are no internal equilibrium calculations.
    """

    epsilon: object
    host: MaFeSiOHLiquid = field(default_factory=MaFeSiOHLiquid)
    components: ClassVar[tuple] = ("Fe", "Si", "O", "H", "P")
    activity_basis: ClassVar[str] = "mole_fraction"
    standard_state_convention: ClassVar[str] = "symmetric"
    reference_model_id: ClassVar[str] = "ma_fe_si_o_h_p_dilute_quadratic_continuation_v1"

    def gex_RT(self, T, P, x):
        x, epsilon = jnp.asarray(x), jnp.asarray(self.epsilon)
        if x.shape != (5,) or epsilon.shape != (4,):
            raise ValueError("Require Fe,Si,O,H,P fractions and PP,PSi,PO,PH coefficients.")
        host_fraction = jnp.sum(x[:4])
        host_excess = host_fraction * self.host.gex_RT(T, P, x[:4] / host_fraction)
        return host_excess + .5 * epsilon[0] * x[4] ** 2 + x[4] * jnp.dot(epsilon[1:], x[1:4])

    def standard_state_shift_RT(self, T):
        """Preserve external P Henry standard; use original four host shifts."""
        shift = self.host.standard_state_shift_RT(T)
        return jnp.concatenate((shift, jnp.zeros((1,), dtype=shift.dtype)))

    def validate_state(self, T, P, x):
        values, epsilon = np.asarray(x), np.asarray(self.epsilon)
        if (values.shape != (5,) or epsilon.shape != (4,)
                or not np.isrealobj(values) or not np.isrealobj(epsilon)
                or np.any(~np.isfinite(values)) or np.any(~np.isfinite(epsilon))
                or np.any(values < 0) or values[:4].sum() <= 0
                or not np.isclose(values.sum(), 1., rtol=0, atol=8 * np.finfo(float).eps)):
            raise ValueError("Require finite normalized five-component fractions and finite coefficients.")
        self.host.validate_state(T, P, values[:4] / values[:4].sum())

    def tree_flatten(self):
        return (self.epsilon, self.host), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


def phosphorus_excess(temperature_k, dry_coefficients, epsilon, solutes):
    """Pure arithmetic form shared with interval and high-precision proofs.

    ``solutes`` are Si,O,H,P fractions after eliminating Fe. Objects implement
    arithmetic and log; no empirical data or standard gauge is reconstructed.
    """
    from exoeos.ma_interval import _ma_excess

    silicon, oxygen, hydrogen, phosphorus = solutes
    # The Ma four-component host dilutes its dry excess by H already.
    # Analytically cancelling the nested normalizations gives dry=1-H-P.
    host = _ma_excess(temperature_k, dry_coefficients,
                      [silicon, oxygen, hydrogen + phosphorus])
    quadratic = epsilon[0] * phosphorus * phosphorus / 2
    for coefficient, value in zip(epsilon[1:], (silicon, oxygen, hydrogen)):
        quadratic = quadratic + coefficient * phosphorus * value
    return host + quadratic


def phosphorus_curvature_lower_bound(model, temperature_k, lower, upper):
    """Outward interval Hessian bound for this exact five-component scalar.

    The present proof accepts the unmodified four-component Ma host only.
    New host interactions require their own scalar-matched proof, never an
    inherited certificate. Fixed coordinates are omitted from Gershgorin.
    """
    from exoeos import MaFeSiOLiquid
    from exoeos.ma_interval import _Interval, _SecondOrder, _interval

    if (type(model) is not MaPhosphorusLiquid or type(model.host) is not MaFeSiOHLiquid
            or type(model.host.dry_model) is not MaFeSiOLiquid):
        raise TypeError("Require the exact phosphorus continuation and unmodified Ma host.")
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    epsilon = np.asarray(model.epsilon, dtype=float)
    coefficients = np.asarray(model.host.dry_model.interaction_K, dtype=float)
    if (lo.shape != (5,) or hi.shape != (5,) or epsilon.shape != (4,)
            or coefficients.shape != (3,) or not np.isfinite(temperature_k) or temperature_k <= 0
            or any(np.any(~np.isfinite(v)) for v in (lo, hi, epsilon, coefficients))
            or np.any(lo < 0) or np.any(lo > hi) or np.any(hi > 1)
            or lo.sum() > 1 or hi.sum() < 1 or lo[0] <= 0):
        raise ValueError("Require finite positive T and a feasible Fe-rich five-component box.")
    variables = [_SecondOrder(_Interval(lo[i + 1], hi[i + 1]), i, dimension=4)
                 for i in range(4)]
    excess = phosphorus_excess(temperature_k, coefficients, epsilon, variables)
    free = [i for i in range(4) if lo[i + 1] < hi[i + 1]]
    if not free:
        return 0.
    diagonal = {i: excess.hessian[i][i] + _interval(hi[i + 1]).reciprocal() for i in free}
    cross = {(i, j): max(abs(excess.hessian[i][j].lo), abs(excess.hessian[i][j].hi))
             for i in free for j in free if i != j}
    comparison = np.array([[diagonal[i].lo if i == j else -cross[i, j]
                            for j in free] for i in free])
    # A numerical eigensystem only proposes positive similarity weights.
    # Acceptance uses outward Gershgorin on D^-1 H D for those exact binary64
    # weights. No eigenvalue computed by LAPACK is used as a rigorous bound.
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
        raise ValueError("The enclosing box does not provide a finite curvature bound.")
    return float(result)
