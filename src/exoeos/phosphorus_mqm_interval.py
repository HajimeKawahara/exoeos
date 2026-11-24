"""Conservative insertion bounds for the unchanged multicomponent MQM scalar.

The bound covers the entire declared P-free host box and unrestricted P.
It is deliberately loose: a negative result is inconclusive. No material
coefficient, phase solver, empirical calibration or composition cap is added.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from . import phosphorus_mqm as binary
from . import phosphorus_mqm_alloy as alloy
from ._conditional_data import R
from .ma_fe_si_o import MaFeSiOLiquid
from .ma_interval import _Interval, _interval


@dataclass(frozen=True)
class MQMInsertionBound:
    """Global lower bound in RT per mole of the original 21 coordinates."""

    lower_rt: float
    chemical_lower_rt: float
    host_linear_lower_rt: float
    host_excess_lower_rt: float
    phosphorus_linear_lower_rt: float
    pair_energy_per_p_lower_rt: float
    entropy_lower_rt: float
    chemical_moles_per_component: tuple


def _inputs(value, shape, name):
    raw = np.asarray(value)
    if not np.isrealobj(raw) or raw.shape != shape:
        raise ValueError(f"{name} must be a real array of shape {shape}.")
    array = raw.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _sum(values):
    result = _interval(0.)
    for value in values:
        result = _checked(result + _checked(_interval(value)))
    return result


def _checked(value):
    if not np.all(np.isfinite([value.lo,value.hi])):
        raise ValueError("The inputs overflow the finite interval evaluation domain.")
    return value


def _host_linear_lower(costs, lower, upper):
    # For every beta, c.x >= beta + sum(l_i*(c_i-beta)_+
    #                                      + u_i*(c_i-beta)_-).
    # A maximum of valid lower bounds remains valid. Candidate beta values
    # are merely exact binary64 constants, not an optimization certificate.
    indices = [i for i in range(20) if i != 4]
    bounds = []
    for beta in [costs[i].lo for i in indices]:
        value = _interval(beta)
        for i in indices:
            difference = _checked(costs[i] - beta)
            positive = _Interval(max(0., difference.lo), max(0., difference.hi))
            negative = _Interval(min(0., difference.lo), min(0., difference.hi))
            value = _checked(value + lower[i]*positive + upper[i]*negative)
        bounds.append(value.lo)
    return max(bounds)


def mqm_alloy_insertion_bound(
    temperature_k, pressure_pa, pure_standards_rt, supporting_potentials_rt,
    phosphorus_epsilon, additional_matrix, host_lower, host_upper, *,
    model, delta_v_p_m3_mol, host_model=None,
):
    """Bound ``(G - supporting_plane)/(R*T)`` on the phase-component simplex.

    Standards and supporting potentials follow the 20 collapsed chemical
    species of :mod:`exoeos.associated_metal`; supplied binary64 values are
    exact constants of this bound. Supporting values can instead be outward
    upper bounds on an externally computed elemental-dual product: with
    nonnegative chemical amounts this remains a conservative insertion bound.
    A caller must enclose that product, rather than silently rounding it.

    ``host_lower/upper`` constrain the normalized P-free collapsed host.
    Their P entries must be zero. P can range from zero to pure P. The Fe
    lower bound must be positive, and the Si/Fe and O/Fe bounding ratios
    must remain below one. All phase coordinates are nonnegative and sum
    to one mole; their chemical amount need not equal one mole.

    Negative P-cross and added-species matrix terms are included explicitly;
    only nonnegative terms are omitted. Pass the same ``host_model`` as to
    the material scalar; omission selects the same default Ma host. Enable JAX float64
    before constructing provider coefficients; reduced-precision coefficients
    are rejected here. The proof concerns the declared mathematical scalar,
    not error bounds on arbitrary floating-point forward evaluations.

    The pair entropy correction is nonnegative mutual information. Chemical
    ideal mixing is bounded by ``-N_chemical*log(20)``. The full pair
    polynomial is bounded on its closed simplex; the unlike pair amount is
    bounded by the total P amount. The remaining negative interactions are
    bounded on the supplied host box. Outward-rounded interval arithmetic,
    including a logarithm series with explicit remainder, encloses all bound
    evaluation operations. No sampling supplies global coverage.

    A positive result excludes insertion against this supporting plane over
    this domain. It does not prove stability of any other phase, or validate
    the empirical material model. A negative result does not prove instability.
    """
    t = float(_inputs(temperature_k, (), "temperature_k"))
    p = float(_inputs(pressure_pa, (), "pressure_pa"))
    volume = float(_inputs(delta_v_p_m3_mol, (), "delta_v_p_m3_mol"))
    standard = _inputs(pure_standards_rt, (20,), "pure_standards_rt")
    plane = _inputs(supporting_potentials_rt, (20,), "supporting_potentials_rt")
    epsilon = _inputs(phosphorus_epsilon, (4,), "phosphorus_epsilon")
    matrix = _inputs(additional_matrix, (18, 18), "additional_matrix")
    lo = _inputs(host_lower, (20,), "host_lower")
    hi = _inputs(host_upper, (20,), "host_upper")
    if (t <= 0 or p <= 0 or np.any(lo < 0) or np.any(hi > 1)
            or np.any(lo > hi) or lo[4] != 0 or hi[4] != 0
            or sum(map(Fraction, lo)) > 1 or sum(map(Fraction, hi)) < 1 or lo[0] <= 0):
        raise ValueError("Require positive T/P and a feasible P-free host box with positive Fe.")
    pair_coefficients = np.asarray(binary.parameters_j_mol(t, model=model))
    _inputs(pair_coefficients, (4,), "pair coefficients")
    host_model = MaFeSiOLiquid() if host_model is None else host_model
    if type(host_model) is not MaFeSiOLiquid:
        raise TypeError("host_model must be the unmodified MaFeSiOLiquid scalar.")
    ma_coefficients = _inputs(host_model.interaction_K, (3,), "host interaction_K")
    collapse = np.asarray(alloy.collapse_matrix(model=model))
    if any(value.dtype != np.dtype('float64') for value in
           (pair_coefficients, np.asarray(host_model.interaction_K), collapse)):
        raise ValueError("Enable JAX float64 for the provider coefficients and collapse map.")
    rt = _checked(_interval(R)*t)
    costs = [_checked(_interval(s)-d) for s, d in zip(standard, plane)]
    costs[4] = _checked(costs[4]+(_interval(p)-1e5)*volume/rt)
    host_linear = _host_linear_lower(costs, lo, hi)

    # Ma Si-self: (1-x)*log(1-x) >= -x for 0 <= x < 1.
    # Its O-self term is nonnegative for the unchanged negative coefficient.
    # For the negative Si-O coefficient use -log(1-x) <= x/(1-x).
    si = _checked(_interval(hi[1])/lo[0])
    oxygen = _checked(_interval(hi[2])/lo[0])
    if si.hi >= 1 or oxygen.hi >= 1:
        raise ValueError("The host box must bound Si/dry and O/dry strictly below one.")
    k_si, k_o, k_cross = [_checked(_interval(value)/t) for value in ma_coefficients]
    if ma_coefficients[0] < 0 or ma_coefficients[1] > 0 or ma_coefficients[2] > 0:
        raise ValueError("Require nonnegative Si-self and nonpositive O-self/cross Ma coefficients.")
    product = si*oxygen
    pair_upper = _checked(product*(-1+1/(1-si)+1/(1-oxygen))
                  +product*product*(1/(1-si)+1/(1-oxygen)-1)/2).hi
    ma_lower = _checked(-k_si*hi[1]+k_cross*max(0., pair_upper)).lo

    # The core includes Na as pseudo-Fe and excludes K. Its denominator is
    # at least lo_Fe times host amount, and is also at least the P amount.
    core_upper = [_interval(v) for v in hi[:18]]
    core_upper[0] = core_upper[0]+hi[19]
    host_indices = [i for i in range(18) if i != 4]
    quadratic = _sum(_interval(min(0., matrix[i,j]))*core_upper[i]*core_upper[j]
                     for i in host_indices for j in host_indices)
    added = quadratic/(2*_interval(lo[0]))
    added = added + _sum((_interval(min(0., matrix[4,i]))+min(0., matrix[i,4]))
                        *core_upper[i]/2 for i in host_indices)
    phosphorus_cross = _sum(_interval(min(0., epsilon[i]))*hi[i] for i in (1,2,3))
    host_excess = _checked(_interval(ma_lower)+added+phosphorus_cross).lo

    # A valid polynomial bound for every named MQM set, including convex
    # or concave quadratic coefficients: 0 <= a,b <= 1, a+b <= 1.
    coefficients = [_interval(value) for value in pair_coefficients]
    delta_lower = _checked(coefficients[0]+_sum(min(0., value.lo) for value in coefficients[1:]))
    pair_lower = _checked(min(0., delta_lower.lo)/(2*_interval(collapse[4,20]))/rt).lo
    phosphorus_linear = _checked(costs[4]+pair_lower+_interval(min(0., matrix[4,4]))/2).lo
    entropy_lower = (-_interval(20.).log()).lo
    chemical_lower = _checked(_interval(min(_checked(_interval(host_linear)+host_excess).lo,
                                   phosphorus_linear))+entropy_lower).lo
    column_totals = [_sum(collapse[:,j]) for j in range(21)]
    minimum = min(value.lo for value in column_totals)
    maximum = max(value.hi for value in column_totals)
    lower = _checked(_interval(chemical_lower)*(minimum if chemical_lower >= 0 else maximum)).lo
    return MQMInsertionBound(lower, chemical_lower, host_linear, host_excess,
                             phosphorus_linear, pair_lower, entropy_lower,
                             (minimum, maximum))
