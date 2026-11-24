"""Outward insertion bounds for the reduced two-power volatile dry11 model.

The caller supplies enclosed coefficients of the declared reduced scalar.
This module does not select a phase, adopt a water law, or certify an arbitrary
floating-point backend. Numerical outputs belong to the calling experiment.
"""

from functools import lru_cache
import math
from numbers import Integral
import time
import heapq

import numpy as np

from .ma_interval import _Interval as I, _interval as iv
from .water_interval import _exp_point_nonpositive


def _sumiv(values):
    s = iv(0.0)
    for x in values:
        s = s + x
    return s


@lru_cache(maxsize=100000)
def _logiv(x):
    return iv(x).log()


def _expiv(x):
    def point(y):
        return (
            _exp_point_nonpositive(y)
            if y <= 0
            else _exp_point_nonpositive(-y).reciprocal()
        )

    return I(point(x.lo).lo, point(x.hi).hi)


def _mag(x):
    return max(abs(x.lo), abs(x.hi))


def _dot(a, b):
    return _sumiv(x * y for x, y in zip(a, b))


def _vals(a, which="lo"):
    return np.array([getattr(x, which) for x in a])


def _power(x, n):
    ans = iv(1.0)
    for _ in range(n):
        ans = ans * x
    return ans


def _intervals(value, shape, name):
    raw = np.asarray(value)
    if not np.isrealobj(raw) or raw.shape != shape + (2,):
        raise ValueError(
            f"{name} must have shape {shape + (2,)} with real interval endpoints."
        )
    raw = raw.astype(float)
    if not np.all(np.isfinite(raw)) or np.any(raw[..., 0] > raw[..., 1]):
        raise ValueError(f"{name} must contain finite ordered intervals.")
    result = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        result[index] = I(*map(float, raw[index]))
    return result.item() if not shape else result


def reduced_two_power_melt_bound(
    linear_rt,
    quadratic_rt,
    entropy_coefficient,
    water_equivalent,
    water_log_terms,
    water_log_predictors,
    predictor_counts,
    water_exponents,
    reference_composition,
    *,
    local_margin=0.03,
    curvature_floor=0.5,
    max_boxes=200000,
    progress=None,
):
    """Enclose the dry11 simplex minimum after conditional guest elimination.

    The mathematical scalar, in RT per dry-component mole, is
    ``q(x) = c.x + r*sum(x*log(x)) + x.W.x/2 - (S.x)*I(t(x))``,
    ``t_i(x) = exp(b_i + v_i.x/(d.x))``; I is the continuous two-power
    water integral. Dry compositions satisfy x>=0 and sum(x)=1. The returned
    lower bound includes the complete simplex, not just the sampled reference.

    Coefficient intervals use a trailing (lower, upper) axis. Their shapes
    before that axis are c=(11,), W=(11,11), r=(), S=(11,), b=(2,) and
    v=(2,11). Positive counts d=(11,) and powers a=(2,) are exact supplied
    binary64 constants. W must be symmetric with exactly zero diagonal.
    The function does not infer these coefficients from a provider callback.
    The caller must establish the guest reduction, domain, gauge and units.

    ``reference_composition`` has eleven positive entries summing to one
    within 1e-12; its first coordinate is reconstructed as the exact real
    expression 1-sum(the last ten entries). The local box uses exact binary64
    caps min(reference+local_margin,1). Interval LDL must certify the stated
    positive curvature floor there, including the water Hessian bound.
    McCormick cuts and entropy/linear-program duals exclude exterior boxes.
    Candidate cuts and dual values may be computed approximately; every
    accepted lower bound is evaluated with outward arithmetic.

    A finite node budget returns ``complete=False`` and no whole-domain
    lower bound when boxes remain. Failure of the local certificate raises
    ValueError; it is inconclusive, not a negative insertion witness.
    Pressure inversion, quadrature and backend evaluation errors are not
    included. No empirical material accuracy or equilibrium classification
    follows from this continuous mathematical bound alone.
    """
    c = _intervals(linear_rt, (11,), "linear_rt")
    W = _intervals(quadratic_rt, (11, 11), "quadratic_rt")
    r = _intervals(entropy_coefficient, (), "entropy_coefficient")
    s = _intervals(water_equivalent, (11,), "water_equivalent")
    base = _intervals(water_log_terms, (2,), "water_log_terms")
    v = _intervals(water_log_predictors, (2, 11), "water_log_predictors")
    arrays = [
        np.asarray(value)
        for value in (predictor_counts, water_exponents, reference_composition)
    ]
    if any(not np.isrealobj(value) for value in arrays):
        raise ValueError("Counts, exponents and reference composition must be real.")
    counts, a, x0 = [value.astype(float) for value in arrays]
    if (
        counts.shape != (11,)
        or a.shape != (2,)
        or x0.shape != (11,)
        or not np.all(np.isfinite(np.r_[counts, a, x0]))
        or np.any(counts <= 0)
        or np.any(a < 0.25)
        or np.any(a > 1)
        or np.any(x0 <= 0)
        or abs(x0.sum() - 1) > 1e-12
        or r.lo <= 0
        or any(value.lo <= 0 for value in s)
        or not np.isfinite(local_margin)
        or local_margin <= 0
        or not np.isfinite(curvature_floor)
        or curvature_floor <= 0
        or isinstance(max_boxes, bool)
        or not isinstance(max_boxes, Integral)
        or max_boxes <= 0
    ):
        raise ValueError(
            "Require positive finite coefficients, a valid dry reference and a positive search budget."
        )
    for i in range(11):
        if W[i, i].lo != 0 or W[i, i].hi != 0:
            raise ValueError("The interaction diagonal must be exactly zero.")
        for j in range(11):
            if W[i, j] != W[j, i] or W[i, j].lo < 0 < W[i, j].hi:
                raise ValueError("Require symmetric interactions with enclosed signs.")
    cap = np.minimum(x0 + local_margin, 1.0)
    y0 = [iv(x) for x in x0[1:]]
    xx = [1 - _sumiv(y0)] + y0
    if any(x.lo <= 0 or x.hi > cap[i] for i, x in enumerate(xx)):
        raise ValueError("The exact reference must be inside the declared local box.")

    def linear_upper(coeff, lo, hi):
        # Any beta gives a valid box-simplex Lagrangian upper bound.
        approx = np.array([x.hi for x in coeff])
        left = 1.0 - sum(lo)
        chosen = approx[0]
        for j in np.argsort(-approx):
            chosen = approx[j]
            left -= hi[j] - lo[j]
            if left <= 0:
                break
        total = iv(float(chosen))
        for cost, l, h in zip(coeff, lo, hi):
            z = cost - float(chosen)
            total = (
                total
                + h * I(max(0.0, z.lo), max(0.0, z.hi))
                + l * I(min(0.0, z.lo), min(0.0, z.hi))
            )
        return total.hi

    def ratio_bounds(row, lo, hi):
        d = [iv(x) for x in counts]
        denlo = max(min(counts), -linear_upper([-x for x in d], lo, hi))
        denhi = min(max(counts), linear_upper(d, lo, hi))
        nlo = -linear_upper([-x for x in row], lo, hi)
        nhi = linear_upper(row, lo, hi)
        raw = I(nlo, nhi) / I(denlo, denhi)
        global_ratio = [x / d for x, d in zip(row, counts)]
        return I(
            max(raw.lo, min(x.lo for x in global_ratio)),
            min(raw.hi, max(x.hi for x in global_ratio)),
        )

    def series(terms, orders=3, gradient=False):
        total = iv(0.0)
        grad = [iv(0.0), iv(0.0)]
        for m in range(1, orders + 1):
            for k in range(m + 1):
                value = (
                    math.comb(m, k)
                    * _power(terms[0], k)
                    * _power(terms[1], m - k)
                    / (k * iv(a[0]) + (m - k) * iv(a[1]))
                )
                total = total + value
                grad[0] = grad[0] + k * value
                grad[1] = grad[1] + (m - k) * value
        w = _sumiv(terms)
        tail = _power(w, orders + 1) / ((orders + 1) * iv(min(a)) * (1 - w))
        total = total + I(0.0, tail.hi)
        gtail = _power(w, orders + 1) / (iv(min(a)) * (1 - w))
        grad = [g + I(0.0, gtail.hi) for g in grad]
        return (total, grad) if gradient else total

    def water_upper(lo, hi):
        terms = [_expiv(b + ratio_bounds(row, lo, hi)) for b, row in zip(base, v)]

        if _sumiv(terms).hi >= 0.5:
            raise ValueError("The water-term bound must remain below one half.")
        Iup = series([iv(x.hi) for x in terms]).hi
        return (iv(linear_upper(s, lo, hi)) * Iup).hi

    # Local insertion value and gradient, with the continuous water minimum.
    S = _dot(s, xx)
    den = _dot([iv(x) for x in counts], xx)
    logs = [b + _dot(row, xx) / den for b, row in zip(base, v)]
    terms = list(map(_expiv, logs))
    integ, igrad = series(terms, orders=4, gradient=True)
    q0 = (
        _dot(c, xx)
        + r * _sumiv(x * x.log() for x in xx)
        + _dot(xx, [_dot(row, xx) for row in W]) / 2
        - S * integ
    )
    grad = []
    for j in range(1, 11):
        glog = [
            (row[j] - row[0]) / den
            - _dot(row, xx) * (iv(counts[j]) - counts[0]) / (den * den)
            for row in v
        ]
        dry = (
            c[j]
            - c[0]
            + r * (xx[j].log() - xx[0].log())
            + _dot([W[j][k] - W[0][k] for k in range(11)], xx)
        )
        grad.append(dry - (s[j] - s[0]) * integ - S * _dot(igrad, glog))

    # Uniform absolute Hessian row sum for S(x)*I(t(x)) on the local box/simplex.
    lo = np.zeros(11)
    hi = cap.copy()
    ratio = [ratio_bounds(row, lo, hi) for row in v]
    tmax = [_expiv(b + rr).hi for b, rr in zip(base, ratio)]
    wt = _sumiv(iv(z) for z in tmax)
    if wt.hi >= 0.5:
        raise ValueError("The local water-term bound must remain below one half.")
    Dbox = I(
        max(min(counts), -linear_upper([iv(-x) for x in counts], lo, hi)),
        min(max(counts), linear_upper([iv(x) for x in counts], lo, hi)),
    )
    smax = linear_upper(s, lo, hi)
    Dgrad = [iv(x) - counts[0] for x in counts[1:]]
    lg = []
    lh = []
    for row, rat in zip(v, ratio):
        delta = [x - row[0] for x in row[1:]]
        lg.append([(dx - rat * dd) / Dbox for dx, dd in zip(delta, Dgrad)])
        lh.append(
            [
                [
                    (
                        -delta[j] * Dgrad[k]
                        - delta[k] * Dgrad[j]
                        + 2 * rat * Dgrad[j] * Dgrad[k]
                    )
                    / (Dbox * Dbox)
                    for k in range(10)
                ]
                for j in range(10)
            ]
        )
    Il = [iv(tmax[i]) / (iv(a[i]) * _power(1 - wt, 2)) for i in range(2)]
    Ill = [
        [
            2 * iv(tmax[i]) * tmax[j] / ((iv(a[i]) + a[j]) * _power(1 - wt, 3))
            + (Il[i] if i == j else 0.0)
            for j in range(2)
        ]
        for i in range(2)
    ]
    sh = [z - s[0] for z in s[1:]]
    waterH = []
    for j in range(10):
        row = []
        for k in range(10):
            dI = [_sumiv(iv(_mag(lg[i][l])) * Il[i] for i in range(2)) for l in (j, k)]
            second = _sumiv(iv(_mag(lh[i][j][k])) * Il[i] for i in range(2))
            second = second + _sumiv(
                iv(_mag(lg[i][j])) * _mag(lg[l][k]) * Ill[i][l]
                for i in range(2)
                for l in range(2)
            )
            row.append((_mag(sh[j]) * dI[1] + _mag(sh[k]) * dI[0] + smax * second).hi)
        waterH.append(row)
    K = max(_sumiv(iv(x) for x in row).hi for row in waterH)
    kappa = curvature_floor
    A = [
        [
            iv(r.lo) / cap[0]
            + W[j + 1][k + 1]
            - W[j + 1][0]
            - W[0][k + 1]
            + W[0][0]
            + (iv(r.lo) / cap[j + 1] - K - kappa if j == k else 0.0)
            for k in range(10)
        ]
        for j in range(10)
    ]
    # Interval LDL; every exact symmetric matrix enclosed by A has positive pivots.
    L = [[iv(float(i == j)) for j in range(10)] for i in range(10)]
    pivots = []
    for j in range(10):
        pivot = A[j][j] - _sumiv(L[j][k] * L[j][k] * pivots[k] for k in range(j))
        if pivot.lo <= 0:
            raise ValueError("The requested local curvature floor is not certified.")
        pivots.append(pivot)
        for i in range(j + 1, 10):
            L[i][j] = (
                A[i][j] - _sumiv(L[i][k] * L[j][k] * pivots[k] for k in range(j))
            ) / pivot
    if not all(math.isfinite(z) for z in (q0.lo, q0.hi, K, *map(_mag, grad))):
        raise ValueError("The local interval evaluation is not finite.")
    local_bound = (q0 - _sumiv(iv(_mag(g)) * _mag(g) for g in grad) / (2 * kappa)).lo

    # Floating candidates for cut and entropy-dual choices; proof evaluation follows.
    Wf = np.array([[z.lo for z in row] for row in W])
    rh = r.hi

    def tighten(lo, hi):
        for _ in range(3):
            sumhi = _sumiv(iv(x) for x in hi)
            sumlo = _sumiv(iv(x) for x in lo)
            lo = np.maximum(lo, np.array([(1 - (sumhi - x)).lo for x in hi]))
            hi = np.minimum(hi, np.array([(1 - (sumlo - x)).hi for x in lo]))
        return np.maximum(lo, 0), np.minimum(hi, 1)

    def entropy_dual(d, lo, hi):
        df = _vals(d)
        left, right = -100.0, 100.0
        for _ in range(35):
            lam = (left + right) / 2.0
            n = np.clip(np.exp(-(df + lam) / rh - 1), lo, hi)
            if n.sum() > 1:
                left = lam
            else:
                right = lam
        lam = (left + right) / 2.0
        value = iv(-lam)
        for z, l, h in zip(df, lo, hi):
            k = iv(float(z)) + lam
            exponent = -(k / rh + 1)
            point = _expiv(exponent)
            if point.hi < l:
                x = float(l)
            elif point.lo > h:
                x = float(h)
            else:
                value = value - rh * point
                continue
            value = value + (iv(0.0) if x == 0 else iv(rh) * x * _logiv(x) + k * x)
        return value.lo

    def bound(lo, hi):
        d = list(c)
        constant = iv(0.0)
        mid = np.minimum(np.maximum(x0, lo), hi)
        for i in range(11):
            for j in range(i + 1, 11):
                w = W[i][j]
                if w.lo >= 0:
                    cuts = [
                        (lo[j], lo[i], -iv(lo[i]) * lo[j]),
                        (hi[j], hi[i], -iv(hi[i]) * hi[j]),
                    ]
                    u = max(cuts, key=lambda z: z[0] * mid[i] + z[1] * mid[j] + z[2].lo)
                elif w.hi <= 0:
                    cuts = [
                        (lo[j], hi[i], -iv(hi[i]) * lo[j]),
                        (hi[j], lo[i], -iv(lo[i]) * hi[j]),
                    ]
                    u = min(cuts, key=lambda z: z[0] * mid[i] + z[1] * mid[j] + z[2].hi)
                else:
                    raise ValueError("interaction sign is not enclosed")
                d[i] = d[i] + w * u[0]
                d[j] = d[j] + w * u[1]
                constant = constant + w * u[2]
        return (iv(entropy_dual(d, lo, hi)) + constant - water_upper(lo, hi)).lo

    heap = []
    serial = 0
    heapq.heappush(heap, (-1e20, serial, np.zeros(11), np.ones(11)))
    pruned = local = empty = 0
    lowers = []
    t0 = time.perf_counter()
    MAX = max_boxes
    processed = 0
    for it in range(MAX):
        if not heap:
            break
        _, _, lo, hi = heapq.heappop(heap)
        processed += 1
        lo, hi = tighten(lo, hi)
        if (
            np.any(lo > hi)
            or _sumiv(iv(z) for z in lo).lo > 1
            or _sumiv(iv(z) for z in hi).hi < 1
        ):
            empty += 1
            continue
        if np.all(hi <= cap):
            local += 1
            continue
        value = bound(lo, hi)
        if not math.isfinite(value):
            raise ValueError("The box bound is not finite.")
        if value >= 0:
            pruned += 1
            lowers.append(value)
            continue
        widths = hi - lo
        priority = widths * (abs(Wf) @ widths + 1.0)
        j = int(np.argmax(priority))
        middle = float((lo[j] + hi[j]) / 2)
        if not lo[j] < middle < hi[j]:
            raise ValueError("Cannot split unresolved box")
        for side in (0, 1):
            nl = lo.copy()
            nh = hi.copy()
            if side == 0:
                nh[j] = middle
            else:
                nl[j] = middle
            serial += 1
            heapq.heappush(heap, (value, serial, nl, nh))
        if progress is not None and it % 1000 == 0:
            progress(
                dict(
                    processed=it + 1,
                    pending=len(heap),
                    lower_rt=value,
                    elapsed_seconds=time.perf_counter() - t0,
                )
            )

    complete = not heap
    result = dict(
        complete=complete,
        lower_rt_per_dry_mole=min(local_bound, min(lowers, default=math.inf))
        if complete
        else None,
        local_lower_rt=local_bound,
        local_value_interval_rt=[q0.lo, q0.hi],
        local_gradient_interval_rt=[[z.lo, z.hi] for z in grad],
        local_upper=cap.tolist(),
        curvature_floor=curvature_floor,
        minimum_ldl_pivot_lower=min(z.lo for z in pivots),
        water_hessian_norm_upper=K,
        processed_boxes=processed,
        excluded_boxes=pruned,
        local_boxes=local,
        empty_boxes=empty,
        unresolved_boxes=len(heap),
        minimum_excluded_lower_rt=min(lowers) if lowers else None,
        elapsed_seconds=time.perf_counter() - t0,
        scope="Continuous reduced scalar on the complete dry11 simplex; numerical water/backend errors excluded.",
    )
    return result
