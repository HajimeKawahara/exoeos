"""Explicit finite K-transfer sensitivity; no inferred empirical K standard."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import ClassVar

import jax.numpy as jnp
from jax import tree_util
import numpy as np

from exoeos.ma_interval import _Interval, _SecondOrder, _interval


def _associated():
    name = '_exoeos_potassium_associated'
    path = Path(__file__).with_name('associate_reference.py')
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise ValueError('Use one potassium provider checkout per process.')
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# These constants can be read without constructing or evaluating a model.
COMPONENTS = ('Fe', 'Si', 'O', 'H', 'P', 'Mg', 'Ca', 'Al', 'Cr', 'Ti',
              'MgO', 'CaO', 'AlO', 'CrO', 'TiO', 'Al2O', 'Cr2O', 'Ti2O', 'K')
FORMULAS = tuple([{name: 1.} for name in COMPONENTS[:10]] +
                 [{name: 1., 'O': 1.} for name in ('Mg', 'Ca', 'Al', 'Cr', 'Ti')] +
                 [{name: 2., 'O': 1.} for name in ('Al', 'Cr', 'Ti')] + [{'K': 1.}])


def make_associated_model(temperature_k, *, temperature_policy='constant'):
    host, receipt = _associated().make_associated_model(
        temperature_k, temperature_policy=temperature_policy)
    return PotassiumAssociatedLiquid(host), receipt


def associated_standards_rt(temperature_k, host_standards_rt, gas_standards_rt,
                            *, potassium_standard_offset_rt):
    """Use the supplied K1 gas standard plus a required sensitivity offset."""
    offset = potassium_standard_offset_rt
    if (isinstance(offset, bool) or not np.isscalar(offset) or not np.isrealobj(offset)
            or not np.isfinite(offset) or 'K1' not in gas_standards_rt
            or not np.isfinite(gas_standards_rt['K1'])):
        raise ValueError('Require a finite explicit K offset and actual K1 gas standard.')
    standards, receipt = _associated().associated_standards_rt(
        temperature_k, host_standards_rt, gas_standards_rt)
    standards = np.r_[standards, gas_standards_rt['K1']+offset]
    receipt.update(component_order=list(COMPONENTS), component_formulas=list(FORMULAS),
                   standard_potentials_rt=standards.tolist())
    receipt['potassium'] = {
        'gas_species': 'K1', 'gas_standard_rt': float(gas_standards_rt['K1']),
        'gas_to_metal_standard_offset_rt': float(offset), 'standard_rt': float(standards[-1]),
        'empirically_calibrated': False,
        'scope': 'Required explicit Henry-standard sensitivity, not an inferred material property or measured uncertainty interval. Finite K atoms are conserved; no universal coupled-error bound follows from inventory caps or a finite parameter scan.'}
    return standards, receipt


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PotassiumAssociatedLiquid:
    """Perspective of the eighteen-species host with ideal finite K mixing."""
    host: object
    components: ClassVar[tuple] = COMPONENTS
    activity_basis: ClassVar[str] = 'mole_fraction'
    standard_state_convention: ClassVar[str] = 'symmetric'
    reference_model_id: ClassVar[str] = 'ma_p_jung_associated_potassium_sensitivity_v1'

    def gex_RT(self, T, P, x):
        x = jnp.asarray(x)
        if x.shape != (19,):
            raise ValueError('Require the nineteen declared chemical species.')
        fraction = jnp.sum(x[:18])
        return fraction*self.host.gex_RT(T, P, x[:18]/fraction)

    def standard_state_shift_RT(self, T):
        shift = self.host.standard_state_shift_RT(T)
        return jnp.concatenate((shift, jnp.zeros(1, dtype=shift.dtype)))

    def validate_state(self, T, P, x):
        x = np.asarray(x)
        if (x.shape != (19,) or not np.isrealobj(x) or np.any(~np.isfinite(x))
                or np.any(x < 0) or x[:18].sum() <= 0
                or not np.isclose(x.sum(), 1., rtol=0., atol=8*np.finfo(float).eps)):
            raise ValueError('Require normalized nonnegative species fractions with a present host.')
        self.host.validate_state(T, P, x[:18]/x[:18].sum())

    def tree_flatten(self):
        return (self.host,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


def associated_excess(temperature_k, dry_coefficients, phosphorus_epsilon, matrix, solutes):
    fraction = 1-solutes[-1]
    return fraction*_associated().associated_excess(
        temperature_k, dry_coefficients, phosphorus_epsilon, matrix,
        [value/fraction for value in solutes[:-1]])


def associated_curvature_lower_bound(model, temperature_k, lower, upper):
    """Outward eighteen-dimensional bound for this exact perspective scalar."""
    a = _associated()
    if type(model) is not PotassiumAssociatedLiquid or type(model.host) is not a.MaAssociatedLiquid:
        raise TypeError('Require the exact declared potassium-associated scalar.')
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lo.shape != (19,) or hi.shape != (19,) or np.any(~np.isfinite(lo))
            or np.any(~np.isfinite(hi)) or np.any(lo < 0) or np.any(lo > hi)
            or np.any(hi > 1) or lo.sum() > 1 or hi.sum() < 1 or lo[0] <= 0):
        raise ValueError('Require a finite feasible Fe-rich nineteen-species box.')
    hi = np.minimum(hi, 1-(lo.sum()-lo))
    if np.any(lo[1:] != 0) or hi[-1] >= .5 or hi[-1] <= 0:
        raise ValueError('The perspective certificate requires zero solute lower bounds and 0 < K upper < 0.5.')
    # Project the complete nineteen-species box into the normalized host.
    # Division is evaluated outward before using its upper endpoint.
    host_hi = np.array([min(1., (_interval(v)/(1-_interval(hi[-1]))).hi)
                        for v in hi[:18]])
    kappa = a.associated_curvature_lower_bound(model.host, temperature_k, lo[:18], host_hi)
    if kappa <= 0:
        # A nonconvex host still has a well-defined scalar. Preserve its
        # conservative, possibly negative bound instead of manufacturing
        # positive curvature or changing the domain.
        variables = [_SecondOrder(_Interval(lo[i+1], hi[i+1]), i, dimension=18)
                     for i in range(18)]
        host = model.host
        value = associated_excess(temperature_k,
            np.asarray(host.host.host.dry_model.interaction_K),
            np.asarray(host.host.epsilon), np.asarray(host.additional_matrix), variables)
        free = [i for i in range(18) if lo[i+1] < hi[i+1]]
        rows = []
        for i in free:
            row = value.hessian[i][i]+_interval(hi[i+1]).reciprocal()
            for j in free:
                if j != i:
                    row = row-max(abs(value.hessian[i][j].lo), abs(value.hessian[i][j].hi))
            rows.append(row.lo)
        result = min(rows)
        if not np.isfinite(result):
            raise ValueError('The nonconvex perspective interval bound is nonfinite.')
        return float(result)
    # For z=y/(1-k), the full Hessian quadratic is
    # (dy+z dk)^T H_host (dy+z dk)/(1-k) + dk^2/[k(1-k)].
    # Young's inequality bounds it below by kappa*||dy||^2/2 +
    # [1/(kmax(1-kmax))-kappa*||z||^2] dk^2. This covers every
    # feasible point in the box without sampling or diagonalization.
    norm2 = sum((_interval(v)*_interval(v) for v in host_hi[1:]), _interval(0.))
    kmax = _interval(hi[-1])
    k_direction = (kmax*(1-kmax)).reciprocal()-_interval(kappa)*norm2
    result = min((_interval(kappa)/2).lo, k_direction.lo)
    if not np.isfinite(result):
        raise ValueError('The perspective curvature lower bound is nonfinite.')
    return float(result)
