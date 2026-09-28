"""Conditional finite Na in the K-associated host on a separate atomic ledger."""
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

from exoeos.ma_interval import _Interval, _SecondOrder, _interval


def _local(name, filename):
    path=Path(__file__).with_name(filename)
    if name in sys.modules:
        module=sys.modules[name]
        if Path(module.__file__).resolve()!=path.resolve():
            raise ValueError('Use one sodium provider checkout per process.')
        return module
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def _potassium():
    return _local('_exoeos_sodium_potassium','potassium_reference.py')


COMPONENTS=('Fe','Si','O','H','P','Mg','Ca','Al','Cr','Ti',
            'MgO','CaO','AlO','CrO','TiO','Al2O','Cr2O','Ti2O','K','Na')
FORMULAS=tuple([{name:1.} for name in COMPONENTS[:10]]+
               [{name:1.,'O':1.} for name in ('Mg','Ca','Al','Cr','Ti')]+
               [{name:2.,'O':1.} for name in ('Al','Cr','Ti')]+[{'K':1.},{'Na':1.}])
CALIBRATION_PATH=Path(__file__).with_name('validation')/'20260928_sodium_host_calibration/assessment.json'
CALIBRATION_SHA256='f24cb74a1cd860a3eebcd0eff5f1e05fa3e0c7cc82c99c53c952138a91175370'
PROJECTIONS=('remove_potassium_0.25','remove_potassium_0.5','remove_potassium_0.9','add_aluminum_silica_0.9')
TEMPERATURE_POLICIES=('constant_delta','published_exchange_slope')


def make_associated_model(temperature_k, *, temperature_policy='constant',
                          hydrogen_oxygen_model='omitted'):
    host,receipt=_potassium().make_associated_model(temperature_k,
        temperature_policy=temperature_policy,hydrogen_oxygen_model=hydrogen_oxygen_model)
    receipt['sodium_excess']={
        'model':'sodium_pseudoiron_excess_v1',
        'component_order':list(COMPONENTS),'component_formulas':list(FORMULAS),
        'parent_component_order':list(host.components),
        'parent_coordinate_map':'x_parent_Fe=x_Fe+x_Na; every other parent species unchanged',
        'activity_identity':'ln(gamma_Na,Henry)=ln(gamma_Fe)',
        'empirically_calibrated':False,
        'scope':'S-free conditional continuation. Na-Si/C/O/H interactions beyond the inherited Fe environment are not measured or fitted.'}
    return SodiumAssociatedLiquid(host),receipt


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SodiumAssociatedLiquid:
    """Parent excess at pseudo-Fe coordinates plus external full20 ideal mixing.

    The pseudo-component changes the excess energy arguments only. Physical
    formulas remain Fe1 and Na1 and are conserved independently by consumers.
    """
    host:object
    components:ClassVar[tuple]=COMPONENTS
    activity_basis:ClassVar[str]='mole_fraction'
    standard_state_convention:ClassVar[str]='symmetric'
    reference_model_id:ClassVar[str]='ma_p_jung_k_sodium_pseudoiron_v1'

    def gex_RT(self,T,P,x):
        x=jnp.asarray(x)
        if x.shape!=(20,):raise ValueError('Require the twenty declared chemical species.')
        parent=x[:19].at[0].add(x[-1])
        return self.host.gex_RT(T,P,parent)

    def standard_state_shift_RT(self,T):
        shift=self.host.standard_state_shift_RT(T)
        return jnp.concatenate((shift,jnp.zeros(1,dtype=shift.dtype)))

    def validate_state(self,T,P,x):
        x=np.asarray(x)
        if (x.shape!=(20,) or not np.isrealobj(x) or np.any(~np.isfinite(x)) or np.any(x<0)
            or not np.isclose(x.sum(),1.,rtol=0.,atol=8*np.finfo(float).eps)):
            raise ValueError('Require normalized finite nonnegative twenty-species fractions.')
        parent=x[:19].copy();parent[0]+=x[-1]
        self.host.validate_state(T,P,parent)

    def tree_flatten(self):return (self.host,),None

    @classmethod
    def tree_unflatten(cls,aux_data,children):
        del aux_data
        return cls(*children)


def sodium_standard_rt(temperature_k,pressure_Pa,fe_metal_standard_rt,native_liquid_properties,
                       *,projection,temperature_policy):
    """Return a one-Na standard and exact recipe; no gas offset is inferred.

    R_Na(T,P) follows native virtual standards and the supplied actual parent
    Fe standard. It is intentionally a different pressure policy from a fixed
    K gas-to-metal offset used by the parent model.
    """
    if projection not in PROJECTIONS or temperature_policy not in TEMPERATURE_POLICIES:
        raise ValueError('Select an explicit sodium projection and temperature policy.')
    numbers=(temperature_k,pressure_Pa,fe_metal_standard_rt)
    if any(isinstance(v,bool) or not np.isscalar(v) or not np.isrealobj(v) or not np.isfinite(v) for v in numbers) or temperature_k<=0 or pressure_Pa<=0:
        raise ValueError('Finite positive T/P and a finite actual Fe-metal standard are required.')
    state=native_liquid_properties
    if state['T_K']!=temperature_k or state['P_Pa']!=pressure_Pa:
        raise ValueError('Native virtual standards must be evaluated at this exact T/P.')
    common_R=float(state['basis']['common_R_J_mol_K'])
    if not np.isclose(common_R,8.31446261815324,rtol=0.,atol=0.):
        raise ValueError('The calibration uses the recorded common gas constant.')
    order=state['component_order'];mu0=state['mu0_RT']
    weights={'na2sio3':.5,'sio2':-.25,'fe2sio4':-.25}
    values={name:float(mu0[order.index(name)]) for name in weights}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError('Require all three finite native component standards.')
    raw=CALIBRATION_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CALIBRATION_SHA256:
        raise ValueError('Sodium reference calibration hash changed.')
    data=json.loads(raw)
    fit=next(row for row in data['conditional_lower_envelopes'] if
             row['projection']==projection and row['liquid_model']=='published'
             and row['temperature_policy']==temperature_policy)
    delta=fit['intercept_A_rt']+fit['inverse_temperature_B_K']/temperature_k
    virtual=sum(weights[k]*v for k,v in values.items())+.5*fe_metal_standard_rt
    standard=float(virtual+delta)
    receipt=dict(model='sodium_projected_reference_henry_v1',temperature_K=float(temperature_k),
        pressure_Pa=float(pressure_Pa),common_R_J_mol_K=common_R,
        projection=projection,temperature_policy=temperature_policy,
        native_component_standard_potentials_rt=values,virtual_component_weights=weights,
        fe_metal_standard_rt=float(fe_metal_standard_rt),fe_metal_reference_weight=.5,
        virtual_reference_rt=float(virtual),delta_rt=float(delta),standard_rt=standard,
        intercept_A_rt=fit['intercept_A_rt'],inverse_temperature_B_K=fit['inverse_temperature_B_K'],
        active_reference_runs=fit['active_runs'],reference_pressure_Pa=1e9,
        reference_temperature_K=[1683.,1783.,1883.],
        reference_calibration_sha256=CALIBRATION_SHA256,
        native_state_sha256=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        native_state_model_id=state['model_id'],native_state_provenance=state['provenance'],
        provider_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        empirical_BSE_calibration=False,
        pressure_policy='Re-evaluate the virtual native component standards and actual parent Fe-metal standard at each P; keep the declared dimensionless delta(T).',
        scope='Minimum finite standard satisfying five projected S-free nondetection constraints. Larger delta is unconstrained; carbon removal, liquid projection, Na interactions and low-pressure/high-temperature continuation are assumptions, not measured uncertainty bounds.')
    return standard,receipt


def associated_excess(temperature_k,dry_coefficients,phosphorus_epsilon,matrix,solutes):
    """Same pure-arithmetic expression; Na cancels from the pseudo-Fe coordinate."""
    return _potassium().associated_excess(temperature_k,dry_coefficients,
                                          phosphorus_epsilon,matrix,solutes[:-1])


def associated_curvature_lower_bound(model,temperature_k,lower,upper):
    """Conservative outward bound, retaining nonconvexity instead of shrinking it."""
    if type(model) is not SodiumAssociatedLiquid or type(model.host) is not _potassium().PotassiumAssociatedLiquid:
        raise TypeError('Require the declared sodium/potassium associated scalar.')
    lo,hi=np.asarray(lower,dtype=float),np.asarray(upper,dtype=float)
    if (lo.shape!=(20,) or hi.shape!=(20,) or np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi))
        or np.any(lo<0) or np.any(lo>hi) or np.any(hi>1) or lo.sum()>1 or hi.sum()<1 or lo[0]<=0):
        raise ValueError('Require a finite feasible twenty-species Fe-rich box.')
    # Keep the supplied outer box; binary64 simplex tightening could round inward.
    variables=[_SecondOrder(_Interval(lo[i+1],hi[i+1]),i,dimension=19) for i in range(19)]
    host=model.host.host
    value=associated_excess(temperature_k,np.asarray(host.host.host.dry_model.interaction_K),
        np.asarray(host.host.epsilon),np.asarray(host.additional_matrix),variables)
    free=[i for i in range(19) if lo[i+1]<hi[i+1]]
    if not free:raise ValueError('Require at least one free solute coordinate.')
    # Drop the positive-semidefinite Fe ideal rank-one Hessian. The remaining
    # ideal diagonal and exact excess Hessian admit an outward Gershgorin bound.
    rows=[]
    for i in free:
        row=value.hessian[i][i]+_interval(hi[i+1]).reciprocal()
        for j in free:
            if j!=i:row=row-max(abs(value.hessian[i][j].lo),abs(value.hessian[i][j].hi))
        rows.append(row.lo)
    result=min(rows)
    if not np.isfinite(result):raise ValueError('The declared interval Hessian bound is nonfinite.')
    return float(result)
