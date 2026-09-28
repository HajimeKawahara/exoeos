"""Independent conservation, scalar, and reference checks for conditional Na20."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import total_solution_gibbs_RT,total_solution_state
from exoeos.ma_interval import _Interval

ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'examples/m2_material/sodium_metal.py'
SPEC=importlib.util.spec_from_file_location('m2_sodium_provider_test',PATH)
M=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=M;SPEC.loader.exec_module(M)
T=2173.15;P=2.7e7
HI=np.array([1.,.02,.01,.04,.02,.002,.00002,.0001,.12,.00001,.003,.00002,.0001,.005,.00001,.000001,.002,.000001,.02,.02])
LO=np.r_[.75,np.zeros(19)]


def model_and_composition():
    model,_=M.make_associated_model(T)
    x=HI*.05;x[0]=1-x[1:].sum()
    return model,x


def test_sodium_has_a_separate_atom_column_and_parent_excess_only():
    model,x=model_and_composition()
    assert model.components[-1]=='Na' and M.FORMULAS[-1]=={'Na':1.}
    parent=x[:19].copy();parent[0]+=x[-1]
    assert float(model.gex_RT(T,P,x))==pytest.approx(float(model.host.gex_RT(T,P,parent)),abs=2e-15)
    atoms={e:sum(n*f.get(e,0) for n,f in zip(x,M.FORMULAS)) for e in ('Fe','Na')}
    assert atoms=={'Fe':x[0],'Na':x[-1]}


def test_full_scalar_ad_euler_and_exact_henry_activity_identity():
    model,x=model_and_composition();standard=np.arange(20.)*.13-9.
    energy=lambda n:total_solution_gibbs_RT(model,T,P,n,standard)
    state=total_solution_state(model,T,P,x,standard)
    np.testing.assert_allclose(jax.grad(energy)(x),state.mu_RT,atol=2e-13)
    np.testing.assert_allclose(np.asarray(jax.hessian(energy)(x))@x,0.,atol=2e-10)
    assert float(state.gibbs_RT)==pytest.approx(float(np.dot(x,state.mu_RT)),abs=2e-13)
    assert float(energy(7.3*x))==pytest.approx(7.3*float(energy(x)),abs=2e-13)
    lngamma=np.asarray(state.mu_RT)-standard-np.log(x)
    assert lngamma[-1]==pytest.approx(lngamma[0],abs=3e-14)
    # Independent extensive parent evaluation plus Fe/Na ideal subdivision.
    parent=x[:19].copy();parent[0]+=x[-1]
    parent_energy=float(total_solution_gibbs_RT(model.host,T,P,parent,standard[:19]))
    split=x[0]*np.log(x[0])+x[-1]*np.log(x[-1])-parent[0]*np.log(parent[0])
    assert float(energy(x))==pytest.approx(parent_energy+split+x[-1]*(standard[-1]-standard[0]),abs=2e-13)


def test_zero_na_recovers_parent_and_zero_fe_is_supported_when_na_hosts_it():
    model,x=model_and_composition();standard=np.arange(20.)*.2
    for index in (19,0):
        y=x.copy()
        y[19 if index==0 else 0]+=y[index];y[index]=0
        model.validate_state(T,P,y)
        state=total_solution_state(model,T,P,y,standard)
        assert np.isfinite(float(state.gibbs_RT))
        assert np.isneginf(np.asarray(state.mu_RT)[index])
        assert np.all(np.isfinite(np.asarray(state.mu_RT)[y>0]))
    y=x[:19].copy();y[0]+=x[-1]
    z=np.r_[y,0.]
    assert float(total_solution_gibbs_RT(model,T,P,z,standard))==pytest.approx(
        float(total_solution_gibbs_RT(model.host,T,P,y,standard[:19])),abs=1e-13)
    empty=total_solution_state(model,T,P,np.zeros(20),standard)
    assert float(empty.gibbs_RT)==0 and np.all(np.isnan(empty.mu_RT))


def test_exact_pure_arithmetic_and_conservative_curvature():
    model,x=model_and_composition();host=model.host.host
    value=M.associated_excess(T,np.asarray(host.host.host.dry_model.interaction_K),
        np.asarray(host.host.epsilon),np.asarray(host.additional_matrix),[_Interval(v,v) for v in x[1:]])
    assert value.lo<=float(model.gex_RT(T,P,x))<=value.hi
    lower=M.associated_curvature_lower_bound(model,T,LO,HI)
    energy=lambda y:total_solution_gibbs_RT(model,T,P,jnp.r_[1-jnp.sum(y),y],jnp.zeros(20))
    assert np.linalg.eigvalsh(np.asarray(jax.hessian(energy)(x[1:])))[0]>=lower


def reference_state():
    return json.loads(M.CALIBRATION_PATH.read_text())['rows'][0]['models']['published']['properties']


def test_native_virtual_standard_and_separate_iron_reference_are_gauge_covariant():
    state=reference_state();t=state['T_K'];p=state['P_Pa']
    standard,receipt=M.sodium_standard_rt(t,p,-8.,state,projection='remove_potassium_0.25',temperature_policy='published_exchange_slope')
    assert receipt['active_reference_runs']==['GGK9']
    assert receipt['fe_metal_standard_rt']==-8.
    changed=copy.deepcopy(state);order=state['component_order'];na,o,si,fe=2.,.3,-1.,4.
    for name,value in [('na2sio3',2*na+si+3*o),('sio2',si+2*o),('fe2sio4',2*fe+si+4*o)]:
        changed['mu0_RT'][order.index(name)]+=value
    shifted,_=M.sodium_standard_rt(t,p,-8.+fe,changed,projection='remove_potassium_0.25',temperature_policy='published_exchange_slope')
    assert shifted-standard==pytest.approx(na,abs=2e-14)
    assert not receipt['empirical_BSE_calibration']
    with pytest.raises(ValueError,match='exact T/P'):
        M.sodium_standard_rt(t,p+1,-8.,state,projection='remove_potassium_0.25',temperature_policy='constant_delta')


def test_options_and_native_standards_are_explicit():
    state=reference_state()
    for projection,policy in [('unknown','constant_delta'),('remove_potassium_0.25','unknown')]:
        with pytest.raises(ValueError,match='explicit'):
            M.sodium_standard_rt(state['T_K'],state['P_Pa'],-8.,state,projection=projection,temperature_policy=policy)
    changed=copy.deepcopy(state);changed['basis']['common_R_J_mol_K']=8.3143
    with pytest.raises(ValueError,match='common gas'):
        M.sodium_standard_rt(state['T_K'],state['P_Pa'],-8.,changed,projection='remove_potassium_0.25',temperature_policy='constant_delta')


def test_absent_fe_and_na_preserves_parent_domain_rejection():
    model,x=model_and_composition()
    x[1]+=x[0]+x[-1];x[0]=0.;x[-1]=0.
    with pytest.raises(ValueError,match='Fe > 0'):
        model.validate_state(T,P,x)


def test_nonconvex_hydrogen_oxygen_host_is_not_certified_by_a_positive_bound():
    model,_=M.make_associated_model(T,hydrogen_oxygen_model='schenck1961_abstract')
    lower=M.associated_curvature_lower_bound(model,T,LO,HI)
    assert np.isfinite(lower) and lower<0
