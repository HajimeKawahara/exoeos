"""Finite K sensitivity preserves the host scalar and conservative curvature."""
import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import total_solution_gibbs_RT,total_solution_state
from exoeos.ma_interval import _Interval

PATH=Path(__file__).resolve().parents[2]/'examples/m2_material/potassium_reference.py'
SPEC=importlib.util.spec_from_file_location('m2_potassium_test_provider',PATH)
M=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=M;SPEC.loader.exec_module(M)
T=2173.15
LO=np.r_[.75,np.zeros(18)]
HI=np.array([1.,.02,.01,.04,.02,.002,.00002,.0001,.12,.00001,.003,.00002,.0001,.005,.00001,.000001,.002,.000001,.02])


def test_explicit_standard_and_atomic_gauge_are_separate_from_material_calibration():
    gas={e+'1':-10.+i for i,e in enumerate(('Mg','Ca','Al','Cr','Ti','K'))};gas['O2']=-15.
    host=np.arange(5.)-9.
    base,receipt=M.associated_standards_rt(T,host,gas,potassium_standard_offset_rt=3.)
    assert base[-1]==gas['K1']+3.
    assert not receipt['potassium']['empirically_calibrated']
    shifted,_=M.associated_standards_rt(T,host,dict(gas,K1=gas['K1']+.7),potassium_standard_offset_rt=3.)
    np.testing.assert_allclose(shifted-base,np.r_[np.zeros(18),.7],atol=1e-14)
    for bad in (None,True,np.nan,np.inf):
        with pytest.raises((ValueError,TypeError)):
            M.associated_standards_rt(T,host,gas,potassium_standard_offset_rt=bad)


def test_perspective_host_limit_and_full_scalar_derivatives():
    model,_=M.make_associated_model(T)
    n=HI*.1;n[-1]=0;n[0]=1-n[1:].sum()
    assert float(model.gex_RT(T,2.7e7,n))==pytest.approx(float(model.host.gex_RT(T,2.7e7,n[:18])),abs=1e-14)
    n[-1]=.003
    standards=np.arange(19.)*.31-12.
    energy=lambda v:total_solution_gibbs_RT(model,T,2.7e7,v,standards)
    state=total_solution_state(model,T,2.7e7,n,standards)
    np.testing.assert_allclose(jax.grad(energy)(n),state.mu_RT,atol=1e-12)
    assert float(energy(4.3*n))==pytest.approx(4.3*float(energy(n)),abs=1e-12)
    np.testing.assert_allclose(np.asarray(jax.hessian(energy)(n))@n,0.,atol=1e-9)


def test_perspective_interval_bound_matches_independent_full_hessian():
    model,_=M.make_associated_model(T)
    lower=M.associated_curvature_lower_bound(model,T,LO,HI)
    assert .344 < lower < .345
    h=model.host
    for fraction in (.1,.7,.99):
        n=HI*fraction;n[0]=1-n[1:].sum()
        value=M.associated_excess(T,np.asarray(h.host.host.dry_model.interaction_K),np.asarray(h.host.epsilon),h.additional_matrix,[_Interval(v,v) for v in n[1:]])
        assert value.lo<=float(model.gex_RT(T,2.7e7,n))<=value.hi
        energy=lambda y:total_solution_gibbs_RT(model,T,2.7e7,jnp.r_[1-jnp.sum(y),y],jnp.zeros(19))
        assert np.linalg.eigvalsh(np.asarray(jax.hessian(energy)(n[1:])))[0]>=lower
    bad=HI.copy();bad[-1]=.04
    assert M.associated_curvature_lower_bound(model,T,LO,bad) < 0


def test_nonconvex_host_retains_a_negative_bound_without_altering_the_scalar():
    model, _ = M.make_associated_model(T)
    host = model.host
    matrix = np.array(host.additional_matrix)
    matrix[2, 3] = matrix[3, 2] = 52.4*np.log(10.)
    changed = M.PotassiumAssociatedLiquid(M._associated().MaAssociatedLiquid(host.host, matrix))
    bound = M.associated_curvature_lower_bound(changed, T, LO, HI)
    assert np.isfinite(bound) and bound < 0
    x = HI*.3
    x[0] = 1-x[1:].sum()
    energy = lambda y: total_solution_gibbs_RT(changed, T, 2.7e7,
        jnp.r_[1-jnp.sum(y), y], jnp.zeros(19))
    assert np.linalg.eigvalsh(np.asarray(jax.hessian(energy)(x[1:])))[0] >= bound


def test_hydrogen_oxygen_option_preserves_the_host_perspective_and_receipt():
    base, _ = M.make_associated_model(T)
    changed, receipt = M.make_associated_model(
        T, hydrogen_oxygen_model='schenck1961_abstract')
    epsilon = 52.4*np.log(10.)
    assert receipt['hydrogen_oxygen']['model'] == 'schenck1961_abstract'
    assert M.associated_curvature_lower_bound(changed, T, LO, HI) < 0
    for potassium in (0., .003):
        x = HI*.1
        x[-1] = potassium
        x[0] = 1-x[1:].sum()
        difference = float(changed.gex_RT(T, 2.7e7, x)-base.gex_RT(T, 2.7e7, x))
        assert difference == pytest.approx(epsilon*x[2]*x[3]/(1-potassium), abs=1e-14)
