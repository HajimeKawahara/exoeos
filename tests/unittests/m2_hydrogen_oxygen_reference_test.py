"""Natural-log conversion and reciprocity of the optional H/O scalar."""

import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import total_solution_gibbs_RT, total_solution_state
from exoeos.ma_interval import _Interval

PATH = Path(__file__).resolve().parents[2]/"examples/m2_material/associate_reference.py"
SPEC = importlib.util.spec_from_file_location("m2_ho_test_provider", PATH)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def test_contemporary_log10_example_is_reconstructed_without_assuming_natural_log():
    model, receipt = M.make_associated_model(1883.15, hydrogen_oxygen_model="schenck1961_abstract")
    ho = receipt["hydrogen_oxygen"]
    assert ho["epsilon_natural_log"] == pytest.approx(52.4*np.log(10))
    assert ho["example_predicted_gamma_H"] == pytest.approx(1.52, abs=.01)
    assert np.exp(52.4*.00343) < 1.20  # The unconverted convention fails this example.
    continued, c = M.make_associated_model(2173.15, temperature_policy="enthalpic",
                                           hydrogen_oxygen_model="schenck1961_abstract")
    assert c["hydrogen_oxygen"]["epsilon_natural_log"] == pytest.approx(52.4*np.log(10)*1883.15/2173.15)
    omitted, _ = M.make_associated_model(1883.15)
    assert omitted.additional_matrix[2, 3] == 0
    with pytest.raises(ValueError, match="hydrogen/oxygen"):
        M.make_associated_model(2173.15, hydrogen_oxygen_model="guessed")


def test_added_potential_is_one_extensive_scalar_and_interval_uses_the_same_HO_term():
    t=2173.15
    new, _ = M.make_associated_model(t, hydrogen_oxygen_model="schenck1961_abstract")
    old, _ = M.make_associated_model(t)
    n=np.full(18,1e-6);n[:5]=[.965,.001,.003,.03,.001]
    eps=new.additional_matrix[2,3]
    energy=lambda model,x:total_solution_gibbs_RT(model,t,2.7e7,x,jnp.zeros(18))
    difference=lambda x:energy(new,x)-energy(old,x)
    assert float(difference(n)) == pytest.approx(eps*n[2]*n[3]/n.sum(),abs=1e-13)
    expected=np.full(18,-eps*n[2]*n[3]/n.sum()**2)
    expected[2]+=eps*n[3]/n.sum();expected[3]+=eps*n[2]/n.sum()
    np.testing.assert_allclose(jax.grad(difference)(n),expected,atol=1e-12)
    assert float(difference(n*1e24))/1e24 == pytest.approx(float(difference(n)),abs=1e-13)
    hessian=np.asarray(jax.hessian(difference)(n))
    np.testing.assert_allclose(hessian,hessian.T,atol=1e-10)
    x=n/n.sum()
    interval=M.associated_excess(t,np.asarray(new.host.host.dry_model.interaction_K),
        np.asarray(new.host.epsilon),new.additional_matrix,[_Interval(v,v) for v in x[1:]])
    assert interval.lo <= float(new.gex_RT(t,2.7e7,x)) <= interval.hi
    upper=np.array([1.,.02,.01,.04,.02,.002,.00002,.0001,.12,.00001,
                    .003,.00002,.0001,.005,.00001,.000001,.002,.000001])
    lower=np.r_[.75,np.zeros(17)]
    # A negative certificate remains negative; the previous positive bound
    # is not attached to the altered scalar.
    bound=M.associated_curvature_lower_bound(new,t,lower,upper)
    assert bound < 0


def test_fixed_point_curvature_interval_distinguishes_nonconvexity_from_loose_bound():
    model,_=M.make_associated_model(2173.15,hydrogen_oxygen_model="schenck1961_abstract")
    x=np.full(18,1e-7);x[1:5]=[.001,.009,.039,.001];x[8]=.05;x[0]=1-x[1:].sum()
    variables=[M._SecondOrder(_Interval(v,v),i,dimension=17) for i,v in enumerate(x[1:])]
    value=M.associated_excess(2173.15,np.asarray(model.host.host.dry_model.interaction_K),
        np.asarray(model.host.epsilon),model.additional_matrix,variables)
    bound=(value.hessian[1][1]+value.hessian[2][2]-2*value.hessian[1][2]
           +M._interval(x[2]).reciprocal()+M._interval(x[3]).reciprocal())
    direction=np.zeros(18);direction[2]=1;direction[3]=-1
    energy=lambda s:total_solution_gibbs_RT(model,2173.15,2.7e7,x+s*direction,jnp.zeros(18))
    numeric=float(jax.grad(jax.grad(energy))(0.))
    assert bound.lo <= numeric <= bound.hi < 0
