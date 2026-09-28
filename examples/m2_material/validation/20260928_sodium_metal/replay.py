"""Actual provider scalar checks and a saved-OH sodium standard receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples/m2_material')]
import sodium_metal as sodium
from exoeos import total_solution_gibbs_RT,total_solution_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output path.')
    started=time.perf_counter()
    source=json.loads(args.physical_audit.read_text())
    extracted=source['formal_reaction_standards']['extraction']
    native=extracted['native_state'];T=native['T_K'];P=native['P_Pa']
    standard,receipt=sodium.sodium_standard_rt(T,P,extracted['standards_rt']['Fe_metal'],native,
        projection='remove_potassium_0.25',temperature_policy='published_exchange_slope')
    rows=[]
    for ho in ('omitted','schenck1961_abstract'):
        model,provenance=sodium.make_associated_model(T,hydrogen_oxygen_model=ho)
        x=np.array([1.,.001,.001,.02,.001,.0001,.000001,.00001,.004,.000001,
                    .0001,.000001,.00001,.0001,.000001,.0000001,.00001,.0000001,.001,.001])
        x[0]=1-x[1:].sum();model.validate_state(T,P,x)
        # Arbitrary other standards exercise the scalar algebra; only Na and Fe
        # below are the actual source-anchored recipe. This is not a source solve.
        standards=np.arange(20.)*.1-10.;standards[0]=receipt['fe_metal_standard_rt'];standards[-1]=standard
        energy=lambda n:total_solution_gibbs_RT(model,T,P,n,standards)
        state=total_solution_state(model,T,P,x,standards)
        gradient=np.asarray(jax.grad(energy)(x))
        gamma=np.asarray(state.mu_RT)-standards-np.log(x)
        euler=float(state.gibbs_RT)-float(x@np.asarray(state.mu_RT))
        parent=x[:19].copy();parent[0]+=x[-1]
        parent_value=float(total_solution_gibbs_RT(model.host,T,P,parent,standards[:19]))
        split=x[0]*np.log(x[0])+x[-1]*np.log(x[-1])-parent[0]*np.log(parent[0])
        independent=parent_value+split+x[-1]*(standards[-1]-standards[0])
        row=dict(hydrogen_oxygen_model=ho,model_receipt=provenance,
            component_fractions=x.tolist(),standard_potentials_rt=standards.tolist(),
            standard_scope='Synthetic other standards; source-anchored Fe and Na only.',
            gibbs_RT=float(state.gibbs_RT),maximum_ad_mu_difference_rt=float(np.max(abs(gradient-np.asarray(state.mu_RT)))),
            euler_residual_rt=euler,independent_parent_plus_split_residual_rt=float(state.gibbs_RT)-independent,
            ln_gamma_Na_minus_ln_gamma_Fe=float(gamma[-1]-gamma[0]))
        assert abs(euler)<1e-12 and row['maximum_ad_mu_difference_rt']<1e-12
        assert abs(row['independent_parent_plus_split_residual_rt'])<1e-12
        assert abs(row['ln_gamma_Na_minus_ln_gamma_Fe'])<1e-12
        rows.append(row)
    result=dict(assessment_id='finite_sodium_scalar_and_source_standard_v1',command=sys.argv,
        source_physical_audit_sha256=hashlib.sha256(args.physical_audit.read_bytes()).hexdigest(),
        original_closure_sha256=source['source']['sha256'],sodium_standard_receipt=receipt,
        scalar_checks=rows,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        files_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(sodium.__file__),Path(sodium._potassium().__file__),ROOT/'examples/m2_material/associate_reference.py']},
        actual_exit_code=0,wall_seconds=time.perf_counter()-started,
        new_equilibrium_or_pressure_closure=False,physical_material_acceptance=False)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')

if __name__=='__main__':main()
