import hashlib, importlib.util,json,sys,time
from pathlib import Path
import numpy as np
root=Path('/tmp/stage2-hydrogen-oxygen-eos-20260928')
sys.path.insert(0,str(root/'src'))
p=root/'examples/m2_material/associate_reference.py'
spec=importlib.util.spec_from_file_location('ho_replay',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
lo=np.r_[.75,np.zeros(17)]
hi=np.array([1.,.02,.01,.04,.02,.002,.00002,.0001,.12,.00001,.003,.00002,.0001,.005,.00001,.000001,.002,.000001])
rows=[]
for policy in ['constant','enthalpic']:
 for selection in ['omitted','schenck1961_abstract']:
  model,receipt=m.make_associated_model(2173.15,temperature_policy=policy,hydrogen_oxygen_model=selection)
  bound=m.associated_curvature_lower_bound(model,2173.15,lo,hi)
  x=np.full(18,1e-7);x[1]=.001;x[2]=.009;x[3]=.039;x[4]=.001;x[8]=.05;x[0]=1-x[1:].sum()
  vals=[m._SecondOrder(m._Interval(v,v),i,dimension=17) for i,v in enumerate(x[1:])]
  excess=m.associated_excess(2173.15,np.asarray(model.host.host.dry_model.interaction_K),np.asarray(model.host.epsilon),model.additional_matrix,vals)
  direction_curvature=(excess.hessian[1][1]+excess.hessian[2][2]-2*excess.hessian[1][2]+m._interval(x[2]).reciprocal()+m._interval(x[3]).reciprocal())
  point_witness={'species_mole_fractions':x.tolist(),'direction':'dn_O=+1,dn_H=-1, other species held fixed; total species amount fixed','outward_curvature_interval_RT_per_species_mol':[direction_curvature.lo,direction_curvature.hi],'negative_curvature_proved':bool(direction_curvature.hi<0)}
  rows.append({'point_curvature_witness':point_witness,'selection':selection,'temperature_policy':policy,'interaction_receipt':receipt,'curvature_lower_bound_RT_per_species_mol':bound,'strict_positive_curvature_certified':bound>0})
source=m.HO_DATA_PATH
record={'record_id':'HO_contemporary_abstract_reconstruction_v1','temperature_K':2173.15,'lower':lo.tolist(),'upper':hi.tolist(),'cases':rows,'reference':json.loads(source.read_text()),'file_sha256':{str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [p,source]},'accepted_coupled_material_domain':None,'scope':'Declared scalar comparison. A negative interval curvature bound does not by itself prove a negative Hessian or a phase instability. No source equilibrium or pressure closure was run.'}
out=root/'examples/m2_material/validation/20260928_hydrogen_oxygen_curvature_v2'
out.mkdir(parents=True,exist_ok=False)
(out/'comparison.json').write_text(json.dumps(record,indent=2)+'\n')
print([(r['selection'],r['temperature_policy'],r['curvature_lower_bound_RT_per_species_mol']) for r in rows])
