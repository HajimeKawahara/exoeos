"""Finite native compatibility checks; these are not a domain certificate."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy

PARAM = Path('/tmp/stage2-rhm-parameters.json')
ARCHIVE = Path('/tmp/stage2-stability-supported-20260927/results/m2_stability_search/20260927/explicit_coordinates/phase_26_rhm-oxide.json')
raw = json.loads(PARAM.read_text())
model = raw['models']['rhm-oxide']
standard = json.load(open('/tmp/stage2-solid-standards/rhm-oxide.json'))
t, p = standard['T_K'], standard['P_bar']
R = raw['native_entropy_R_J_mol_K']
mu = np.asarray(standard['mu0_J_mol'])[[0,1,2,4]]
a = 17477 + .010758*(p-1)
b = 3189 + .035089*(p-1)
assert R*t >= a and b >= 0
refs = np.asarray([a-2*R*t*np.log(2), -2*R*t*.0730205*np.log(2), a-2*R*t*np.log(2), -2*R*t*.0730205*np.log(2)])

terms = {}
for field, factor in [('H_polynomial',1),('S_polynomial',-t),('V_polynomial',p-1)]:
 for coefficient,power in model[field]:
  key=tuple(power)
  terms[key]=terms.get(key,0)+coefficient*factor
powers=np.asarray(list(terms),dtype=int)
coefficients=np.asarray(list(terms.values()))
site_matrix=np.zeros((len(model['entropy_sites']),5))
site_constants=np.zeros(len(model['entropy_sites']))
site_coeffs=np.asarray([q['multiplicity']*R*t for q in model['entropy_sites']])
for i,q in enumerate(model['entropy_sites']):
 for coefficient,power in q['polynomial']:
  if sum(power)==0: site_constants[i]+=coefficient
  else:
   assert sum(power)==1
   site_matrix[i,power.index(1)]+=coefficient


def site_energy(v):
 vals=site_matrix@v+site_constants
 if vals.min() < -1e-12: return np.inf
 vals=np.maximum(vals,0)
 return coefficients@np.prod(v**powers,axis=1) + site_coeffs@xlogy(vals,vals)

archive=json.loads(ARCHIVE.read_text())['phases'][0]
trials=archive['trials'] + [archive['best_fresh_trial']]
records=[]
for i,trial in enumerate(trials):
 native=trial.get('provider_candidate')
 if not native or 'gibbs_J' not in native: continue
 x=np.asarray(trial['coordinates'])
 x=x/x.sum()
 r0,r1,e=x[2],x[0],x[3]
 def coordinates(q):
  return np.array([r0*(1+q[0])/2,r0*(1-q[0])/2,r1*(1+q[1])/2,r1*(1-q[1])/2,e])
 def objective(q): return site_energy(coordinates(q))/(R*t)
 choices=[minimize(objective,start,bounds=[[-1,1],[-1,1]],method='L-BFGS-B',
                   options={'ftol':1e-14,'gtol':1e-9,'maxiter':200})
          for start in [[0,0],[.8,.8],[-.8,.8]]]
 best=min(choices,key=lambda z:z.fun)
 model_mix=best.fun*R*t-x@refs
 native_moles=sum(native['native_endmember_moles'])
 native_mix=native['gibbs_J']/native_moles-x@mu
 records.append({'archive_trial_index':i,'fractions':x.tolist(),
  'native_mixing_gibbs_J_mol':float(native_mix),'declared_locally_minimized_mixing_gibbs_J_mol':float(model_mix),
  'difference_J_mol':float(model_mix-native_mix),'order_parameters':(best.x*np.array([r0,r1])).tolist(),
  'finite_optimizer_success':bool(best.success)})
result={'scope':'Finite archived native compatibility only; local order minimization and finite probes are not a formal global bound.',
 'parameter_sha256':hashlib.sha256(PARAM.read_bytes()).hexdigest(),
 'archive_sha256':hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
 'T_K':t,'P_bar':p,'pure_reference_J_mol':refs.tolist(),
 'high_temperature_identity_margin_J_mol':R*t-a,
 'source_symbolic_checks':['Every source H coefficient is expanded after the exact Mn-free site substitution.',
  'Every logarithmic coefficient in source S is reconstructed exactly before dropping zero Mn occupations.',
  'H is affine in source pressure; V is its exact symbolic derivative.',
  'All eight pinned SRO spline ordinates are exactly equal rational numbers.'],
 'finite_comparison_count':len(records),'maximum_absolute_difference_J_mol':max(abs(q['difference_J_mol']) for q in records),
 'records':records}
Path('/tmp/stage2-rhm-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
