"""Compare the independently lifted plagioclase equation with archived probes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import xlogy

parameter_path=Path('/tmp/stage2-plagioclase-parameters.json')
archive_path=Path('/tmp/stage2-stability-supported-20260927/results/m2_stability_search/20260927/explicit_coordinates/phase_14_plagioclase.json')
raw=json.loads(parameter_path.read_text())
model=raw['models']['plagioclase']
native=json.load(open('/tmp/stage2-solid-standards/plagioclase.json'))
archive=json.loads(archive_path.read_text())['phases'][0]
t,p,R=native['T_K'],native['P_bar'],raw['native_entropy_R_J_mol_K']
mu=np.asarray(native['mu0_J_mol'])

def poly(terms,v):
 return sum(coefficient*np.prod(np.asarray(v)**power) for coefficient,power in terms)

records=[]
for i,trial in enumerate(archive['trials']+[archive['best_fresh_trial']]):
 candidate=trial.get('provider_candidate')
 if not candidate or 'gibbs_J' not in candidate: continue
 moles=np.asarray(candidate['native_endmember_moles'])
 x=moles/moles.sum()
 v=x[:2]
 g=poly(model['H_polynomial'],v)-t*poly(model['S_polynomial'],v)+(p-1)*poly(model['V_polynomial'],v)
 g+=R*t*sum(s['multiplicity']*xlogy(poly(s['polynomial'],v),poly(s['polynomial'],v)) for s in model['entropy_sites'])
 native_mix=candidate['gibbs_J']/moles.sum()-x@mu
 records.append({'archive_trial_index':i,'fractions':x.tolist(),'native_mixing_gibbs_J_mol':float(native_mix),
  'declared_mixing_gibbs_J_mol':float(g),'difference_J_mol':float(g-native_mix)})
result={'scope':'Finite archived native compatibility only. Parameter values and polynomial topology came from symbolic instruction lifting; no finite-point fit was used.',
 'parameter_sha256':hashlib.sha256(parameter_path.read_bytes()).hexdigest(),
 'archive_sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),
 'T_K':t,'P_bar':p,'comparison_count':len(records),
 'maximum_absolute_difference_J_mol':max(abs(q['difference_J_mol']) for q in records),
 'records':records}
Path('/tmp/stage2-plagioclase-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
