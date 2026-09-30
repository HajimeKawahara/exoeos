import hashlib,json,sys,numpy as np
from pathlib import Path
from m2_expanded_source import build_expanded_bse_problem
original=Path('/tmp/stage2-final-reclosed-h1e24-sakao-oh-16/closure.json')
closed=json.loads(original.read_text())
source=closed['runs'][0]['roots'][0]['source']
record,budget,callbacks,initial,metadata=build_expanded_bse_problem(
 Path(closed['arguments']['inventory']),Path('/tmp/stage2-liquid-derivative-eos-20260927'),
 Path(closed['arguments']['runtime']),closed['arguments']['worker_python'],
 temperature_k=source['temperature_K'],pressure_bar=source['pressure_bar'],
 scenario=closed['provider_scenario']['values'],gas_model=source['gas_model'],liquid_model='published_water',initialization='canonical')
old_names=[name for values in source['source_record']['phases'].values() for name in values]
old_amounts=dict(zip(old_names,source['source_result']['component_amounts_mol']))
names=record['phases']['silicate']; n=np.array([old_amounts[name] for name in names])
t,p=source['temperature_K'],source['pressure_bar']
state=callbacks['silicate'](t,p,n)
energy,gradient=callbacks['silicate'].energy_value_and_grad_rt(t,p,n)
present=n>0
result=dict(model_id=metadata['host_ledger']['model_id'],scope='Single fixed old OH composition; no new equilibrium or pressure closure',
 source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),T_K=t,P_bar=p,
 component_order=names,component_amounts_mol=n.tolist(),gibbs_RT=state.gibbs_rt,
 mu_RT=[float(v) if a else None for v,a in zip(state.mu_rt,present)],
 independent_energy_RT=energy,maximum_active_derivative_difference_RT=float(np.max(np.abs(gradient[present]-state.mu_rt[present]))),
 relative_Euler_residual=float((n[present]@state.mu_rt[present]-state.gibbs_rt)/abs(state.gibbs_rt)),
 provider_ledger=metadata['host_ledger'])
Path('/tmp/stage2-water-native-point-20260928.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print({key:result[key] for key in ('model_id','maximum_active_derivative_difference_RT','relative_Euler_residual')})
