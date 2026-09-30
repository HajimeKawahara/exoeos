import sys,json,pathlib,importlib.util,hashlib
import numpy as np
phase=sys.argv[1]
runtime=pathlib.Path('/tmp/exoeos-pr3-melts/runtime/alphamelts-py-2.3.2-ubuntu_22_04-x86_64')
spec=importlib.util.spec_from_file_location('evaluator','/tmp/stage2-candidate-eos-20260927/examples/melts_liquid_evaluator.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev);ev.check_runtime(runtime)
sys.path.insert(0,str(runtime));from meltsdynamic import MELTSdynamic
model=MELTSdynamic(1);en=model.engine;en.temperature=1900.;en.pressure=267.20283416157343;en.setSystemProperties('Log fO2 Path','None')
basis=ev.molar_candidate_properties(model,phase);matrix=np.asarray(basis['native_endmember_oxide_mass_g_per_mol']);ox=matrix@np.full(matrix.shape[1],1/matrix.shape[1]);en.calcEndMemberProperties(phase,ox.tolist())
result={'phase':phase,'T_K':2173.15,'P_bar':en.pressure,'native_basis':basis,'endmember_formulas':model.endMemberFormulas[phase], 'native_status_failed':en.status.failed,'request_oxide_mass_g':ox.tolist(), 'mu0_J_mol':np.asarray(en.mu0[phase]).tolist(),'x':np.asarray(en.X[phase]).tolist(), 'source_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
pathlib.Path('/tmp/stage2-solid-standards/'+phase+'.json').write_text(json.dumps(result,indent=2)+'\n')
