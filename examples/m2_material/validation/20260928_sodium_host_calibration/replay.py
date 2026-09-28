"""Evaluate declared Na proxy hosts; never modify source equilibrium models."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT/'examples'), str(ROOT/'examples/m2_material')]
import sodium_reference as reference
import sodium_host_calibration as calibration
import melts_liquid_evaluator as native
import melts_liquid_mixing as mixing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    started = time.perf_counter()
    data = json.loads(reference.DATA.read_text())
    provider = mixing.make_published_liquid_evaluator(native, runtime=args.runtime,
                                                     python_executable=args.python)
    constraints, rows = {}, []
    for row in reference.replay(data):
        if row['run'] not in ('GGK1', 'GGK2', 'GGK7', 'GGK8', 'GGK9'):
            continue
        activity = next(x for x in data['supplement_TableS2_cells'] if x[0] == row['run'])
        oxides = calibration.measured_oxide_masses(data, row)
        for method, ratio in (('remove_potassium', .25), ('remove_potassium', .5),
                              ('remove_potassium', .9), ('add_aluminum_silica', .9)):
            projected = calibration.projected_host(native, oxides, method=method,
                                                   potassium_aluminum_ratio=ratio)
            label = f'{method}_{ratio}'
            outputs = {}
            for label_model, evaluator in (('native', native), ('published', provider)):
                try:
                    properties = evaluator.evaluate_liquid(row['temperature_K'], 1e9,
                        projected['component_moles'], runtime=args.runtime,
                        python_executable=args.python)
                    result = calibration.henry_offset_constraint(properties,
                        alloy_x_na_upper=row['alloy_x_Na'], alloy_x_fe=row['alloy_x_Fe'],
                        gamma_fe=reference._value(activity[4]))
                    result.update(run=row['run'], temperature_K=row['temperature_K'])
                    constraints.setdefault((label, label_model), []).append(result)
                    outputs[label_model] = dict(status='available', properties=properties,
                                                conditional_constraint=result)
                except (ValueError, RuntimeError) as error:
                    outputs[label_model] = dict(status='unavailable', reason=str(error))
            rows.append(dict(run=row['run'], reference=row, projection=projected,
                             projection_label=label, models=outputs))
            print(row['run'], label, {k:v.get('conditional_constraint', {}).get('delta_lower_rt')
                                      for k,v in outputs.items()}, flush=True)
    envelopes = []
    for (projection, model), values in constraints.items():
        for policy, coefficient in (('constant_delta', 0.),
                                    ('published_exchange_slope', 14340.*math.log(10.))):
            result = calibration.lower_envelope(values, inverse_temperature_coefficient_K=coefficient)
            result.update(projection=projection, liquid_model=model, temperature_policy=policy,
                          constraint_count=len(values), temperature_continued_K=2173.15,
                          delta_at_2173_15_K=result['intercept_A_rt']+coefficient/2173.15)
            envelopes.append(result)
    files = [reference.DATA, Path(calibration.__file__), Path(native.__file__), Path(mixing.__file__),
             native.REFERENCE_PATH, mixing.PARAMETER_PATH, Path(__file__)]
    result = dict(command=sys.argv, wall_seconds=time.perf_counter()-started,
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  common_R_J_mol_K=native.COMMON_R, reference_pressure_Pa=1e9,
                  rows=rows, conditional_lower_envelopes=envelopes,
                  native_standard_state_receipts=provider.standard_state_receipts,
                  physical_domain_accepted=False, empirical_BSE_upper_bound=False,
                  absolute_Na_standard_automatically_adopted=False,
                  finite_source_equilibrium_executed=False,
                  policy=['Preserve measured Na/Fe amounts; change only declared proxy host oxides.',
                          'Use original alloy Na detection limits and calculated carbon denominator.',
                          'Adopt gamma_Na,Henry = gamma_Fe in the S-free continuation; Na-Si/C effects are not measured zeros.',
                          'The thermal slope is borrowed from the FeS exchange regression, not fitted to S-free censoring.',
                          'Pressure continuation of the virtual oxide standard is an explicit model choice; no low-pressure calibration.'])
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')

if __name__ == '__main__':
    main()
