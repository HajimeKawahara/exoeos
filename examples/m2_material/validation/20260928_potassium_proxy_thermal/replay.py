"""Declared exchange-slope continuation of saved ideal-K proxy constraints."""
from pathlib import Path
import hashlib
import json
import math
import time

started=time.perf_counter()
HERE=Path(__file__).resolve().parent
source=HERE.parent/'20260928_sodium_proxy_trace/assessment.json'
data=json.loads(source.read_text())
rows=[]
for row in data['potassium']:
    if row['model']!='current_ideal_K':continue
    shift=12530.*math.log(10.)*(1/data['T_K']-1/row['reference_temperature_K'])
    rows.append(dict(run=row['run'],projection=row['projection'],reference_temperature_K=row['reference_temperature_K'],
        constant_delta_gas_offset_rt=row['explicit_constant_delta_continuation_gas_offset_rt'],
        declared_thermal_standard_shift_rt=shift,
        continued_gas_offset_rt=row['explicit_constant_delta_continuation_gas_offset_rt']+shift,
        frozen_trace_mole_fraction=row['frozen_trace_mole_fraction']*math.exp(-shift),
        frozen_trace_to_global_K_inventory=row['frozen_trace_to_global_K_inventory']*math.exp(-shift)))
result=dict(assessment_id='conditional_potassium_proxy_exchange_slope_v1',
    source_assessment_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    primary_source='https://doi.org/10.1038/s41598-018-25505-6',
    declared_log10_K_slope_K=-12530.,original_fit='log10 K_K = 4.41(13) - 12530(235)/T',
    coefficient_uncertainty_propagated=False,T_K=data['T_K'],P_Pa=data['P_Pa'],rows=rows,
    finite_source_equilibrium_executed=False,physical_calibration_accepted=False,empirical_BSE_upper_bound=False,
    scope=['Only the central published FeS exchange slope is borrowed; the proxy constraint intercept is retained.',
           'Native virtual standards have their own T/P dependence; no equation identifies exchange K with a mass partition ratio.',
           'Silicate projection, reference alloy denominator, unmodeled Si/C activity differences, and low-pressure continuation remain explicit assumptions.',
           'The frozen trace response is not a finite metal/silicate allocation or pressure closure.'])
output=HERE/'assessment.json'
with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
receipt=dict(actual_exit_code=0,wall_seconds=time.perf_counter()-started,
             files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),output)})
with (HERE/'execution.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
