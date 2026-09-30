"""Reuse saved reference potentials and one frozen OH state; no equilibrium."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'examples/m2_material'))
import sodium_reference as reference
import sodium_host_calibration as calibration


def potassium_combination(order, potentials):
    # KO0.5 = KAlSiO4 - 0.5 Al2O3 - SiO2; then subtract 0.5 FeO.
    return sum(coefficient*potentials[order.index(name)] for name,coefficient in
               [('kalsio4',1.),('al2o3',-.5),('sio2',-.75),('fe2sio4',-.25)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closure',type=Path,required=True)
    parser.add_argument('--physical-audit',type=Path,required=True)
    parser.add_argument('--reference-assessment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output path.')
    closure=json.loads(args.closure.read_text());root=closure['runs'][0]['roots'][0]
    source=root['source']; audit=json.loads(args.physical_audit.read_text())
    assert audit['source']['sha256']==hashlib.sha256(args.closure.read_bytes()).hexdigest()
    extraction=audit['formal_reaction_standards']['extraction'];state=extraction['native_state']
    assert state['T_K']==root['temperature_base_k'] and state['P_Pa']==root['pressure_base_pa']
    fe0=extraction['standards_rt']['Fe_metal']
    elements=source['source_internal_record']['elements']
    potentials=dict(zip(elements,source['source_internal_result']['elemental_potentials_rt']))
    inventories=dict(zip(root['elements'],root['element_amounts_mol']))
    iron_x=source['metal_selection']['metal_composition'][0]
    ln_gamma_fe=potentials['Fe']-fe0-math.log(iron_x)
    virtual_na=calibration.virtual_reference_rt(state['component_order'],state['mu0_RT'])+.5*fe0
    virtual_k=potassium_combination(state['component_order'],state['mu0_RT'])+.5*fe0
    gas=source['source_metadata']['standards']['common_gas']
    # Atomic gas hvector is zero in this retained FastChem convention. Record
    # the actual saved K gauge, not an independently substituted JANAF value.
    k_gas_zero=gas['element_gauge_rt'][gas['elements'].index('K')]
    saved=json.loads(args.reference_assessment.read_text())
    sodium=[]
    for row in saved['conditional_lower_envelopes']:
        if row['liquid_model']!='published':continue
        delta=row['delta_at_2173_15_K']
        fraction=math.exp(potentials['Na']-virtual_na-delta-ln_gamma_fe)
        sodium.append(dict(projection=row['projection'],temperature_policy=row['temperature_policy'],
            delta_rt=delta,standard_rt=virtual_na+delta,frozen_trace_mole_fraction=fraction,
            frozen_host_linear_amount_mol=root['metal_amount_mol']*fraction,
            frozen_trace_to_global_Na_inventory=root['metal_amount_mol']*fraction/inventories['Na']))
    data=json.loads(reference.DATA.read_text());metal=data['supplement_TableS4_cells']
    potassium=[]
    for row in saved['rows']:
        observed=row['reference'];name=observed['run'].replace('GGK','GGK-')
        continued=observed['run'] in ('GGK7','GGK8','GGK9')
        laser=reference._column(metal,33 if continued else 0,52 if continued else 24,61 if continued else 33,name)
        epma=reference._column(metal,33 if continued else 0,35 if continued else 2,49 if continued else 20,name)
        epma={k.split()[0]:v for k,v in epma.items()}
        xk=(observed['alloy_x_Na']*reference._value(laser['K'])/reference.ATOMIC_MASS['K']
            /(observed['metal_Na_mass_ppm']['value']/reference.ATOMIC_MASS['Na']))
        xsi=(observed['alloy_x_Fe']*reference._value(epma['Si'])/reference.ATOMIC_MASS['Si']
             /(reference._value(epma['Fe'])/reference.ATOMIC_MASS['Fe']))
        activity=next(v for v in data['supplement_TableS2_cells'] if v[0]==observed['run'])
        gfe=reference._value(activity[4]); gk=reference._value(activity[5])
        properties=row['models']['published']['properties'];order=properties['component_order']
        mix=potassium_combination(order,properties['mu_RT'])-potassium_combination(order,properties['mu0_RT'])
        ideal_delta=mix+.5*math.log(gfe*observed['alloy_x_Fe'])-math.log(xk)
        epsilon=10.42*1783./observed['temperature_K']
        for policy,ln_gamma_k in [('current_ideal_K',0.),('conditional_pseudoFe_linear_Si',math.log(gfe)+epsilon*xsi)]:
            delta=ideal_delta-ln_gamma_k
            # A separate constant dimensionless offset continuation to the OH
            # T/P. It does not assert temperature-independent exchange K or D.
            applied_ln_gamma=(0. if policy=='current_ideal_K' else
                              ln_gamma_fe+10.42*1783./root['temperature_base_k']*source['metal_selection']['metal_composition'][1])
            x_trace=math.exp(potentials['K']-(virtual_k+delta)-applied_ln_gamma)
            potassium.append(dict(run=observed['run'],projection=row['projection_label'],model=policy,
                reference_temperature_K=observed['temperature_K'],original_metal_K_mass_ppm=reference.observation(laser['K']),
                original_metal_x_K=xk,original_metal_x_Si=xsi,gamma_Fe_table=gfe,gamma_K_table=gk,
                table_ln_gamma_K_over_gamma_Fe=math.log(gk/gfe),
                fig_S7_linear_Henry_ln_gamma_K_over_gamma_Fe=epsilon*xsi,
                projected_liquid_virtual_mixing_rt=mix,equivalent_delta_RK_rt=delta,
                explicit_constant_delta_continuation_gas_offset_rt=virtual_k+delta-k_gas_zero,
                frozen_trace_mole_fraction=x_trace,
                frozen_trace_to_global_K_inventory=root['metal_amount_mol']*x_trace/inventories['K']))
    fits=[]
    for projection in sorted({x['projection'] for x in potassium}):
        for policy in ('current_ideal_K','conditional_pseudoFe_linear_Si'):
            values=[x for x in potassium if x['projection']==projection and x['model']==policy]
            matrix=np.array([[1.,1/x['reference_temperature_K']] for x in values])
            y=np.array([x['equivalent_delta_RK_rt'] for x in values]);coefficients=np.linalg.lstsq(matrix,y,rcond=None)[0]
            residual=y-matrix@coefficients
            fits.append(dict(projection=projection,model=policy,intercept_A_rt=float(coefficients[0]),
                inverse_temperature_B_K=float(coefficients[1]),unweighted_residual_rt=residual.tolist(),
                max_abs_residual_rt=float(max(abs(residual))),
                fitted_delta_at_2173_15_K=float(coefficients[0]+coefficients[1]/2173.15),
                interpretation='Descriptive proxy-host fit; no measured uncertainty or accepted K calibration.'))
    result=dict(source_closure_sha256=hashlib.sha256(args.closure.read_bytes()).hexdigest(),
        physical_audit_sha256=hashlib.sha256(args.physical_audit.read_bytes()).hexdigest(),
        reference_assessment_sha256=hashlib.sha256(args.reference_assessment.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        T_K=root['temperature_base_k'],P_Pa=root['pressure_base_pa'],
        preserved_Fe_metal_standard_rt=fe0,alloy_standard_shift_rt=extraction['alloy_standard_shift_rt'],
        virtual_Na_reference_rt=virtual_na,virtual_K_reference_rt=virtual_k,
        retained_K_gas_standard_rt=k_gas_zero,source_element_order=elements,
        source_elemental_potentials_rt=source['source_internal_result']['elemental_potentials_rt'],
        frozen_metal_amount_mol=root['metal_amount_mol'],source_metal_composition=source['metal_selection']['metal_composition'],
        frozen_ln_gamma_Fe=ln_gamma_fe,global_Na_inventory_mol=inventories['Na'],global_K_inventory_mol=inventories['K'],
        sodium=sodium,potassium=potassium,potassium_proxy_temperature_fits=fits,
        finite_source_equilibrium_executed=False,pressure_reclosed=False,physical_calibration_accepted=False,
        scope=['All source elemental potentials and non-Na/K reservoirs remain frozen.',
               'Trace amount is old metal particle moles times the infinite-dilution fraction; it is not a finite partition solution.',
               'S-free Na limits constrain only conditional projected hosts; Na-Si/C interactions remain unmeasured.',
               'K proxy silicate amounts differ from the measured K analysis; original metal K is retained explicitly.',
               'The K Figure S7 first-order Si interaction is a separate candidate scalar, not silently applied on top of Table S2 activity corrections.',
               'Constant delta continuation at native virtual standards is distinct from constant D or constant exchange K.'])
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')

if __name__=='__main__':main()
