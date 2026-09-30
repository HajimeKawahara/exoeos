"""Independent finite compatibility review of the full-site prototype models.

All optimization here is numerical evidence, not a global-bound proof.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar
from scipy.special import xlogy

PATH=Path('/tmp/stage2-pyx-spinel-models.json')
MODELS=json.loads(PATH.read_text())
R=8.3143
ROOT=Path('/tmp/stage2-stability-supported-20260927/results/m2_stability_search/20260927/explicit_coordinates')


def sparse(terms,dimensions):
    return (np.asarray([q[0] for q in terms]),np.asarray([q[1] for q in terms],int).reshape(-1,dimensions))


def linear(row,dimensions):
    constant=0.;coeff=np.zeros(dimensions)
    for c,power in row:
        if sum(power)==0:constant+=c
        else:
            assert sum(power)==1
            coeff[power.index(1)]+=c
    return constant,coeff


def make_energy(model,T,P):
    n=len(model['coordinate_bounds'])
    terms={}
    for field,factor in [('H_polynomial',1.),('S_polynomial',-T),('V_polynomial',P-1)]:
        for c,power in model[field]:
            power=tuple(power)
            terms[power]=terms.get(power,0)+factor*c/(R*T)
    co,po=sparse([[c,list(p)] for p,c in terms.items()],n)
    linear_sites=[linear(s['polynomial'],n) for s in model['entropy_sites']]
    site_c=np.asarray([s['multiplicity'] for s in model['entropy_sites']])
    site_b=np.asarray([q[0] for q in linear_sites])
    site_A=np.asarray([q[1] for q in linear_sites]).reshape(-1,n)
    def function(v,gradient=False):
        val=co@np.prod(v**po,axis=1) if len(co) else 0.
        occupations=site_A@v+site_b
        if occupations.min(initial=0)<-1e-8 or occupations.max(initial=0)>1+1e-8:
            return (1e3+np.sum(np.minimum(occupations,0)**2),np.zeros(n)) if gradient else 1e3+np.sum(np.minimum(occupations,0)**2)
        q=np.clip(occupations,0,1)
        val+=site_c@xlogy(q,q)
        if not gradient:return float(val)
        grad=[]
        for j in range(n):
            powers=po.copy();powers[:,j]=np.maximum(powers[:,j]-1,0)
            grad.append((co*po[:,j])@np.prod(v**powers,axis=1) if len(co) else 0.)
        grad=np.asarray(grad)+site_A.T@(site_c*(1+np.log(np.maximum(q,1e-35))))
        return float(val),grad
    constraints=[]
    for const,row in linear_sites:
        constraints.extend([(const,row),(1-const,-row)])
    for row in model.get('nonnegative_polynomials',[]):constraints.append(linear(row,n))
    for i,(lower,upper) in enumerate(model['coordinate_bounds']):
        row=np.zeros(n);row[i]=1
        constraints.extend([(-lower,row),(upper,-row)])
    return function,constraints


def state_map(phase,x,s):
    if phase=='spinel':
        r0,r1,r2,r3=x[3],x[0],x[4],x[2]
        return np.array([(r0+s[0])/2,1-r1-r2-r3-s[1],r3-s[2],
                         (r0-s[0])/4,(1-r1-r2-r3+s[1])/2,(r3+s[2])/2,r1])
    r0,r1,r2,r3,r4,r5=x[2],x[3]+x[6]/2,x[4]-x[6]/2,x[5]+x[6]/2,x[6],x[1]
    return np.array([r0-(r5+s[1])/2,(2*r3+r4-2*s[0])/4,(2*r3+r4+2*s[0])/4,
                     (r1+r2)/2,(r5+s[1])/2,(r5-s[1])/2,r4,
                     (4*r1+2*r3-r4+2*s[0])/8])


def minimize_order(phase,model,x,energy,constraints):
    ns=3 if phase=='spinel' else 2
    if phase=='spinel' and np.count_nonzero(x>1e-14)==1:
        j=int(np.argmax(x))
        def pure_order(u):
            return np.array([2*u-1,u,0.]) if j==3 else np.array([0.,u,0.]) if j==1 else np.array([0.,0.,u]) if j==2 else np.zeros(3)
        opt=minimize_scalar(lambda u:energy(state_map(phase,x,pure_order(u))),bounds=(0,1),method='bounded',options={'xatol':1e-14})
        val,u=min([(float(opt.fun),float(opt.x))]+[(energy(state_map(phase,x,pure_order(u))),u) for u in [0.,1.]])
        coords=state_map(phase,x,pure_order(u))
        return {'status':'ok','value_rt':val,'coordinates':coords.tolist(),'order':pure_order(u).tolist(),'min_constraint':min(c+a@coords for c,a in constraints),'success':bool(opt.success)}
    base=state_map(phase,x,np.zeros(ns))
    matrix=np.stack([state_map(phase,x,np.eye(ns)[j])-base for j in range(ns)],axis=1)
    A=np.stack([row@matrix for c,row in constraints]);b=np.array([c+row@base for c,row in constraints])
    # Determine any fixed order coordinates on boundary compositions. These
    # tiny LPs only establish numerical optimizer bounds; they are not evidence
    # for any global Gibbs-energy certificate.
    ranges=[]
    for j in range(ns):
        objective=np.eye(ns)[j]
        low=linprog(objective,A_ub=-A,b_ub=b,bounds=[(None,None)]*ns,method='highs')
        high=linprog(-objective,A_ub=-A,b_ub=b,bounds=[(None,None)]*ns,method='highs')
        if not low.success or not high.success:
            return {'status':'numerical_domain_infeasible','lp_messages':[low.message,high.message]}
        ranges.append([low.fun,-high.fun])
    lower,upper=np.asarray(ranges).T
    free=(upper-lower)>1e-11
    widths=(upper-lower)[free]
    fixed=lower.copy();fixed[~free]=(lower[~free]+upper[~free])/2
    vbase=base+matrix@fixed
    vmatrix=matrix[:,free]*widths
    if not free.any():
        return {'status':'ok','value_rt':energy(vbase),'coordinates':vbase.tolist(),
                'order':fixed.tolist(),'min_constraint':float((A@fixed+b).min()),'success':True}
    Aq=A[:,free]*widths;bq=A@fixed+b
    keep=np.linalg.norm(Aq,axis=1)>1e-13
    assert bq[~keep].min(initial=0)>=-1e-10
    Aq=Aq[keep];bq=bq[keep]
    # Chebyshev center and two LP vertices are deterministic feasible starts.
    center=linprog(np.r_[np.zeros(sum(free)),-1.],A_ub=np.c_[-Aq,np.linalg.norm(Aq,axis=1)],
                  b_ub=bq,bounds=[(0,1)]*sum(free)+[(0,None)],method='highs')
    seed=center.x[:-1] if center.success else np.full(sum(free),.5)
    starts=[seed]
    for sign in [-1,1]:
        corner=linprog(sign*np.arange(1,sum(free)+1),A_ub=-Aq,b_ub=bq,bounds=[(0,1)]*sum(free),method='highs')
        if corner.success:starts.append(.8*seed+.2*corner.x)
    def fun(q):
        value,gradient=energy(vbase+vmatrix@q,True)
        return value,vmatrix.T@gradient
    results=[]
    for start in starts:
        opt=minimize(fun,start,jac=True,method='SLSQP',bounds=[(0,1)]*sum(free),
                     constraints={'type':'ineq','fun':lambda q:Aq@q+bq,'jac':lambda q:Aq},
                     options={'ftol':1e-13,'maxiter':300})
        if (Aq@opt.x+bq).min(initial=0)>=-1e-8:results.append(opt)
    best=min(results,key=lambda z:z.fun)
    # A trace endmember creates an order interval only ~1e-8 wide. Its scaled
    # SLSQP gradient can be too small, so refine each coordinate by a bounded
    # scalar solve on the exact conditional feasible interval.
    refined=best.x.copy()
    for sweep in range(10):
        previous=fun(refined)[0]
        for j in range(len(refined)):
            rem=Aq@refined+bq-Aq[:,j]*refined[j]
            lo,hi=0.,1.
            for slope,intercept in zip(Aq[:,j],rem):
                if slope>1e-20:lo=max(lo,-intercept/slope)
                elif slope < -1e-20:hi=min(hi,-intercept/slope)
            if hi-lo<1e-13:continue
            def scalar(z):
                candidate=refined.copy();candidate[j]=z
                return fun(candidate)[0]
            choice=minimize_scalar(scalar,bounds=(lo,hi),method='bounded',options={'xatol':1e-14})
            candidates=[(float(choice.fun),float(choice.x)),(scalar(lo),lo),(scalar(hi),hi),(scalar(refined[j]),refined[j])]
            refined[j]=min(candidates)[1]
        if previous-fun(refined)[0]<1e-14:break
    if fun(refined)[0]<best.fun:
        best.x=refined;best.fun=fun(refined)[0]
    order=fixed.copy();order[free]+=widths*best.x
    return {'status':'ok','value_rt':float(best.fun),'coordinates':(vbase+vmatrix@best.x).tolist(),
            'order':order.tolist(),'min_constraint':float((A@order+b).min()),'success':bool(best.success)}


result={'scope':'Numerical finite native compatibility after the Opx pure-reference clino=TRUE correction; this is not a global stability or native-uniform-error proof.',
        'parameter_sha256':hashlib.sha256(PATH.read_bytes()).hexdigest(),'phases':[]}
for phase,model in MODELS.items():
    native=json.load(open('/tmp/stage2-solid-standards/'+phase+'.json'))
    T,P=native['T_K'],native['P_bar']
    mu=np.asarray(native['mu0_J_mol'])
    energy,constraints=make_energy(model,T,P)
    refs=[];reference_minima=[]
    for ref in model['pure_reference_models']:
        fun,_=make_energy(ref,T,P)
        opt=minimize_scalar(lambda u:fun(np.array([u])),bounds=(0,1),method='bounded',options={'xatol':1e-14})
        choices=[(float(opt.fun),float(opt.x)),(fun(np.array([0.])),0.),(fun(np.array([1.])),1.)]
        val,coord=min(choices)
        refs.append(val*R*T)
        reference_minima.append({'endmember_index':ref['endmember_index'],'gibbs_J_mol':val*R*T,'coordinate':coord})
    path=ROOT/('phase_'+{'clinopyroxene':'05','orthopyroxene':'04','spinel':'25'}[phase]+'_'+phase+'.json')
    archive=json.loads(path.read_text())['phases'][0]
    records=[]
    for i,trial in enumerate(archive['trials']+[archive['best_fresh_trial']]):
        candidate=trial.get('provider_candidate')
        if not candidate or candidate.get('gibbs_J') is None:continue
        moles=np.array(candidate['native_endmember_moles']);x=moles/moles.sum()
        opt=minimize_order(phase,model,x,energy,constraints)
        if opt['status']!='ok':
            records.append({'archive_trial_index':i,'fractions':x.tolist(),**opt});continue
        mix=opt['value_rt']*R*T-x@refs
        native_mix=candidate['gibbs_J']/moles.sum()-x@mu
        records.append({'archive_trial_index':i,'fractions':x.tolist(),'native_mixing_J_mol':float(native_mix),
                        'declared_mixing_J_mol':float(mix),'difference_J_mol':float(mix-native_mix),**opt})
    compared=[q for q in records if 'difference_J_mol' in q]
    summary={'phase':phase,'T_K':T,'P_bar':P,'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'comparison_count':len(compared),'reference_minima':reference_minima,
             'maximum_absolute_difference_J_mol':max(abs(q['difference_J_mol']) for q in compared),
             'optimizer_failure_count':sum(not q['success'] for q in compared),'records':records}
    print(json.dumps({k:v for k,v in summary.items() if k not in ['records','reference_minima']}),flush=True)
    result['phases'].append(summary)
    Path('/tmp/stage2-pyx-spinel-validation.json').write_text(json.dumps(result,indent=2)+'\n')
