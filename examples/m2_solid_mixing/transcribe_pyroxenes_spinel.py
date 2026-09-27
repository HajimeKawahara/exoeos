"""Transcribe pinned pyroxene and spinel site expressions; requires SymPy."""

import argparse
import json
import re
import hashlib
from pathlib import Path
from transcribe_parameters import Macros
import sympy as sp
import numpy as np
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-directory', required=True)
parser.add_argument('--native-standards-directory', required=True)
parser.add_argument('--host-properties', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
root = Path(args.source_directory)
v = sp.symbols('v0:8')
r = sp.symbols('r0:6')
s = sp.symbols('s0:3')
t, p = sp.symbols('t p')
R = sp.Symbol('R')
host = json.loads(Path(args.host_properties).read_text())


def poly(expr, vs):
    return [[float(c), list(k)] for k, c in sp.Poly(sp.expand(expr), *vs).terms() if c != 0]


def safe(expr):
    return expr.replace(sp.Function('SQUARE'), lambda a: a*a)


def entropy_terms(expr):
    # Return source entropy/R as linear site terms and its remaining polynomial.
    terms = []
    for node in expr.atoms(sp.log):
        if not node.args[0].free_symbols:
            continue
        factor = sp.expand(expr).coeff(node)
        occupation = node.args[0]
        mult = sp.simplify(-factor/sp.Rational('8.3143')/occupation)
        assert not mult.free_symbols, (node, mult)
        terms.append((mult, occupation))
    residual = expr.xreplace(
        {node: 0 for node in expr.atoms(sp.log) if node.args[0].free_symbols})
    return terms, sp.expand(residual)


def build(phase, source, vs, subs, sites, emap, extra_constraints, bounds, clino=True):
    m = Macros(source, root, clino)
    # Handles static const as well as calibration declarations with identical defaults.
    txt = re.sub(r'/\*.*?\*/', '', m.data, flags=re.S).replace('\\\n', ' ')
    m.defs.update({q.group(1): q.group(2) for q in re.finditer(
        r'static (?:const )?double\s+(\w+)\s*=\s*([^;]+);', txt)})
    H = safe(m.get('H')).subs(subs)
    S = safe(m.get('S')).subs(subs)
    for name, expr in sites.items():
        S = S.subs(sp.Symbol(name), expr)
    terms, Smisc = entropy_terms(S)
    V = (r[2]*r[3]*(m.get('WV1')*r[3]+m.get('WV2')*r[2])
         ).subs(subs) if phase == 'spinel' else m.get('V').subs(subs)
    assert not (H.free_symbols | Smisc.free_symbols | V.free_symbols) - \
        set(vs)-{R}, (H.free_symbols, Smisc.free_symbols, V.free_symbols)
    Smisc = Smisc.subs(R, sp.Rational('8.3143'))
    native = json.loads(
        (Path(args.native_standards_directory)/(phase+'.json')).read_text())
    matrix = np.array(native['native_basis']['native_endmember_oxide_mass_g_per_mol']
                      )/np.array(host['oxide_molar_masses_g_mol'])[:, None]
    exact = np.round(2*matrix)/2
    assert np.max(abs(exact-matrix)) < 1e-8
    model = {'phase': phase, 'expression_id': 'magma_published_'+phase+'_v1', 'source_file': source+'.c', 'source_sha256': hashlib.sha256(m.path.read_bytes()).hexdigest(), 'coordinate_order': [str(x) for x in vs], 'coordinate_bounds': bounds, 'endmember_names': (['chromite', 'hercynite', 'magnetite', 'spinel', 'ulvospinel'] if phase == 'spinel' else ['diopside', 'clinoenstatite', 'hedenbergite', 'alumino-buffonite', 'buffonite', 'essenite', 'jadeite']), 'native_endmember_indices': list(
        range(len(emap))), 'endmember_polynomials': [poly(x, vs) for x in emap], 'H_polynomial': poly(H, vs), 'S_polynomial': poly(Smisc, vs), 'V_polynomial': poly(V, vs), 'entropy_sites': [{'multiplicity': float(c), 'polynomial': poly(x, vs)} for c, x in terms], 'nonnegative_polynomials': [poly(x, vs) for x in extra_constraints], 'required_absent_elements': [], 'boundary_policy': 'Continuous zero site entropy', 'barrier': None, 'endmember_oxide_moles': exact.T.tolist()}
    return model, m


a, b, c, d, e, f, g, h = v
# Pyroxenes: M1 Fe/Al/Fe3/Ti; M2 Fe/Mg/Na; tetrahedral Al.
subs = {r[0]: a+e, r[1]: 2*h-c+g/2, r[2]: 2*d-2*h+c-g/2,
        r[3]: b+c-g/2, r[4]: g, r[5]: e+f, s[0]: c-b, s[1]: e-f}
site_r = {'xfe2m1': r[0]-(r[5]+s[1])/2, 'xal3m1': (2*r[3]+r[4]-2*s[0])/4, 'xfe3m1': (2*r[3]+r[4]+2*s[0])/4, 'xti4m1': (r[1]+r[2])/2, 'xmg2m1': 1-r[0]-r[3]-(r[1]+r[2]+r[4]-r[5]-s[1])/2, 'xfe2m2': (
    r[5]+s[1])/2, 'xmg2m2': (r[5]-s[1])/2, 'xna1m2': r[4], 'xca2m2': 1-r[4]-r[5], 'xal3tet': (4*r[1]+2*r[3]-r[4]+2*s[0])/8, 'xfe3tet': (4*r[2]+2*r[3]-r[4]-2*s[0])/8, 'xsi4tet': (4-2*r[1]-2*r[2]-2*r[3]+r[4])/4}
sites = {k: sp.expand(x.subs(subs)) for k, x in site_r.items()}
emap = [1-r[0]-r[1]-r[2]-r[3]-r[4]/2-r[5], r[5],
        r[0], r[1]-r[4]/2, r[2]+r[4]/2, r[3]-r[4]/2, r[4]]
constraints = [r[0], 2-r[0], r[4], 1-r[4], r[1]+r[3], 2-r[1]-r[3], r[2]+r[3], 2-r[2]-r[3], r[1]+r[2], 1-r[1]-r[2], r[4]+r[5], 1-r[4]-r[5], 1-r[0]-r[3]+r[5] -
               (r[1]+r[2]+r[4])/2, 1+r[0]+r[3]-r[5]+(r[1]+r[2]+r[4])/2, 1-r[1]-r[2]-r[3]+r[4]/2, r[1]+r[2]+r[3]-r[4]/2, r[5], r[3]+r[4]/2, (r[1]+r[2])/2+r[3]-r[4]/2, 1-r[1]-r[2]-r[3]-r[4]/2]
models = {}
for phase in ['clinopyroxene', 'orthopyroxene']:
    model, m = build(phase, phase, v, subs, sites, [x.subs(subs) for x in emap], [
                     x.subs(subs) for x in constraints], [[0, 1]]*8, clino=phase == 'clinopyroxene')
    # pureOrder and purePyx explicitly pin clino=TRUE for both structural phases.
    pure_m = Macros(phase, root, True)
    refs = []
    for j, key in enumerate(['DI', 'EN', 'HD', 'CA', 'CF', 'ES', 'JD']):
        expr = safe(pure_m.get(key+'_G'))
        u = sp.Symbol('s')
        terms, misc = entropy_terms(expr/(-t)) if key == 'ES' else ([], 0)
        if key == 'ES':
            # The two (1+-s) log terms are 2 times a normalized binary entropy after s=2u-1.
            misc = expr.xreplace({node: 0 for node in expr.atoms(
                sp.log) if node.args[0].free_symbols})+2*R*t*sp.log(2)
            expr = misc.subs(u, 2*v[0]-1)
        else:
            expr = expr
        expr = sp.expand(expr.subs(R, sp.Rational('8.3143')))
        assert not expr.free_symbols - \
            {t, p, v[0]}, (phase, key, expr.free_symbols)
        refs.append({'endmember_index': j, 'coordinate_bounds': [[0, 1]], 'H_polynomial': poly(expr.subs({t: 0, p: 1}), (v[0],)), 'S_polynomial': poly(-sp.diff(expr, t), (v[0],)), 'V_polynomial': poly(
            sp.diff(expr, p), (v[0],)), 'entropy_sites': [{'multiplicity': 2., 'polynomial': poly(x, (v[0],))} for x in [v[0], 1-v[0]]] if key == 'ES' else []})
    model['pure_reference_models'] = refs
    models[phase] = model
# Spinel, using independent tetrahedral Mg/Al/Fe3 and octahedral Mg/Al/Fe3/Cr.
vs = v[:7]
subs = {r[0]: a+2*d, r[1]: g, r[2]: 1-g -
        (c+2*f)/2-b/2-e, r[3]: (c+2*f)/2, s[0]: a-2*d, s[1]: e-b/2, s[2]: f-c/2}
site_r = {'xmg2tet': (r[0]+s[0])/2, 'xfe2tet': r[2]-r[0]/2-s[0]/2+s[1]+r[1]+s[2], 'xal3tet': 1-r[1]-r[2]-r[3]-s[1], 'xfe3tet': r[3]-s[2], 'xmg2oct': (r[0]-s[0])/4,
          'xfe2oct': (2-r[0]+s[0]-2*s[1]-2*r[1]-2*s[2])/4, 'xal3oct': (1-r[1]-r[2]-r[3]+s[1])/2, 'xfe3oct': (r[3]+s[2])/2, 'xcr3oct': r[1], 'xti4oct': r[2]/2}
sites = {k: sp.expand(x.subs(subs)) for k, x in site_r.items()}
emap = [r[1], 1-r[0]-r[1]-r[2]-r[3], r[3], r[0], r[2]]
model, m = build('spinel', 'spinel', vs, subs, sites, [x.subs(subs) for x in emap], [(1+r[2]-r[0]).subs(subs), (1-r[1]-r[2]-r[3]).subs(
    subs), (r[0]+r[1]+r[2]+r[3]).subs(subs)]+[sp.expand(x.subs(subs)) for x in [1-r[1], 1-r[2], 1-r[3]]], [[0, 1]]*7)
refs = []
for j, key in enumerate(['CR', 'HC', 'MT', 'SP', 'UV']):
    # All pure-order coordinates are u in [0,1], with source spinel s0=2u-1.
    srcs = {s[0]: 2*v[0]-1, s[1]: v[0], s[2]: v[0]}
    H = safe(m.get(key+'_H')).subs(srcs)
    S = safe(m.get(key+'_S')).subs(srcs)
    if key in ['HC', 'MT', 'SP']:
        # Every pure binary site has these four occupations. MT also has ss4*u.
        extra = m.get('ss4')*v[0] if key == 'MT' else sp.Integer(0)
        esites = [(1, v[0]), (1, 1-v[0]), (2, (1-v[0])/2), (2, (1+v[0])/2)]
        Smisc = extra
    else:
        esites = []
        Smisc = S
    refs.append({'endmember_index': j, 'coordinate_bounds': [[0, 1]], 'H_polynomial': poly(H, (v[0],)), 'S_polynomial': poly(Smisc.subs(R, sp.Rational(
        '8.3143')), (v[0],)), 'V_polynomial': [], 'entropy_sites': [{'multiplicity': float(c), 'polynomial': poly(x, (v[0],))} for c, x in esites]})
model['pure_reference_models'] = refs
models['spinel'] = model
Path(args.output).write_text(json.dumps(models, indent=2)+'\n')
for k, x in models.items():
    print(k, len(x['H_polynomial']), len(
        x['S_polynomial']), len(x['entropy_sites']))
