"""Developer-only symbolic transcription; requires SymPy 1.14.0.

Run on the cited source revision and inspect the emitted source SHA256 values.
Only the committed numeric parameter file is loaded by the runtime provider.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import sympy as sp


class Macros:
    def __init__(self, name, root, clino):
        self.path = root / (name + '.c')
        self.data = self.path.read_text()
        s = re.sub(r'/\*.*?\*/', '', self.data, flags=re.S)
        s = re.sub(r'//[^\n]*', '', s).replace('\\\n', ' ')
        self.defs = {
            m.group(1): m.group(2).strip() for m in re.finditer(
                r'^#define\s+(\w+)\s+([^\n]+)', s, re.M)}
        self.defs.update({m.group(1): m.group(2) for m in re.finditer(
            r'static double\s+(\w+)\s*=\s*([^;]+);', s)})
        self.cache = {}
        self.clino = clino

    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        if key not in self.defs:
            return sp.Symbol(key)
        raw = self.defs[key]
        if '(clino)' in raw and '?' in raw:
            choices = raw.split('?', 1)[1].split(':', 1)
            raw = choices[0] if self.clino else choices[1]
        s = re.sub(r'([rs])\[(\d+)\]', r'\1\2', raw)
        s = s.replace('LOG(', 'log(')
        expr = sp.sympify(
            s,
            rational=True,
            locals={
                'S': sp.Symbol('S'),
                'H': sp.Symbol('H'),
                'V': sp.Symbol('V'),
                'R': sp.Symbol('R')})
        expr = expr.xreplace({x: self.get(str(x))
                              for x in expr.free_symbols if str(x) in self.defs})
        self.cache[key] = expr
        return expr


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe pinned MAGMA expressions and native molar ledgers.")
    parser.add_argument("--source-directory", required=True)
    parser.add_argument("--native-standards-directory", required=True)
    parser.add_argument("--host-properties", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--additional-models", nargs="*", default=[],
                        help="Independent pinned pyroxene/spinel, rhm and binary-instruction model fragments.")
    args = parser.parse_args()
    root = Path(args.source_directory)
    v = sp.symbols('v0:6')
    t, p = sp.symbols('t p')
    models = {}

    def poly(expr, vs):
        return [[float(coeff), list(powers)] for powers, coeff in sp.Poly(
            sp.expand(expr), *vs).terms() if coeff != 0]

    def make(phase, source, vs, subs, components, emap, sites, constraints=(
    ), requires_absent=(), indices=None, entropy_extra=0, barrier=None):
        m = Macros(source, root, phase != 'orthoamphibole')
        h = sp.expand(m.get('H').subs(subs))
        vol = sp.expand(m.get('V').subs(
            subs)) if 'V' in m.defs else sp.Integer(0)
        entropy = m.get('S').subs(subs)
        # Delete only logarithms with nonconstant arguments: these are
        # represented as site terms.
        entropy = entropy.xreplace({node: 0 for node in sp.preorder_traversal(
            entropy) if node.func == sp.log and node.args[0].free_symbols})
        # The remaining symbols are site names from the source, multiplying
        # deleted logs; expansion removes them.
        entropy = sp.simplify(entropy) + entropy_extra
        expr = sp.expand(h - t * entropy + (p - 1) * vol)
        if phase == 'garnet':
            expr = sp.expand(h - t * entropy + p * vol)
        if phase == 'olivine':
            expr = sp.expand(h - t * entropy)
        hterm = expr.subs({t: 0, p: 1})
        sterm = -sp.diff(expr, t)
        vterm = sp.diff(expr, p)
        assert not (hterm.free_symbols | sterm.free_symbols |
                    vterm.free_symbols) - set(vs), (phase, expr)
        data = {'phase': phase, 'expression_id': 'magma_published_' + phase.replace('-', '_') + '_v1', 'source_file': source + '.c', 'source_sha256': hashlib.sha256(m.path.read_bytes()).hexdigest(),
                'coordinate_order': [str(x) for x in vs], 'coordinate_bounds': [[0, 1] for x in vs],
                'endmember_names': components, 'native_endmember_indices': indices if indices is not None else list(range(len(components))),
                'endmember_polynomials': [poly(x, vs) for x in emap], 'H_polynomial': poly(hterm, vs), 'S_polynomial': poly(sterm, vs), 'V_polynomial': poly(vterm, vs),
                'entropy_sites': [{'multiplicity': float(c), 'polynomial': poly(x, vs)} for c, x in sites],
                'nonnegative_polynomials': [poly(x, vs) for x in constraints], 'required_absent_elements': list(requires_absent),
                'boundary_policy': 'Continuous 0 log 0 site entropy; no endpoint derivatives are supplied.', 'barrier': barrier}
        # Canonical oxide stoichiometry is recovered as exact half-integers and
        # checked against the native molar ledger.
        native = json.load(
            open(str(Path(args.native_standards_directory) / (phase + '.json'))))
        matrix = np.array(native.get('native_basis', native)[
                          'native_endmember_oxide_mass_g_per_mol'])
        host = json.load(open(args.host_properties))
        masses = np.array(host['oxide_molar_masses_g_mol'])
        ox = matrix / masses[:, None]
        exact = np.round(2 * ox) / 2
        assert np.max(np.abs(exact - ox)) < 1e-8
        data['endmember_oxide_moles'] = [exact[:, i].tolist()
                                         for i in data['native_endmember_indices']]
        models[phase] = data

    a, b, c = v[:3]
    r = sp.symbols('r0:5')
    s = sp.symbols('s0:4')
    make(
        'hornblende', 'hornblende', (a, b), {
            r[0]: a, r[1]: b}, [
            'pargasite', 'ferropargasite', 'magnesiohastingsite'], [
                1 - a - b, a, b], [
                    (4, a), (4, 1 - a), (1, b), (1, 1 - b)])
    make('biotite', 'biotite', (a,), {sp.Symbol('xan'): a, sp.Symbol(
        'xph'): 1 - a}, ['annite', 'phlogopite'], [a, 1 - a], [(1, a), (1, 1 - a)])
    make('garnet',
         'garnet',
         (a,
          b),
         {sp.Symbol('xal'): a,
          sp.Symbol('xgr'): b,
          sp.Symbol('xpy'): 1 - a - b},
         ['almandine',
          'grossular',
          'pyrope'],
         [a,
          b,
          1 - a - b],
         [(3,
           a),
          (3,
           b),
             (3,
              1 - a - b)],
         [1 - a - b])
    make('alkali-feldspar',
         'feldspar',
         (a,
          b),
         {sp.Symbol('xab'): a,
          sp.Symbol('xan'): b,
          sp.Symbol('xor'): 1 - a - b},
         ['albite',
          'anorthite',
          'sanidine'],
         [a,
          b,
          1 - a - b],
         [(1,
           a),
          (1,
           b),
             (1,
              1 - a - b)],
         [1 - a - b])
    make('kalsilite',
         'kalsilite',
         (a,
          b,
          c),
         {r[0]: a,
          r[1]: b,
             r[2]: c},
         ['Na4Al4Si4O16',
             'K4Al4Si4O16',
             'Na3Al3Si5O16',
             'CaNa2Al4Si4O16'],
         [1 - a - b - c,
             a,
             b,
             c],
         [(4,
           a),
             (4,
              b / 4),
             (4,
              1 - a - b / 4 - c / 2),
             (4,
              c / 4)],
         [1 - b - c,
             1 - a - b / 4 - c / 2],
         barrier={'numerator_J_mol': float(np.sqrt(np.finfo(float).eps)),
                  'coordinate_index': 1,
                  'zero_limit': '+infinity'})
    make('leucite',
         'leucite',
         (a,
          b),
         {r[0]: a,
          r[1]: b},
         ['leucite',
             'analcime',
             'Na-leucite'],
         [a,
             b,
             1 - a - b],
         [(1,
           a * (1 - b)),
             (1,
              (1 - a) * (1 - b)),
             (1,
              b),
             (1.5,
              2 * a * b / 3),
             (1.5,
              2 * (1 - a) * b / 3),
             (1.5,
              1 - 2 * b / 3)])
    for phase in ['alloy-solid', 'alloy-liquid']:
        make(phase, phase, (a,), {sp.Symbol('xni'): a, sp.Symbol(
            'xfe'): 1 - a}, ['Fe', 'Ni'], [1 - a, a], [(1, a), (1, 1 - a)])
    make('cummingtonite',
         'cummingtonite',
         (a,
          b,
          c),
         {r[0]: 2 * (2 * a + 3 * b + 2 * c) / 7 - 1,
          s[0]: a - b,
             s[1]: a - c},
         ['cummingtonite',
             'grunerite'],
         [1 - (2 * a + 3 * b + 2 * c) / 7,
             (2 * a + 3 * b + 2 * c) / 7],
         [(2,
           a),
             (2,
              1 - a),
             (3,
              b),
             (3,
              1 - b),
             (2,
              c),
             (2,
              1 - c)])
    make('olivine',
         'olivine',
         (a,
          b,
          c),
         {r[0]: -1,
          r[1]: a + b - 1,
             r[2]: -1,
             r[3]: -1,
             r[4]: c,
             s[0]: 0,
             s[1]: b - a,
             s[2]: 0,
             s[3]: 0},
         ['fayalite',
             'monticellite',
             'forsterite'],
         [(a + b) / 2,
             c,
             1 - (a + b) / 2 - c],
         [(1,
           a),
             (1,
              1 - a),
             (1,
              b),
             (1,
              c),
             (1,
              1 - b - c)],
         [1 - b - c],
         requires_absent=['Mn',
                          'Ni',
                          'Co'],
         indices=[1,
                  4,
                  5])
    # Independent site variables xK(LS), xVac(LS), xK(SS), xCa(SS) span the
    # full ordered nepheline domain.
    d = v[3]
    r0 = (a + 3 * c) / 4
    r1 = b - 3 * d
    r2 = 3 * d
    so = a - c
    make(
        'nepheline', 'nepheline', (a, b, c, d), {
            r[0]: r0, r[1]: r1, r[2]: r2, s[0]: so}, [
            'Na4Al4Si4O16', 'K4Al4Si4O16', 'Na3Al3Si5O16', 'CaNa2Al4Si4O16'], [
                1 - r0 - r1 - r2, r0, r1, r2], [
                    (1, a), (1, b), (1, 1 - a - b), (-1, 1 - 3 * d), (3, c), (3, d), (3, 1 - c - d)], [
                        1 - a - b, 1 - c - d, b - 3 * d, 1 - 3 * d], barrier={
                            'numerator_J_mol': float(
                                np.sqrt(
                                    np.finfo(float).eps)), 'polynomial': poly(
                                        b - 3 * d, (a, b, c, d)), 'zero_limit': '+infinity'})
    for phase in ['clinoamphibole', 'orthoamphibole']:
        make(phase, 'amphibole', (a, b, c, d), {r[0]: 2 *
                                                (2 *
                                                 a +
                                                 3 *
                                                 b +
                                                 2 *
                                                 c) /
                                                7 -
                                                1, r[1]: d, s[0]: a -
                                                b, s[1]: a -
                                                c}, ['cummingtonite', 'grunerite', 'tremolite'], [1 -
                                                                                                  (2 *
                                                                                                   a +
                                                                                                   3 *
                                                                                                   b +
                                                                                                   2 *
                                                                                                   c) /
                                                                                                  7 -
                                                                                                  d, (2 *
                                                                                                      a +
                                                                                                      3 *
                                                                                                      b +
                                                                                                      2 *
                                                                                                      c) /
                                                                                                  7, d], [(2, a), (2, 1 -
                                                                                                                   a -
                                                                                                                   d), (2, d), (3, b), (3, 1 -
                                                                                                                                        b), (2, c), (2, 1 -
                                                                                                                                                     c)], [1 -
                                                                                                                                                           a -
                                                                                                                                                           d])
    make('melilite', 'melilite', (a, b, c, d), {r[0]: a, r[1]: b, r[2]: c, s[0]: a -
                                                d}, ['akermanite', 'gehlenite', 'Fe-akermanite', 'Na-melilite'], [1 -
                                                                                                                  a -
                                                                                                                  b -
                                                                                                                  c, a, b, c], [(2, 1 -
                                                                                                                                 c), (2, c), (1, 1 -
                                                                                                                                              a -
                                                                                                                                              b -
                                                                                                                                              c), (1, b), (1, d), (1, a +
                                                                                                                                                                   c -
                                                                                                                                                                   d), (2, a -
                                                                                                                                                                        d /
                                                                                                                                                                        2), (2, 1 -
                                                                                                                                                                             a +
                                                                                                                                                                             d /
                                                                                                                                                                             2)], [1 -
                                                                                                                                                                                   a -
                                                                                                                                                                                   b -
                                                                                                                                                                                   c, a +
                                                                                                                                                                                   c -
                                                                                                                                                                                   d, a -
                                                                                                                                                                                   d /
                                                                                                                                                                                   2])
    models['melilite']['pure_reference_bounds'] = [{'endmember_index': 1, 'enthalpy_lower_J_mol': 0., 'enthalpy_upper_J_mol': 12000. + 38354. / 4., 'entropy_site_groups': [[1, 2], [
        2, 2]], 'derivation':'For pure gehlenite, s is in [0,1], H=12000s+38354s(1-s), and the two normalized site groups have multiplicities 1 and 2. Thus 0<=H<=12000+38354/4 and -3R ln2<=-S<=0. This encloses every admissible pure-order state, not only a numerical minimum.'}]
    e = v[4]
    make('ortho-oxide',
         'ortho-oxide',
         (a,
          b,
          c,
          d,
          e),
         {r[0]: a,
          r[1]: b,
             s[0]: 2 * c - 1 + a + b,
             s[1]: 2 * d - a,
             s[2]: 2 * e - b},
         ['pseudobrookite',
             'ferropseudobrookite',
             'karrooite'],
         [1 - a - b,
             a,
             b],
         [(1,
           d),
             (1,
              e),
             (1,
              c),
             (1,
              1 - c - d - e),
             (2,
              (a - d) / 2),
             (2,
              (b - e) / 2),
             (2,
              1 - a - b - c / 2),
             (2,
              (a + b + c + d + e) / 2)],
         [1 - a - b,
             1 - c - d - e,
             a - d,
             b - e,
             1 - a - b - c / 2])
    models['ortho-oxide']['pure_reference_bounds'] = [{'endmember_index': j, 'enthalpy_lower_J_mol': 0. if j == 0 else -12500., 'enthalpy_upper_J_mol': 23500. if j == 0 else 11000., 'entropy_site_groups': [[1, 2], [2 if j == 0 else 1, 2]],
                                                       'derivation':'Published pureOox: PB_H=12500-11000s-1500s^2; FE_H=MG_H=-11000s-1500s^2, -1<=s<=1. The PB entropy is the sum of normalized binary-site entropies of multiplicities 1 and 2. FE/MG entropy is a binary entropy plus at most ln2. Interval bounds enclose every admissible pure-order state.'} for j in range(3)]
    for fragment in args.additional_models:
        incoming = json.loads(Path(fragment).read_text())
        additions = incoming.get("models", incoming)
        if set(additions) & set(models):
            raise ValueError("An additional fragment would replace an existing declared model.")
        for model in additions.values():
            model.pop("native_standard_receipt_sha256", None)
            if model["phase"] == "plagioclase":
                model["source_disassembly_audit"] = "plagioclase_instruction_audit.json"
        models.update(additions)
    result = {
        'schema': 'magma_site_mixing_polynomials_v1',
        'upstream_commit': '705a0fb315e5054d18275a580562f6121c8e458c',
        'upstream_url': 'https://github.com/magmasource/MAGMA',
        'native_entropy_R_J_mol_K': 8.3143,
        'oxide_order': json.loads(
            Path(
                args.host_properties).read_text())['oxide_order'],
        'models': models}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print('models', list(models))


if __name__ == '__main__':
    main()
