"""Package the audited Ca--Mg--Fe amphibole site polynomials."""

import hashlib
import json
from pathlib import Path
from pprint import pformat

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'examples/m2_solid_mixing/parameters.json'
DESTINATION = ROOT/'src/exoeos/magma_solids/_hydrous_coefficients.py'


def render():
    data = json.loads(SOURCE.read_text())
    lines = ['"""Pinned amphibole coefficients; regenerate with extract_hydrous_solid_coefficients.py."""',
             '', '# Source parameter SHA256: '+hashlib.sha256(SOURCE.read_bytes()).hexdigest(), '']
    for phase in ('clinoamphibole', 'orthoamphibole'):
        d = data['models'][phase]
        compact = {'size': 4, **{k: d[k+'_polynomial'] for k in ('H', 'S', 'V')},
                   'sites': [(s['multiplicity'], s['polynomial']) for s in d['entropy_sites']]}
        lines += ['# '+d['source_file']+' SHA256: '+d['source_sha256'],
                  phase.upper()+' = '+pformat(compact, width=100, sort_dicts=False), '']
    return '\n'.join(lines)


if __name__ == '__main__':
    DESTINATION.write_text(render())
