"""Lift the pinned gmixPlg Gibbs-output instruction path into a declared model.

This audit interprets SSE arithmetic over reals, retains binary64 constants, and
models log as the real logarithm. It is not a floating-point/libm error proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

import numpy as np
import sympy as sp

BINARY_SHA256 = 'c218ef6f7ba5aef4b0760f3176530d1335457d6e2716823a1d0de3a5d13301e5'
FUNCTION_START = 0x29f300
FUNCTION_SIZE = 0xb04
PATH_START = 0x29f548
PATH_OUTPUT_STORE = 0x29f75b


def poly(expression, variables):
    return [[float(c), list(p)] for p, c in sp.Poly(sp.expand(expression), *variables).terms() if c]


def lift(binary):
    binary = Path(binary)
    raw = binary.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BINARY_SHA256
    # In this pinned ELF, .text and .rodata virtual addresses equal file offsets.
    # Check the ELF64 section headers directly instead of assuming that mapping.
    header = struct.unpack_from('<16sHHIQQQIHHHHHH', raw)
    shoff, shentsize, shnum = header[6], header[11], header[12]
    sections = [struct.unpack_from(
        '<IIQQQQIIQQ', raw, shoff+i*shentsize) for i in range(shnum)]
    text_section = next(
        q for q in sections if q[3] <= FUNCTION_START < q[3]+q[5])
    assert text_section[3] == text_section[4]

    disassembly = subprocess.check_output(
        ['objdump', '-d', '--no-show-raw-insn', '--disassemble=gmixPlg', str(binary)], text=True)
    a, b, c, T, P = sp.symbols('xab xan xor T P', positive=True)
    # The prologue at 0x29f328--0x29f36b loads xab=input[0], xan=input[1],
    # xor=1-input[0]-input[1], clamps each to DBL_EPSILON, and stores these lanes.
    stack = {'-0x40(%rbp)': [a, 0], '-0x70(%rbp)': [b, b],
             '-0x50(%rbp)': [c, b], '-0x80(%rbp)': [T, 0], '-0xb0(%rbp)': [P, 0]}
    registers = {}
    constants = {}
    audit = []
    output = None

    def load(operand, address=None):
        if operand in registers:
            return list(registers[operand])
        if operand in stack:
            return list(stack[operand])
        if '%rip' in operand:
            assert address is not None
            section = next(q for q in sections if q[3] <= address < q[3]+q[5])
            assert section[3] == section[4]
            doubles = struct.unpack_from('<dd', raw, address)
            constants[hex(address)] = {'binary64_values': list(doubles),
                                       'first_double_hex': raw[address:address+8].hex(),
                                       'first_double_ratio': list(doubles[0].as_integer_ratio())}
            return [sp.Rational(d) for d in doubles]
        raise AssertionError(('Unknown operand', operand))

    for line in disassembly.splitlines():
        match = re.match(r'\s*([0-9a-f]+):\s+(\w+)\s*(.*)', line)
        if not match:
            continue
        address = int(match[1], 16)
        if not PATH_START <= address <= PATH_OUTPUT_STORE:
            continue
        instruction = match[2]
        rest = match[3].split('#', 1)[0].strip()
        operands = [q.strip() for q in rest.split(',')]
        comment = re.search(r'# ([0-9a-f]+)', line)
        constant_address = int(comment[1], 16) if comment else None
        audit.append({'address': hex(address),
                      'instruction': instruction, 'operands': rest})
        if instruction == 'mov':
            assert rest == '%rdx,%r13'
            continue
        if instruction == 'call':
            assert 'log@plt' in rest
            value = sp.log(registers['%xmm0'][0])
            # Other vector registers are caller-clobbered by the ABI. The
            # audited path reloads the quantities it needs after these calls.
            registers = {'%xmm0': [value, sp.Symbol('unused_log_high_lane')]}
            continue
        if instruction in ('movsd', 'movapd'):
            src, dst = operands
            value = load(src, constant_address)
            if address == PATH_OUTPUT_STORE:
                assert instruction == 'movsd' and dst == '0x0(%r13)'
                output = value[0]
            elif dst.startswith('%xmm'):
                registers[dst] = value if instruction == 'movapd' else [
                    value[0], 0]
            else:
                assert '%rbp' in dst
                stack[dst] = value if instruction == 'movapd' else [value[0], 0]
            continue
        if instruction == 'xorpd':
            assert operands[0] == operands[1]
            registers[operands[1]] = [sp.Integer(0), sp.Integer(0)]
            continue
        assert instruction in (
            'addsd', 'subsd', 'mulsd'), (address, instruction)
        src, dst = operands
        left = registers[dst][0]
        right = load(src, constant_address)[0]
        value = left+right if instruction == 'addsd' else left - \
            right if instruction == 'subsd' else left*right
        registers[dst] = [value, registers[dst][1]]

    assert output is not None
    G = sp.expand(output)
    H = sp.expand(G.subs({T: 0, P: 1}))
    S = sp.expand(-sp.diff(G, T))
    V = sp.expand(sp.diff(G, P))
    assert sp.expand(G - (H-T*S+(P-1)*V)) == 0
    expected_H = (7924*a*b*(b+c/2)
                  - sp.Rational('6498.5')*a*c*(c+b/2)
                  - sp.Rational('6498.5')*a*c*(a+b/2)
                  + sp.Rational('6498.5')*b*c*(c+a/2)
                  + sp.Rational('6498.5')*b*c*(b+a/2)
                  + sp.Rational('9055.5')*a*b*c)
    r = sp.Rational(struct.unpack_from('<d', raw, 0x32f038)[0])
    w = sp.Rational(struct.unpack_from('<d', raw, 0x32f1d8)[0])
    expected_S = -r*(a*sp.log(a)+b*sp.log(b)+c*sp.log(c)) + \
        w*a*c*(c+b/2)+w*a*c*(a+b/2)
    vab, vba, van, vtri = [sp.Rational(struct.unpack_from('<d', raw, n)[0]) for n in [
        0x32f1e8, 0x32f1f0, 0x32f1f8, 0x32f1a0]]
    expected_V = vab*a*c*(c+b/2)+vba*a*c*(a+b/2)+van*b*c*(c+a/2)+vtri*a*b*c
    assert sp.expand(H-expected_H) == 0
    assert sp.expand(S-expected_S) == 0
    assert sp.expand(V-expected_V) == 0
    receipt = {
        'schema': 'pinned_native_instruction_expression_audit_v1',
        'binary_sha256': BINARY_SHA256, 'binary_name': binary.name, 'symbol': 'gmixPlg',
        'symbol_virtual_address': hex(FUNCTION_START), 'symbol_size_bytes': FUNCTION_SIZE,
        'symbol_bytes_sha256': hashlib.sha256(raw[FUNCTION_START:FUNCTION_START+FUNCTION_SIZE]).hexdigest(),
        'G_output_path_start': hex(PATH_START), 'G_output_store': hex(PATH_OUTPUT_STORE),
        'lifted_instruction_count': len(audit),
        'real_arithmetic_H': str(H), 'real_arithmetic_S': str(S), 'real_arithmetic_V': str(V),
        'rodata_constants': constants, 'instructions': audit,
        'algebraic_identity_checks': {'G_equals_H_minus_TS_plus_Pminus1V': True,
                                      'H_equals_declared_Margules_form': True,
                                      'S_equals_declared_site_plus_excess_entropy': True,
                                      'V_equals_declared_volume_form': True},
        'interpretation': 'The complete FIRST-mask output path is interpreted over real arithmetic with exact binary64 literals and real log. The native prologue clamps each fraction to DBL_EPSILON. The declared closed simplex model uses continuous x*log(x) limits and unclamped fractions. Floating-point arithmetic, libm error, and endpoint clamp error are not uniformly bounded here.',
    }
    return H, S, V, r, (a, b, c), receipt, disassembly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--native-standards-directory', required=True)
    parser.add_argument('--host-properties', required=True)
    parser.add_argument('--output-prefix', required=True)
    args = parser.parse_args()
    H, S, V, R, x, receipt, disassembly = lift(args.binary)
    a, b, c = x
    v = sp.symbols('v0:2')
    substitutes = {a: v[0], b: v[1], c: 1-v[0]-v[1]}
    S_extra = sp.expand(S+R*(a*sp.log(a)+b*sp.log(b)+c*sp.log(c)))
    native_path = Path(args.native_standards_directory)/'plagioclase.json'
    native = json.loads(native_path.read_text())
    host = json.loads(Path(args.host_properties).read_text())
    matrix = np.asarray(native['native_basis']['native_endmember_oxide_mass_g_per_mol']
                        )/np.asarray(host['oxide_molar_masses_g_mol'])[:, None]
    exact = np.round(2*matrix)/2
    assert np.max(np.abs(exact-matrix)) < 1e-8
    model = {
        'phase': 'plagioclase',
        'expression_id': 'alphamelts_2_3_2_pinned_binary_plagioclase_v1',
        'source_file': 'libalphamelts.so:gmixPlg', 'source_sha256': BINARY_SHA256,
        'source_kind': 'pinned_native_binary_instruction_transcription',
        'source_symbol_sha256': receipt['symbol_bytes_sha256'],
        'source_disassembly_audit': 'stage2-plagioclase-instruction-audit.json',
        'coordinate_order': [str(q) for q in v], 'coordinate_bounds': [[0, 1], [0, 1]],
        'endmember_names': ['albite', 'anorthite', 'sanidine'],
        'native_endmember_indices': [0, 1, 2],
        'endmember_polynomials': [poly(q.subs(substitutes), v) for q in x],
        'endmember_oxide_moles': [exact[:, j].tolist() for j in range(3)],
        'H_polynomial': poly(H.subs(substitutes), v),
        'S_polynomial': poly(S_extra.subs(substitutes), v),
        'V_polynomial': poly(V.subs(substitutes), v),
        'entropy_sites': [{'multiplicity': 1., 'polynomial': poly(q.subs(substitutes), v)} for q in x],
        'nonnegative_polynomials': [poly(1-v[0]-v[1], v)],
        'required_absent_elements': [], 'barrier': None,
        'boundary_policy': 'Continuous 0 log 0 entropy on the closed simplex; native endpoint clamping is excluded from this real-arithmetic declaration.',
        'native_binary_uniform_error_bound': None,
        'empirical_calibration_domain_established': False,
        'native_standard_receipt_sha256': hashlib.sha256(native_path.read_bytes()).hexdigest(),
        'parameter_domain': {'T_K': {'lower': 0, 'lower_inclusive': False, 'upper': None},
                             'P_bar': {'lower': 0, 'lower_inclusive': False, 'upper': None}},
        'provenance_scope': 'The declared equation is recovered from gmixPlg in exactly the pinned alphaMELTS 2.3.2 Ubuntu binary SHA. It is not asserted to be the published MAGMA feldspar.c equation, or equivalent to every native build. Sparse coefficients retain the audited binary64 constants; arithmetic/log/clamp uniform error is separate.',
    }
    result = {'schema': 'magma_site_mixing_polynomials_v1',
              'upstream_commit': None, 'upstream_url': 'https://github.com/magmasource/alphaMELTS',
              'native_binary_sha256': BINARY_SHA256, 'native_entropy_R_J_mol_K': float(R),
              'oxide_order': host['oxide_order'], 'models': {'plagioclase': model}}
    Path(args.output_prefix +
         '-parameters.json').write_text(json.dumps(result, indent=2)+'\n')
    Path(args.output_prefix +
         '-instruction-audit.json').write_text(json.dumps(receipt, indent=2)+'\n')
    Path(args.output_prefix+'-disassembly.txt').write_text(disassembly)
    print(json.dumps({'instructions': receipt['lifted_instruction_count'], 'binary_sha256': BINARY_SHA256,
                      'symbol_sha256': receipt['symbol_bytes_sha256'], 'H_terms': len(model['H_polynomial']),
                      'S_terms': len(model['S_polynomial']), 'V_terms': len(model['V_polynomial'])}))


if __name__ == '__main__':
    main()
