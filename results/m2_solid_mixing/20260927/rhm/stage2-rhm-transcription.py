"""Transcribe pinned five-endmember rhomsghiorso.c on the complete Mn-free face.

The expression leaves both surviving order parameters free. This file writes
only the independent parameter fragment; it does not call a native MELTS solver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import sympy as sp

SOURCE_SHA256 = "9c06e35c78c8e522bb05d6d1933fa34806ddc927c2002980872d5fbeb00cec5e"
UPSTREAM_COMMIT = "705a0fb315e5054d18275a580562f6121c8e458c"
R = sp.Rational("8.3143")


class RhmMacros:
    """Read only the pinned numeric constants and object-like expressions."""

    def __init__(self, path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256
        source = re.sub(r"/\*.*?\*/", "", raw.decode(), flags=re.S)
        source = re.sub(r"//[^\n]*", "", source).replace("\\\n", " ")
        self.defs = {m.group(1): m.group(2).strip() for m in re.finditer(
            r"^#define\s+(\w+)\s+([^\n]+)", source, re.M)}
        self.defs.update({m.group(1): m.group(2).strip() for m in re.finditer(
            r"static double\s+(\w+)\s*=\s*([^;]+);", source)})
        self.cache = {}
        self.sro = sp.Rational(self.defs["SROconst"])
        assert all(sp.Rational(self.defs[f"SRO{c}"]) == self.sro
                   for c in range(600, 1400, 100))
        assert [int(self.defs[name]) for name in ("NR", "NS", "NA")] == [4, 3, 5]

    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        if key not in self.defs:
            return sp.Symbol(key)
        raw = re.sub(r"([rs])\[(\d+)\]", r"\1\2", self.defs[key])
        expr = sp.sympify(raw, rational=True, locals={
            "S": sp.Symbol("S"), "H": sp.Symbol("H"), "G": sp.Symbol("G"),
            "R": sp.Symbol("R"), "fSRO": lambda t, derivative: self.sro if derivative == 0 else sp.Integer(0)})
        expr = expr.xreplace({x: self.get(str(x)) for x in expr.free_symbols
                             if str(x) in self.defs})
        self.cache[key] = expr
        return expr


def polynomial(expr, variables):
    return [[float(coefficient), list(powers)] for powers, coefficient in
            sp.Poly(sp.expand(expr), *variables).terms() if coefficient != 0]


def transcribe_rhm(source_directory, native_standards_directory, host_properties):
    """Return one provider-compatible model, including reference specifications."""
    source = RhmMacros(Path(source_directory) / "rhomsghiorso.c")
    a, b, c, d, e = v = sp.symbols("v0:5")
    r = sp.symbols("r0:4")
    s = sp.symbols("s0:3")
    p = sp.Symbol("p")
    h = 1 - a - b - c - d - e
    substitutions = {r[0]: a+b, r[1]: c+d, r[2]: 0, r[3]: e,
                     s[0]: a-b, s[1]: c-d, s[2]: 0}
    enthalpy_pressure = sp.expand(source.get("H").subs(substitutions))
    enthalpy = sp.expand(enthalpy_pressure.subs(p, 1))
    volume = sp.expand(sp.diff(enthalpy_pressure, p))
    assert sp.diff(enthalpy_pressure, p, 2) == 0
    assert not (enthalpy.free_symbols | volume.free_symbols) - set(v)

    # Extract every source logarithm with its signed coefficient. This avoids
    # evaluating absent Mn contributions as the indeterminate product 0*log(0).
    entropy = sp.expand(source.get("S"))
    occupations = {
        "xfe2a": a, "xfe2b": b, "xmg2a": c, "xmg2b": d,
        "xmn2a": 0, "xmn2b": 0, "xti4a": b+d, "xti4b": a+c,
        "xfe2ID": (a+b)/2, "xmg2ID": (c+d)/2, "xmn2ID": 0,
        "xti4ID": (a+b+c+d)/2, "xfe3ID": h, "xal3ID": e,
    }
    sites = []
    constant = entropy
    recovered = 0
    for name, occupation in occupations.items():
        q = sp.Symbol(name)
        coefficient = sp.simplify(entropy.coeff(sp.log(q)) / q)
        assert not coefficient.free_symbols
        contribution = coefficient * q * sp.log(q)
        constant -= contribution
        recovered += contribution
        if occupation != 0 and coefficient != 0:
            sites.append({"source_site": name,
                          "multiplicity": float(-coefficient / R),
                          "polynomial": polynomial(occupation, v)})
    constant = sp.simplify(constant)
    assert sp.simplify(constant - 2 * source.sro * R * sp.log(2)) == 0
    assert sp.simplify(entropy - recovered - constant) == 0
    # In G the A/B multiplicities are +1, divalent/Ti ID coefficients are
    # -2*SRO, and trivalent ID coefficients are 2*(1-SRO).
    assert len(sites) == 11

    native_path = Path(native_standards_directory) / "rhm-oxide.json"
    native = json.loads(native_path.read_text())
    host = json.loads(Path(host_properties).read_text())
    masses = np.asarray(host["oxide_molar_masses_g_mol"])
    oxide_raw = np.asarray(native["native_basis"]["native_endmember_oxide_mass_g_per_mol"]) / masses[:, None]
    oxide_exact = np.round(oxide_raw * 2) / 2
    assert np.max(np.abs(oxide_exact - oxide_raw)) < 1e-8
    indices = [0, 1, 2, 4]
    endmembers = [c+d, h, a+b, e]
    data = {
        "phase": "rhm-oxide",
        "expression_id": "magma_published_rhm_oxide_mn_free_v1",
        "source_file": "rhomsghiorso.c", "source_sha256": SOURCE_SHA256,
        "coordinate_order": [str(x) for x in v],
        "coordinate_meanings": ["Fe2_A", "Fe2_B", "Mg_A", "Mg_B", "Al_A=Al_B"],
        "coordinate_bounds": [[0., 1.] for _ in v],
        "source_r_polynomials": [polynomial(x, v) for x in (a+b, c+d, 0, e)],
        "source_s_polynomials": [polynomial(x, v) for x in (a-b, c-d, 0)],
        "endmember_names": ["geikielite", "hematite", "ilmenite", "corundum"],
        "native_endmember_indices": indices,
        "endmember_polynomials": [polynomial(x, v) for x in endmembers],
        "endmember_oxide_moles": [oxide_exact[:, i].tolist() for i in indices],
        "H_polynomial": polynomial(enthalpy, v),
        "S_polynomial": polynomial(constant, v),
        "V_polynomial": polynomial(volume, v),
        "entropy_sites": sites,
        "nonnegative_polynomials": [polynomial(h, v)],
        "required_absent_elements": ["Mn"],
        "boundary_policy": "Continuous 0 log 0 site entropy; no endpoint derivatives are supplied.",
        "barrier": None,
        "parameter_domain": {
            "T_K": {"lower": 0., "lower_inclusive": False, "upper": None},
            "P_bar": {"lower": 0., "lower_inclusive": False, "upper": None},
            "SRO_default_only": True,
            "SRO_value": float(source.sro),
            "SRO_spline_nodes_K": [873.15 + 100*i for i in range(8)],
            "SRO_spline_values": [float(source.sro)] * 8,
            "SRO_derivation": "All eight source ordinates SRO600 through SRO1300 equal SROconst=0.0730205. Natural-spline construction starts with u[0]=y2[0]=y2[7]=0; zero adjacent slopes imply all u and y2 are zero. On every interval, including endpoint extrapolation, a+b=1, so fSRO(T,0)=0.0730205 and derivatives 1 through 3 vanish. Runtime parameter resets are outside this fixed-default declaration. No finite mathematical T bound follows from this spline; no empirical calibration domain is asserted.",
        },
        "reference_policy": "G_mix=G_site-sum(x_i*G_pure_i). H/S/V above describe G_site, without pureRhm subtraction.",
        "native_standard_receipt_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "native_binary_uniform_error_bound": None,
        "empirical_calibration_domain_established": False,
    }
    u = sp.Symbol("u")
    pure_models = []
    pure_bounds = []
    for j, prefix, source_order in [(0, "GK", s[1]), (1, "HM", None),
                                     (2, "IL", s[0]), (3, "CR", None)]:
        ph = source.get(prefix + "_H")
        if source_order is not None:
            ph = sp.expand(ph.subs(source_order, 2*u-1))
            ps = 0
            psites = [{"multiplicity": 2., "polynomial": polynomial(q, (u,))}
                      for q in (u, 1-u)]
            pure_bounds.append({
                "endmember_index": j,
                "enthalpy_lower_J_mol": 0.,
                "enthalpy_upper_J_mol": 17477. + 3189./4,
                "volume_lower_J_mol_bar": 0.,
                "volume_upper_J_mol_bar": 0.010758 + 0.035089/4,
                "entropy_site_groups": [[2, 2]],
                "derivation": "For z=s^2 in [0,1], H(P)=A(P)*(1-z)+B(P)*z*(1-z), A=17477+0.010758*(P_bar-1), B=3189+0.035089*(P_bar-1). For every P_bar>0, A,B>0 and 0<=H(P)<=A+B/4. The pure entropy is twice a normalized binary-site entropy, in [0,2*R*ln(2)]. The stated H(1 bar) and V bounds must be combined by signed interval multiplication of (P_bar-1), or use the sharper direct A+B/4 bound. These bounds enclose every order state, independently of native minimizer convergence.",
            })
        else:
            ph = sp.Integer(ph)
            ps = source.get(prefix + "_S")
            psites = []
            pure_bounds.append({
                "endmember_index": j,
                "enthalpy_lower_J_mol": 0., "enthalpy_upper_J_mol": 0.,
                "entropy_lower_J_mol_K": float(ps),
                "entropy_upper_J_mol_K": float(ps),
                "entropy_site_groups": [],
                "derivation": "HM_G=CR_G=-2*T*R*SRO*ln(2) exactly under the declared equal-ordinate default spline. The entropy constant must be retained; zero site groups do not mean zero entropy.",
            })
        pure_models.append({
            "endmember_index": j, "native_endmember_index": indices[j],
            "coordinate_order": ["u"], "coordinate_bounds": [[0., 1.]],
            "H_polynomial": polynomial(ph.subs(p, 1), (u,)),
            "S_polynomial": polynomial(ps, (u,)),
            "V_polynomial": polynomial(sp.diff(ph, p), (u,)),
            "entropy_sites": psites,
            "nonnegative_polynomials": [], "barrier": None,
        })
    data["pure_reference_models"] = pure_models
    data["pure_reference_bounds"] = pure_bounds
    data["pure_reference_high_temperature_identity"] = {
        "native_endmember_indices": [0, 2, 3],
        "retained_model_endmember_indices": [0, 2],
        "condition": "P_bar>0 and native_R*T_K >= 17477+0.010758*(P_bar-1)",
        "gibbs_minimum_J_mol": "17477+0.010758*(P_bar-1)-2*native_R*T_K*ln(2)",
        "proof": "For |s|<=1, (1+s)ln(1+s)+(1-s)ln(1-s)=sum_{n>=1} s^(2n)/(n*(2n-1)) >= s^2. Therefore g(s)-g(0)>=s^2*(R*T-A)+B*s^2*(1-s^2)>=0 when R*T>=A and B>=0. This proves the global pure-state minimum, without sampling or trusting pureOrder Newton convergence. It does not bound floating-point native binary error.",
    }
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", default="/tmp/stage2-magma-source")
    parser.add_argument("--native-standards-directory", default="/tmp/stage2-solid-standards")
    parser.add_argument("--host-properties", default="/tmp/stage2-liquid-global-fixed2-h1e24/host_properties.json")
    parser.add_argument("--output", default="/tmp/stage2-rhm-parameters.json")
    args = parser.parse_args()
    host = json.loads(Path(args.host_properties).read_text())
    model = transcribe_rhm(args.source_directory, args.native_standards_directory, args.host_properties)
    result = {"schema": "magma_site_mixing_polynomials_v1",
              "upstream_commit": UPSTREAM_COMMIT,
              "upstream_url": "https://github.com/magmasource/MAGMA",
              "native_entropy_R_J_mol_K": float(R), "oxide_order": host["oxide_order"],
              "models": {"rhm-oxide": model}}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": args.output, "source_sha256": SOURCE_SHA256,
                      "H_terms": len(model["H_polynomial"]),
                      "V_terms": len(model["V_polynomial"]),
                      "entropy_sites": len(model["entropy_sites"])}))


if __name__ == "__main__":
    main()
