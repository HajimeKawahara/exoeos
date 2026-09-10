"""Reproduce the Fe-Si-O reference from published analytic activity equations.

This is an offline reference calculation, not a native ExoEOS backend.
It uses Decimal arithmetic and imports neither ExoEOS nor JAX.
"""

import argparse
import ast
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path


REFERENCE = Path(__file__).with_name("fe_si_o_ma2001.json")
D = Decimal


def activities(parameters, temperature, composition):
    """Evaluate Vogel A1 and the full solute equation in Buchan equation 6."""
    p = {key: D(value) for key, value in parameters.items()}
    scale = p["reference_temperature_K"] / temperature
    a = p["epsilon_SiSi_at_reference_temperature"] * scale
    b = p["epsilon_OO_times_temperature_K"] / temperature
    c = p["epsilon_SiO_at_reference_temperature"] * scale
    ls = p["ln_gamma0_Si_at_reference_temperature"] * scale
    lo = p["ln_gamma0_O_constant"] + p["ln_gamma0_O_inverse_temperature_K"] / temperature
    _, x, y = composition
    lx, ly = (1 - x).ln(), (1 - y).ln()
    ax, ay = 1 / (1 - x), 1 / (1 - y)

    # The five terms of Vogel A1; products remove the removable ln(1-x)/x
    # singularities before evaluation. The sum over ordered pairs has two terms.
    fe = a * (x + lx) + b * (y + ly)
    fe -= c * (x * y + y * lx + x * ly)
    fe += c * (x * y + x * ly - x * y * ax)
    fe += c * (x * y + y * lx - x * y * ay)
    fe += c * x**2 * y**2 * (ax + ay - 1) / 2
    fe -= c * x**2 * y**2 * (ax + ay + x * ax**2 / 2 - 1)
    fe -= c * x**2 * y**2 * (ay + ax + y * ay**2 / 2 - 1)

    def solute_correction(i, j, li, self_interaction):
        return (
            li - self_interaction * (1 - i).ln()
            - c * (j + (1 - j).ln() - j / (1 - i))
            + c * j**2 * i
            * (1 / (1 - i) + 1 / (1 - j) + i / (2 * (1 - i)**2) - 1)
        )

    si = solute_correction(x, y, ls, a)
    oxygen = solute_correction(y, x, lo, b)
    source = [fe, fe + si, fe + oxygen]
    shift = [D(0), ls + a, lo + b]
    formal = [value - offset for value, offset in zip(source, shift)]
    gce_oxygen = solute_correction(y, x, lo, -scale)
    return {
        "ln_gamma_source": source,
        "ln_gamma_formal": formal,
        "mu0_formal_minus_source_over_RT": shift,
        "gex_source_RT": sum(xi * gi for xi, gi in zip(composition, source)),
        "gex_formal_RT": sum(xi * gi for xi, gi in zip(composition, formal)),
        "young_printed_solute_corrections": [si, oxygen],
        "gce_solute_corrections": [si, gce_oxygen],
    }


def regenerate(record):
    """Retain provenance/inputs and replace calculated reference outputs."""
    with localcontext() as context:
        context.prec = 60
        for state in record["states"]:
            temperature = D(str(state["T_K"]))
            composition = [D(str(value)) for value in state["x"]]
            if temperature <= 0 or state["P_Pa"] <= 0:
                raise ValueError("Reference T and P must be positive.")
            if sum(composition) != 1 or composition[0] <= 0 or min(composition) < 0:
                raise ValueError("Reference requires normalized nonnegative fractions and Fe>0.")
            values = activities(record["parameters"], temperature, composition)
            for key, value in values.items():
                state[key] = [float(v) for v in value] if isinstance(value, list) else float(value)
    return record


def check_gce(record, checkout):
    """Check the exact pinned upstream expressions, without running its solver."""
    pin = record["gce_comparison"]
    source = (checkout / pin["path"]).read_bytes()
    if hashlib.sha256(source).hexdigest() != pin["sha256"]:
        raise ValueError("GCE source does not match the pinned SHA-256.")
    expressions = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in pin["assignments"]:
                expressions[name] = compile(ast.Expression(node.value), pin["path"], "eval")
    if set(expressions) != set(pin["assignments"]):
        raise ValueError("Missing pinned GCE activity assignments.")
    checked = 0
    for state in record["states"]:
        _, x, y = state["x"]
        if x == 0 or y == 0:
            continue  # Upstream uses ln(1-x)/x without an exact-zero limit.
        namespace = {"T_SME": state["T_K"], "var": {"Si_metal": x, "O_metal": y}}
        for name, expected in zip(pin["assignments"], state["gce_solute_corrections"]):
            actual = eval(expressions[name], {"__builtins__": {}, "log": math.log}, namespace)
            if not math.isclose(actual, expected, rel_tol=5e-12, abs_tol=5e-12):
                raise AssertionError((state["id"], name, actual, expected))
        checked += 1
    print(f"Pinned GCE expressions agree at {checked} compositions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the committed fixture without writing.")
    parser.add_argument("--gce-checkout", type=Path, help="Optionally compare a local pinned GCE checkout.")
    args = parser.parse_args()
    original = REFERENCE.read_text()
    record = regenerate(json.loads(original))
    rendered = json.dumps(record, indent=2, allow_nan=False) + "\n"
    if args.check:
        if original != rendered:
            raise SystemExit("Reference differs; regenerate and review the fixture.")
        print("Fe-Si-O reference is reproducible.")
    else:
        REFERENCE.write_text(rendered)
    if args.gce_checkout is not None:
        check_gce(record, args.gce_checkout)


if __name__ == "__main__":
    main()
