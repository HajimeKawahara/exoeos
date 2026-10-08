"""Audit reconstructed CD2021 potentials against their original tables.

Run with JAX_ENABLE_X64=1. Downloads are checksum-verified by the existing
loader. Residuals measure backend differences, not physical error bounds.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exoeos import ChabrierDebrasTableLoader, MassThermodynamicState


def summarize(reference, reconstructed):
    result = {}
    for name in MassThermodynamicState._fields:
        original = np.asarray(getattr(reference, name))
        actual = np.asarray(getattr(reconstructed, name))
        residual = actual - original
        finite = np.isfinite(residual) & np.isfinite(original)
        nonzero = finite & (original != 0)
        absolute = np.abs(residual[finite])
        relative = np.abs(residual[nonzero] / original[nonzero])
        result[name] = {
            "nonfinite_count": int(np.count_nonzero(~finite)),
            "zero_reference_count": int(np.count_nonzero(finite & ~nonzero)),
            "max_absolute_SI": float(absolute.max()) if absolute.size else None,
            "rms_absolute_SI": float(np.sqrt(np.mean(absolute**2))) if absolute.size else None,
            "absolute_relative_quantiles_50_95_100": (
                np.quantile(relative, [0.5, 0.95, 1.0]).tolist() if relative.size else None
            ),
        }
    result["samples"] = int(original.size)
    for label, values in (
        ("nonpositive_pressure", reconstructed.P),
        ("nonpositive_entropy", reconstructed.s),
        ("nonpositive_cv", reconstructed.cv),
        ("nonpositive_dP_drho", reconstructed.pressure_density_derivative),
    ):
        result[label] = int(np.count_nonzero(np.asarray(values) <= 0))
    return result


def audit(loader):
    original = loader.load()
    potential = original.to_helmholtz()
    temperatures = np.asarray(potential.temperatures)
    densities = np.asarray(potential.mass_densities)
    evaluate = jax.jit(jax.vmap(potential.state_trho))
    reference = jax.jit(jax.vmap(original.state_trho))
    result = {"checksums": loader.checksums, "citation": loader.citation}
    for grid_name, ts, rs in (
        ("nodes", temperatures, densities),
        ("cell_centers", np.sqrt(temperatures[:-1] * temperatures[1:]),
         np.sqrt(densities[:-1] * densities[1:])),
    ):
        tt, rr = (a.ravel() for a in np.meshgrid(ts, rs, indexing="ij"))
        for region, mask in (
            ("full_rectangle", np.ones(tt.shape, dtype=bool)),
            ("qmd_box", (tt >= 1e3) & (tt <= 8e4) & (rr >= 200) & (rr <= 9000)),
        ):
            result[f"{grid_name}_{region}"] = summarize(
                reference(tt[mask], rr[mask]), evaluate(tt[mask], rr[mask])
            )

    # Independently tabulated TP values are not reconstruction constraints.
    ts, ps = np.meshgrid(np.geomspace(1e3, 8e4, 20), np.geomspace(1e9, 1e13, 20))
    tp = jax.jit(jax.vmap(original.state_tp))(ts.ravel(), ps.ravel())
    result["independent_tp_samples_at_original_density"] = summarize(
        tp, evaluate(ts.ravel(), tp.mass_density)
    )

    ts, rs = np.meshgrid([2000.0, 10000.0, 50000.0], [300.0, 1000.0, 5000.0])
    forward = evaluate(ts.ravel(), rs.ravel())
    inverse = jax.jit(jax.vmap(
        lambda t, p, r: potential.state_tp(t, p, density_bounds=(0.99 * r, 1.01 * r))
    ))(ts.ravel(), forward.P, rs.ravel())
    result["potential_tp_roundtrip"] = {
        "samples": int(ts.size),
        "bracket_fraction_of_target_density": [0.99, 1.01],
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(inverse.rho))),
        "max_relative_density_error": float(np.max(np.abs(inverse.rho / rs.ravel() - 1))),
        "max_relative_pressure_error": float(np.max(np.abs(inverse.P / forward.P - 1))),
    }

    # External AD checks do not use the stored response properties.
    def identities(t, r):
        state = potential.state_trho(t, r)
        ds_dr = jax.grad(lambda rho: potential.state_trho(t, rho).s)(r)
        dp_dt = jax.grad(lambda temp: potential.state_trho(temp, r).P)(t)
        du_dt = jax.grad(lambda temp: potential.state_trho(temp, r).u)(t)
        du_dr = jax.grad(lambda rho: potential.state_trho(t, rho).u)(r)
        return jnp.stack((
            (r**2 * ds_dr + dp_dt) / jnp.maximum(jnp.abs(dp_dt), 1.0),
            (du_dt - state.cv) / jnp.maximum(jnp.abs(state.cv), 1.0),
            (r**2 * du_dr + t * dp_dt - state.P) / jnp.maximum(jnp.abs(state.P), 1.0),
        ))

    tcheck, rcheck = np.meshgrid(np.geomspace(1100, 75000, 10), np.geomspace(220, 8500, 10))
    checks = jax.jit(jax.vmap(identities))(tcheck.ravel(), rcheck.ravel())
    result["max_scaled_identity_residuals_maxwell_cv_first_law"] = (
        np.max(np.abs(checks), axis=0).tolist()
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["Y0275", "Y0292", "Y0297"])
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise ValueError("Set JAX_ENABLE_X64=1 before running this audit.")
    report = {
        "method": "Local C2 biquintic Hermite of a/T in ln(T), ln(rho)",
        "reconstruction": "a=u-Ts; source P,u,s and three response constraints at nodes",
        "relative_residual_definition": "abs(potential-original)/abs(original); zero references excluded",
        "qmd_box_SI": {"temperature": [1000, 80000], "mass_density": [200, 9000]},
        "scope": "Backend differences and sampled local stability, not physical accuracy or a domain certificate",
        "jax_version": jax.__version__,
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
            for name in ("src/exoeos/helmholtz_table.py", "src/exoeos/state.py",
                         "src/exoeos/chabrier_debras.py", "examples/validate_helmholtz_table.py")
        },
        "variants": {},
    }
    for variant in args.variants:
        report["variants"][variant] = audit(
            ChabrierDebrasTableLoader(variant=variant, cache_directory=args.cache_directory)
        )
        print(f"Audited {variant}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
