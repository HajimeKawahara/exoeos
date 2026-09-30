"""Compare the explicitly selected published callback with native controls."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import melts_liquid_evaluator as native
import melts_liquid_mixing as mixing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-properties", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path.")
    raw = args.host_properties.read_bytes()
    source = json.loads(raw)
    t, p = source["T_K"], source["P_Pa"]
    common_r = source["basis"]["common_R_J_mol_K"]
    provider = mixing.make_published_liquid_evaluator(native, runtime=args.runtime,
                python_executable=args.python, common_R=common_r)
    n = np.asarray(source["component_moles"])
    started = time.perf_counter()
    provider.evaluate_liquid(t, p, n, common_R=common_r)
    initial_seconds = time.perf_counter()-started
    probes = [("host", n), ("scaled_small", 1e-8*n), ("scaled_large", 1e8*n)]
    probes += [("pure_"+name, np.eye(19)[i]) for i, name in enumerate(native.COMPONENTS)
               if i not in mixing.PARAMETERS["unsupported_positive_indices"]]
    for first, second in ((0, 7), (0, 18), (3, 5), (7, 11)):
        for fraction in (1e-12, 1e-6, .5, 1-1e-6):
            probes.append((f"binary_{first}_{second}_{fraction}",
                           fraction*np.eye(19)[first]+(1-fraction)*np.eye(19)[second]))
    rows = []
    for label, amounts in probes:
        published = provider.evaluate_liquid(t, p, amounts, common_R=common_r)
        present = amounts > 0
        euler = published["gibbs_RT"]-float(amounts[present]@np.asarray(published["mu_RT"], float)[present])
        row = {"label": label, "published_properties": published,
               "published_euler_residual_rt_per_mol": euler/amounts.sum()}
        try:
            receipt = native.evaluate_liquid(t, p, amounts, runtime=args.runtime,
                        python_executable=args.python, common_R=common_r)
            row.update(native_properties=receipt, native_status="available",
                       energy_difference_rt_per_mol=(published["gibbs_RT"]-receipt["gibbs_RT"])/amounts.sum(),
                       maximum_potential_difference_rt=float(np.nanmax(np.abs(
                           np.asarray(published["mu_RT"], float)-np.asarray(receipt["mu_RT"], float)))))
        except (ValueError, RuntimeError) as error:
            row.update(native_status="unavailable", native_reason=str(error))
        rows.append(row)
        print(label, row["native_status"], row.get("maximum_potential_difference_rt"), flush=True)
    started = time.perf_counter()
    for _ in range(1000):
        provider.evaluate_liquid(t, p, n, common_R=common_r)
    elapsed = time.perf_counter()-started
    result = {"command": sys.argv, "source_sha256": hashlib.sha256(raw).hexdigest(),
              "exoeos_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model_id": provider.MODEL_ID, "checks": rows,
              "standard_state_receipts": provider.standard_state_receipts,
              "first_call_seconds": initial_seconds, "cached_1000_calls_seconds": elapsed,
              "native_standard_calls": len(provider.standard_state_receipts),
              "native_binary_error_bound_certified": False, "empirical_calibration": False}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
