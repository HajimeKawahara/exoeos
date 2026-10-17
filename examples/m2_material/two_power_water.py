"""Check the generic two-power scalar against an independent monotone inverse."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from exoeos._water_isotherm import two_power_water_g_rt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    args.output.mkdir(parents=True, exist_ok=True)
    c, a = np.array([5e-5, 6e-4]), np.array([.8, .45])
    w = np.geomspace(1e-12, .5, 100)
    h = w/(1-w)
    scalar = lambda h: two_power_water_g_rt(h, 1., c, a)
    mu = np.asarray(jax.jit(jax.vmap(jax.grad(scalar)))(jnp.asarray(h)))
    expected = np.array([brentq(lambda z: np.logaddexp.reduce(np.log(c)+a*z)-np.log(x), -200., 200.) for x in w])
    error = mu-expected
    np.savetxt(args.output/"derivative.csv", np.column_stack((w, mu, expected, error)), delimiter=",",
               header="water_mass_fraction,mu_from_scalar_rt,ln_pressure_from_independent_inverse,error_rt", comments="")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    axes[0].semilogx(w, expected, label="Independent inverse")
    axes[0].semilogx(w[::5], mu[::5], "o", ms=3, label="dG/dn from JAX")
    axes[0].set(xlabel="Water mass fraction", ylabel="Water chemical potential / RT", title="Generic two-power isotherm")
    axes[0].legend(fontsize=8)
    axes[1].semilogx(w, error)
    axes[1].set(xlabel="Water mass fraction", ylabel="Derivative minus inverse [RT]", title="Numerical consistency; no calibration claim")
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(args.output/"derivative.png", dpi=180)
    print("maximum absolute error RT", np.max(np.abs(error)))


if __name__ == "__main__":
    main()
