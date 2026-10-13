"""Independent pure-water standards and Zhang-Duan pressure corrections.

Requires a C compiler, NumPy and SciPy only for regeneration; never imports
JAX or ExoEOS. Ordinary tests read magma_fluid_v1.json offline.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


REVISION = "705a0fb315e5054d18275a580562f6121c8e458c"
WATER_SHA256 = "5ee8b5d59eb813c50fd66d2a9b47b317a0cde327c6ac924967d2521f770a8b08"
R, RH = 8.31446261815324, 1.80152*4.6152
# Zhang and Duan (2009), Eq. 8 and Tables 3/4; density in the paper's scale.
COEFFICIENTS = np.array([
    .0295177298930, -6337.56452413, -275265.428882,
    .00129128089283, -145.797416153, 76593.8947237,
    .00000258661493537, .521265321460, -139.839523753,
    -.0000000236335007175, .00535026383543, -.271106499510,
    25038.7836486, .732267260410, .0154833359970,
])
SCALE = (2.88/3.691)**3/1000
FIELDS = ("g_J_mol", "s_J_mol_K", "cp_J_mol_K", "v_m3_mol",
          "dvdt_m3_mol_K", "dvdp_m3_mol_Pa")


def zd_state(t, p):
    tr = 154*t/510
    b, c, d, e = [a[0]+a[1]/tr**2+a[2]/tr**3
                  for a in COEFFICIENTS[:12].reshape(4, 3)]
    f, beta, gamma = COEFFICIENTS[12:]

    def zm1(q):
        return (b*q+c*q**2+d*q**4+e*q**5
                + f/tr**3*q**2*(beta+gamma*q**2)*np.exp(-gamma*q**2))

    def residual(q):
        return q*(1+zm1(q))*R*t/SCALE-p

    grid = np.r_[0., np.geomspace(1e-8, 128., 1000)]
    upper = np.flatnonzero([residual(q) >= 0 for q in grid])[0]
    q = brentq(residual, grid[upper-1], grid[upper], xtol=1e-14)
    z = 1+zm1(q)
    # Integrate the published compressibility, independently of the package
    # residual Helmholtz primitive and implicit-root implementation.
    ar = quad(lambda x: zm1(x)/x if x else b, 0., q, epsabs=1e-12)[0]
    return {"rho_mol_m3": q/SCALE, "Z": z, "lnphi": ar+z-1-np.log(z)}


def first(fun, x, h):
    return (fun(x-2*h)-8*fun(x-h)+8*fun(x+h)-fun(x+2*h))/(12*h)


def second(fun, x, h):
    return (-fun(x-2*h)+16*fun(x-h)-30*fun(x)+16*fun(x+h)-fun(x+2*h))/(12*h*h)


def generate(source):
    source = source.read_bytes()
    if hashlib.sha256(source).hexdigest() != WATER_SHA256:
        raise ValueError("Pinned water.c SHA-256 mismatch.")
    shim = """#include <math.h>
#include <float.h>
#include <stdio.h>
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
#define ABS(a) fabs(a)
#define MODE__MELTS 1
#define MODE_xMELTS 2
#define MODE__MELTSandCO2 3
#define MODE__MELTSandCO2_H2O 4
static int calculationMode = MODE__MELTS;
"""
    wrapper = """
void reference(double t, double p, double *out) {
  double g,h,s,cp,dcp,v,vt,vp,vtt,vtp,vpp;
  whaar(p,t,&g,&h,&s,&cp,&dcp,&v,&vt,&vp,&vtt,&vtp,&vpp);
  out[0]=g; out[1]=s; out[2]=cp;
}
"""
    standards = []
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        (directory/"silmin.h").write_text(shim)
        (directory/"reference.c").write_bytes(source+wrapper.encode())
        subprocess.run(["cc", "-shared", "-fPIC", "-O2", str(directory/"reference.c"),
                        "-lm", "-o", str(directory/"reference.so")], check=True)
        library = ctypes.CDLL(str(directory/"reference.so"))
        library.reference.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        library.reference.restype = None

        def haar(t, p):
            out = (ctypes.c_double*3)()
            library.reference(t, p, out)
            return np.array(out)

        for t in (700., 1000., 1200., 1479., 1480., 1481., 1700., 1800., 2200., 2500.):
            dilute = []
            for p in (.001, .0005):
                corrected = haar(t, p)+np.array([-RH*t*np.log(p), RH*np.log(p), 0.])
                dilute.append(corrected)
            limit = 2*dilute[1]-dilute[0]
            standards.append({"T_K": t, **dict(zip(FIELDS[:3], limit)),
                              "haar_real_1bar_J_mol": haar(t, 1.)[0],
                              "dilute_corrected_properties": [v.tolist() for v in dilute]})

    states = []
    for standard in standards:
        t = standard["T_K"]
        for p in (1e5, 5e7, 5e8):
            state = zd_state(t, p)
            def correction(ti):
                return R*ti*(np.log(p/1e5)+zd_state(ti, p)["lnphi"])
            def volume(ti, pi):
                return 1/zd_state(ti, pi)["rho_mol_m3"]
            values = (standard["g_J_mol"]+correction(t),
                      standard["s_J_mol_K"]-first(correction, t, .5),
                      standard["cp_J_mol_K"]-t*second(correction, t, .5),
                      volume(t, p), first(lambda ti: volume(ti, p), t, .5),
                      first(lambda pi: volume(t, pi), p, p*1e-4))
            states.append({"T_K": t, "P_Pa": p, **state, **dict(zip(FIELDS, values))})
    return {
        "reference_kind": "compiled_Haar_dilute_limit_and_independent_ZD09_quadrature",
        "standard_source": {"revision": REVISION, "path": "sources/water.c", "sha256": WATER_SHA256},
        "eos_source": "https://doi.org/10.1016/j.gca.2009.01.021",
        "method": "Unmodified compiled water.c at 0.001/0.0005 bar, remove RH*T*ln(p/bar), extrapolate linearly to zero pressure. ZD09 Eq.8/Tables3-4 in NumPy; SciPy brentq and quadrature of (Z-1)/rho. Thermal residual and volume derivatives: five-point differences (0.5 K, 0.0001 P). No JAX/ExoEOS imports.",
        "constants": {"EOS_R_J_mol_K": R, "Haar_R_J_mol_K": RH, "P0_Pa": 1e5},
        "standards": standards, "states": states,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--water-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(generate(args.water_source), indent=2, allow_nan=False)+"\n")
