"""Native JAX MELTS Fe(II)/Fe(III) silicate liquid and silicate solids.

The liquid basis is (SiO2, Mg2SiO4[, Fe2SiO4][, CaSiO3, Al2O3]), not oxides.
The six-component ferric basis appends Fe2O3 to CMFAS; DRY11_COMPONENTS
adds TiO2, MgCr2O4, Na2SiO3, KAlSiO4 and Ca3(PO4)2. Solid iron remains
Fe(II); olivine is the binary forsterite--fayalite face. No redox solve is done.
Temperatures are K, pressures Pa, amounts mol, and Gibbs energies J.
Enable JAX x64 for thermodynamic derivatives and phase comparisons.
ELEMENTS and element_matrix(components) define the conserved atom ledger:
rows are elements, columns are the caller's ordered component formulas.
See the shared property contract in documents/native_magma.rst for amount
bases, standard-state alignment, reduced units and derivative domains.

Equations and coefficients: MAGMA revision
705a0fb315e5054d18275a580562f6121c8e458c, sources/gibbs.c,
includes/{liq_struct_data,sol_struct_data,param_struct_data_v34}.h,
and sources/{olivine,clinopyroxene,orthopyroxene}.c (declared composition faces).
Silica additionally includes explicitly documented pinned-runtime offsets.
This is the MELTS (not pMELTS) polynomial EOS and liquid mixing model.
It does not establish stability against omitted phases or a calibrated
MgO--SiO2 phase diagram. See documents/native_magma.rst for provenance.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy


COMPONENTS = ("SiO2", "Mg2SiO4")
TERNARY_COMPONENTS = COMPONENTS + ("Fe2SiO4",)
CMAS_COMPONENTS = COMPONENTS + ("CaSiO3", "Al2O3")
CMFAS_COMPONENTS = TERNARY_COMPONENTS + ("CaSiO3", "Al2O3")
FERRIC_COMPONENTS = CMFAS_COMPONENTS + ("Fe2O3",)
DRY11_COMPONENTS = FERRIC_COMPONENTS + ("TiO2", "MgCr2O4", "Na2SiO3", "KAlSiO4", "Ca3(PO4)2")
DRY11_OXIDES = ("SiO2", "MgO", "FeO", "CaO", "Al2O3", "Fe2O3", "TiO2", "Cr2O3", "Na2O", "K2O", "P2O5")
OLIVINE_COMPONENTS = ("Mg2SiO4", "Fe2SiO4")
CLINOPYROXENE_COMPONENTS = ("CaMgSi2O6", "CaFeSi2O6")
PYROXENE_COMPONENTS = ("CaSiO3", "MgSiO3", "FeSiO3")
SILICA_POLYMORPHS = ("quartz", "tridymite", "cristobalite")
ELEMENTS = ("O", "Mg", "Si", "Fe", "Al", "Ca", "Na", "K", "Ti", "Cr", "P")
_ELEMENT_COUNTS = {
    # Atom counts per mol of each API coordinate, in ELEMENTS order.
    "SiO2": (2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    "Mg2SiO4": (4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    "Fe2SiO4": (4, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0),
    "CaSiO3": (3, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0),
    "Al2O3": (3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0),
    "Fe2O3": (3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0),
    "TiO2": (2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    "MgCr2O4": (4, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0),
    "Na2SiO3": (3, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0),
    "KAlSiO4": (4, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0),
    "Ca3(PO4)2": (8, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2),
    "MgO": (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    "FeO": (1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    "CaO": (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
    "Cr2O3": (3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0),
    "Na2O": (1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0),
    "K2O": (1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0),
    "P2O5": (5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2),
    "CaMgSi2O6": (6, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0),
    "CaFeSi2O6": (6, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0),
    "MgSiO3": (3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    "FeSiO3": (3, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    "CaAl2Si2O8": (8, 0, 2, 0, 2, 1, 0, 0, 0, 0, 0),
}
R = 8.3143  # J/(mol K); retain the MELTS entropy convention.
W = 3421.0  # J/mol; SiO2--Mg2SiO4 liquid interaction.
# Upper triangle in DRY11_COMPONENTS order; all 55 H interactions (J/mol).
# S and V interactions vanish on this dry face of param_struct_data_v34.h.
_LIQUID_W = tuple((0.,)*(i+1) + row for i, row in enumerate((
    (W, 23660.9, -863.7, -39120., 8110.3, 26266.7, 27886.3, -99039., -33921.7, 61891.6),
    (-37256.7, -31731.9, -32880.3, -71518.6, -4178.3, 46049.2, -41876.9, 22323.1, -23208.8),
    (-12970.8, -30509., -179064.9, 5209.1, -82971.8, -90533.8, 23649.4, 87410.3),
    (-57917.9, 12076.6, -35372.5, 30704.7, -13247.1, 17111.1, 37070.3),
    (-17089.4, -29449.8, -31770.3, -130785., -25859.2, 52220.8),
    (-84756.9, 21605.9, -149662.2, 57555.9, -4213.9),
    (-72303.4, -15415.6, -48094.6, 25938.8),
    (113646., 75709.1, 5341.8), (6522.8, 15571.9), (17100.6,), (),
)))
_TR = 298.15


def element_matrix(components):
    """Return integer atom counts A with element inventory = A @ n.

    Rows follow ELEMENTS; columns follow the supplied nonempty sequence of
    supported formula strings (e.g. CMFAS_COMPONENTS or PYROXENE_COMPONENTS).
    The shape is (len(ELEMENTS), len(components)). With n in mol of API
    coordinates, A @ n is mol of atoms. Pure anorthite uses ("CaAl2Si2O8",).
    Formulas may repeat when assembling multiple phases. Names are static
    metadata; obtain A before JAX transformations. This lookup neither maps
    oxide inventories into components nor validates a phase's site domain.
    """
    try:
        counts = tuple(_ELEMENT_COUNTS[name] for name in components)
    except KeyError as error:
        raise ValueError(f"Unsupported magma component: {error.args[0]}") from error
    if not counts:
        raise ValueError("components must contain at least one supported formula.")
    return jnp.asarray(counts, dtype=jnp.int32).T


def liquid_moles_from_oxides(m):
    """Map DRY11_OXIDES mol to DRY11_COMPONENTS mol, conserving every atom.

    This linear JAX transform checks shape (11,) only. The caller must
    validate finite nonnegative input AND output before evaluating G:
    MgO >= Cr2O3, CaO >= 3*P2O5, Al2O3 >= K2O, and residual SiO2 >= 0.
    Negative residuals are retained, never clipped or renormalized.
    FeO and Fe2O3 are supplied independently; this is not a redox solve.
    """
    m = jnp.asarray(m) * 1.0
    if m.shape != (11,):
        raise ValueError("m must have shape (11,) in DRY11_OXIDES order.")
    si, mg, fe, ca, al, ferric, ti, cr, na, k, p = m
    f, a, c = (mg-cr)/2, fe/2, ca-3*p
    return jnp.array([si-f-a-c-na-2*k, f, a, c, al-k, ferric, ti, cr, na, 2*k, p])


def oxide_moles_from_liquid(n):
    """Map DRY11_COMPONENTS mol to DRY11_OXIDES mol without normalization.

    Shape (11,) is checked; finite nonnegative amounts belong to the caller.
    This is the inverse linear transform of liquid_moles_from_oxides.
    """
    n = jnp.asarray(n) * 1.0
    if n.shape != (11,):
        raise ValueError("n must have shape (11,) in DRY11_COMPONENTS order.")
    q, f, a, c, al, ferric, ti, cr, na, k, p = n
    return jnp.array([q+f+a+c+na+k, 2*f+cr, 2*a, c+3*p,
                      al+k/2, ferric, ti, cr, na, k/2, p])


def _tp(T, P):
    temperature, pressure = jnp.asarray(T) * 1.0, jnp.asarray(P) * 1.0
    if temperature.ndim != 0 or pressure.ndim != 0:
        raise ValueError("T and P must be scalars; use jax.vmap for batches.")
    return temperature, pressure


def _berman_h_s(T, h0, s0, cp):
    # Cp = k0 + k1/sqrt(T) + k2/T**2 + k3/T**3, referenced to 298.15 K.
    k0, k1, k2, k3 = cp
    h = (h0 + k0 * (T - _TR) + 2 * k1 * (jnp.sqrt(T) - jnp.sqrt(_TR))
         - k2 * (1 / T - 1 / _TR) - k3 / 2 * (T**-2 - _TR**-2))
    s = (s0 + k0 * jnp.log(T / _TR) - 2 * k1 * (T**-.5 - _TR**-.5)
         - k2 / 2 * (T**-2 - _TR**-2) - k3 / 3 * (T**-3 - _TR**-3))
    return h, s


def _berman_gibbs(T, P, h0, s0, cp, v0, eos):
    h, s = _berman_h_s(T, h0, s0, cp)
    v1, v2, v3, v4 = eos
    dt, dp = T - _TR, P / 1e5 - 1
    return h - T * s + v0 * (
        (1 + v3 * dt + v4 * dt**2) * dp + v1 * dp**2 / 2 + v2 * dp**3 / 3)


def clinopyroxene_standard_gibbs(T, P):
    """Return [diopside, hedenbergite] standards in J/mol, at scalar K/Pa.

    These are the Ca(Mg,Fe)Si2O6 endpoints of MELTS clinopyroxene.
    Pure-reference mixing corrections vanish on this Ca-saturated face.
    No Ca-poor, Al-bearing or Na-bearing pyroxene components are included.
    """
    T, P = _tp(T, P)
    return _berman_gibbs(
        T, P, jnp.array([-3200583., -2842221.]), jnp.array([142.5, 174.2]),
        jnp.array([[305.41, 307.89], [-1604.9, -1597.3],
                   [-7.1660e6, -6.9925e6], [9.2184e8, 9.3522e8]]),
        jnp.array([6.620, 6.7894]),
        jnp.array([[-.872e-6, -.9925e-6], [1.707e-12, 1.4835e-12],
                   [27.795e-6, 31.371e-6], [83.082e-10, 83.672e-10]]))


def clinopyroxene_gibbs(T, P, n):
    """Return extensive Di--Hd clinopyroxene G for [n_Di, n_Hd] mol.

    The MELTS Ca-saturated face has one mixed Mg/Fe site, ideal site
    entropy and W=7029.12 J/mol. Ca occupies M2 and Si occupies the
    tetrahedral sites; no internal ordering degree of freedom remains.
    Zero amounts use continuous energy limits. Composition derivatives
    require both amounts positive. Input/x64 contracts match liquid_gibbs.
    """
    T, P = _tp(T, P)
    amounts = jnp.asarray(n) * 1.0
    if amounts.shape != (2,):
        raise ValueError("n must have shape (2,) in (CaMgSi2O6, CaFeSi2O6) order.")

    def present(values):
        x = values / values.sum()
        return (values @ clinopyroxene_standard_gibbs(T, P)
                + R * T * jnp.sum(xlogy(values, x)) + 7029.12 * values[0] * x[1])

    return jax.lax.cond(
        jnp.all(amounts == 0), lambda values: jnp.zeros_like(T + P + values.sum()),
        present, amounts)


def _pyroxene_coefficients(T, P, c, f, clino):
    # Ca--Mg--Fe restriction of the pinned Sack--Ghiorso site polynomial,
    # after subtracting the *monoclinic* pure references for BOTH phases.
    # Return [constant, b, b**2] coefficients, where b is M2 Fe occupancy.
    if clino:
        h = jnp.array([
            5857.6*c**3 - 11537.38*c*c*f - 33137.28*c*c - 679.9*c*f*f
            + 19299.6932816*c*f + 27279.68*c - 6349.22*f*f - 733.1932816*f,
            34287.88*c*c - 20710.8*c*f - 49654.6132816*c - 679.9*f*f
            + 14476.64*f - 4257.22,
            42781.4*c + 679.9*f - 12939.02])
        s = jnp.array([
            3.64848984*c*c*f + 2.895328*c*c - 1.48812328*c*f - 2.895328*c
            - 2.16036656*f,
            -3.64848984*c*c + 1.48812328*c + 2.16036656, 0.])
        v = jnp.array([
            -.066944*c**3 + .17781996532*c*c*f + .138072*c*c + .023012*c*f*f
            - .18156613064*c*f - .071128*c - .023012*f*f + .00374616532*f,
            -.42258396532*c*c + .175728*c*f + .44725016532*c + .023012*f*f
            - .037656,
            -.39748*c - .023012*f + .037656])
    else:
        h = jnp.array([
            10878.4*c**3 - 22593.6*c*c*f - 47948.64*c*c - 836.8*c*f*f
            + 27363.36*c*f + 44422.7832*c - 7531.2*f*f + 3316.6568*f - 5020.8,
            65270.4*c*c - 40166.4*c*f - 73554.72*c - 836.8*f*f
            + 18618.8*f - 6987.28,
            82006.4*c + 836.8*f - 19455.6])
        s = jnp.array([
            1.15955376*c*f + 3.7045136*c - 1.98798576*f - 2.3237936,
            1.15955376*(1-c), 0.])
        v = jnp.array([
            -.100416*c**3 + .23012*c*c*f + .117152*c*c + .029288*c*f*f
            - .3048082*c*f + .1204992*c - .043409*f*f + .111798*f - .0619232,
            -.602496*c*c + .284512*c*f + .5626472*c + .029288*f*f - .150624,
            -.6276*c - .029288*f + .029288])
    return h - T*s + (P/1e5-1)*v


def _pyroxene_order(T, c, f, coefficients):
    lower, upper = jnp.maximum(0., f-1), jnp.minimum(f, 1-c)

    def interior(_):
        def slope(b):
            return (coefficients[1] + 2*coefficients[2]*b + R*T*(
                jnp.log(1-f+b) + jnp.log(b) - jnp.log(f-b) - jnp.log(1-c-b)))

        def solve(fun, initial):
            def step(_, bracket):
                lo, hi = bracket
                mid = (lo+hi)/2
                positive = fun(mid) > 0
                return jnp.where(positive, lo, mid), jnp.where(positive, mid, hi)

            lo, hi = jax.lax.fori_loop(0, 64, step, (lower, upper))
            return (lo+hi)/2

        # Differentiate the stationarity equation, not the bisection decisions.
        return jax.lax.custom_root(slope, (lower+upper)/2, solve,
                                   lambda linear, rhs: rhs/linear(jnp.ones_like(rhs)))

    return jax.lax.cond(upper > lower, interior, lambda _: lower, operand=None)


def pyroxene_gibbs(T, P, n, *, phase="clinopyroxene"):
    """Return extensive Ca--Mg--Fe pyroxene G in J, with relaxed site order.

    ``n`` is [CaSiO3, MgSiO3, FeSiO3] in mol, for either ``clinopyroxene``
    or ``orthopyroxene``. These are physical inventory coordinates, not
    independent pure minerals: require nonnegative amounts and
    n_Ca <= n_Mg+n_Fe (at most one Ca on M2 per two-Si formula unit).
    The caller validates values; scalar K/Pa and x64 contracts match liquid G.

    The Sack--Ghiorso Ca--Mg--Fe face has Mg/Fe on M1 and Ca/Mg/Fe on M2.
    One bounded Fe site occupancy minimizes G. Its curvature is positive
    over 1200--2200 K and 1--5000 bar; no low-temperature branch selection
    is supplied. JIT, vmap and implicit first/second derivatives are supported
    inside this domain and the composition interior. At composition edges
    only continuous G and fixed-composition T/P derivatives are supported.
    An absent phase has exactly zero G. No Na, Al, Ti or Fe(III) is admitted.
    The original two-component ``clinopyroxene_gibbs`` API is unchanged.
    """
    T, P = _tp(T, P)
    amounts = jnp.asarray(n) * 1.0
    if amounts.shape != (3,):
        raise ValueError("n must have shape (3,) in (CaSiO3, MgSiO3, FeSiO3) order.")
    if phase not in ("clinopyroxene", "orthopyroxene"):
        raise ValueError("phase must be 'clinopyroxene' or 'orthopyroxene'.")

    def present(values):
        total = values.sum()/2  # mol of the two-Si pyroxene formula unit.
        c, f = values[0]/total, values[2]/total
        coefficients = _pyroxene_coefficients(T, P, c, f, phase == "clinopyroxene")
        b = _pyroxene_order(T, c, f, coefficients)
        sites = jnp.array([1-f+b, f-b, c, 1-c-b, b])
        mixing = coefficients @ jnp.array([1., b, b*b]) + R*T*jnp.sum(xlogy(sites, sites))
        di, hd = clinopyroxene_standard_gibbs(T, P)
        # Both MELTS structures use the same monoclinic En standard.
        en = 2*enstatite_gibbs(T, P) - (-5020.8 + 2.3237936*T - .0619232*(P/1e5-1))
        return total*((c-f)*di + (1-c)*en + f*hd + mixing)

    return jax.lax.cond(
        jnp.all(amounts == 0), lambda values: jnp.zeros_like(T + P + values.sum()),
        present, amounts)


def anorthite_gibbs(T, P):
    """Return pure CaAl2Si2O8 feldspar G in J/mol at scalar K/Pa.

    Uses the MELTS anorthite standard, including the tabulated Carpenter
    I1--C1 correction: dH=3.7*4184 J/mol and dS=dH/2200 J/(mol K).
    With Na and K absent, feldspar is this pure endpoint; this function
    is not an albite--anorthite or ternary feldspar solution model.
    Input/x64 contracts match forsterite_gibbs.
    """
    T, P = _tp(T, P)
    return _berman_gibbs(
        T, P, -4228730. + 3.7 * 4184., 200.186 + 3.7 * 4184. / 2200.,
        (439.37, -3734.1, 0., -3.1702e8), 10.075,
        (-1.272e-6, 3.176e-12, 10.918e-6, 41.985e-10))


def _olivine_h_s(T, *, iron=False):
    # Integrate Cp = k0 + k1/sqrt(T) + k3/T**3 from the Berman reference.
    h0, s0, k0, k1, k3 = ((-1479360., 150.930, 248.93, -1923.9, -1.3910e8)
                          if iron else (-2174420., 94.010, 238.64, -2001.3, -1.1624e8))
    h = (h0 + k0 * (T - _TR)
         + 2.0 * k1 * (jnp.sqrt(T) - jnp.sqrt(_TR))
         - 0.5 * k3 * (T**-2 - _TR**-2))
    s = (s0 + k0 * jnp.log(T / _TR)
         - 2.0 * k1 * (T**-0.5 - _TR**-0.5)
         - k3 / 3.0 * (T**-3 - _TR**-3))
    return h, s


def forsterite_gibbs(T, P):
    """Return crystalline Mg2SiO4 standard Gibbs energy in J/mol.

    ``T`` and ``P`` are scalars in K and Pa. Positive finite inputs must
    be validated by the caller outside JAX transformations. Pressure uses
    the Berman polynomial relative to 1 bar, with volume in J/bar/mol.
    """
    T, P = _tp(T, P)
    h, s = _olivine_h_s(T)
    dt, dp = T - _TR, P / 1e5 - 1.0
    pressure_g = 4.366 * (
        (1.0 + 29.464e-6 * dt + 88.633e-10 * dt**2) * dp
        - 0.791e-6 * dp**2 / 2.0 + 1.351e-12 * dp**3 / 3.0
    )
    return h - T * s + pressure_g


def fayalite_gibbs(T, P):
    """Return the olivine Fe2SiO4 standard G in J/mol at scalar K/Pa.

    Uses the Berman EOS of the olivine endmember, not the separate pure
    fayalite phase's Vinet EOS. Input/x64 contracts match forsterite_gibbs.
    """
    T, P = _tp(T, P)
    h, s = _olivine_h_s(T, iron=True)
    dt, dp = T - _TR, P / 1e5 - 1.0
    return h - T * s + 4.630 * (
        (1.0 + 26.546e-6 * dt + 79.482e-10 * dt**2) * dp
        - 0.730e-6 * dp**2 / 2.0
    )


def olivine_gibbs(T, P, n):
    """Return extensive binary olivine G in J for [n_Fo, n_Fa] in mol.

    The Fe/Mg face of MELTS olivine.c has equal occupancies on its two
    cation sites at magmatic temperatures. Its entropy is twice the ideal
    endmember entropy and W = 20300 + 0.015*(P/bar - 1) J/mol. No redox,
    Ca, Mn, Ni or Co is included. Zero amounts use continuous energy limits;
    composition derivatives require both amounts positive, as for liquid G.
    """
    T, P = _tp(T, P)
    amounts = jnp.asarray(n) * 1.0
    if amounts.shape != (2,):
        raise ValueError("n must have shape (2,) in (Mg2SiO4, Fe2SiO4) order.")

    def present(values):
        x = values / values.sum()
        standards = jnp.stack((forsterite_gibbs(T, P), fayalite_gibbs(T, P)))
        interaction = 20300.0 + 0.015 * (P / 1e5 - 1.0)
        return (values @ standards + 2 * R * T * jnp.sum(xlogy(values, x))
                + interaction * values[0] * x[1])

    return jax.lax.cond(
        jnp.all(amounts == 0), lambda values: jnp.zeros_like(T + P + values.sum()),
        present, amounts,
    )


def enstatite_gibbs(T, P):
    """Return pure Mg orthopyroxene G in J per mol of MgSiO3.

    Scalar ``T`` is K and ``P`` is Pa, with the same input/x64 contract
    as ``forsterite_gibbs``. This is the MELTS orthopyroxene endpoint,
    not a minimum over enstatite polymorphs or a pyroxene solid solution.
    The native Mg2Si2O6 standard is monoclinic. Add the orthopyroxene
    endpoint minus its monoclinic pure-reference mixing energy before
    dividing by two. Coefficients follow sol_struct_data.h and
    orthopyroxene.c at the module's pinned MAGMA revision.
    """
    T, P = _tp(T, P)
    dt, dp = T - _TR, P / 1e5 - 1.0
    k0, k1, k2, k3 = 333.16, -2401.2, -4.5412e6, 5.5830e8
    h = (-3086083.0 + k0 * dt + 2.0 * k1 * (jnp.sqrt(T) - jnp.sqrt(_TR))
         - k2 * (1.0 / T - 1.0 / _TR) - k3 / 2.0 * (T**-2 - _TR**-2))
    s = (135.164 + k0 * jnp.log(T / _TR)
         - 2.0 * k1 * (T**-0.5 - _TR**-0.5)
         - k2 / 2.0 * (T**-2 - _TR**-2) - k3 / 3.0 * (T**-3 - _TR**-3))
    pressure_g = 6.3279 * (
        (1.0 + 24.656e-6 * dt + 74.670e-10 * dt**2) * dp
        - 0.749e-6 * dp**2 / 2.0 + 0.447e-12 * dp**3 / 3.0
    )
    # Per mol Mg2Si2O6: dH=-5020.8 J, dS=-2.3237936 J/K,
    # dV=-0.0619232 J/bar. No ordering freedom remains at the pure Mg end.
    correction = -5020.8 + 2.3237936 * T - 0.0619232 * dp
    return (h - T * s + pressure_g + correction) / 2.0


def silica_gibbs(T, P):
    """Return [quartz, tridymite, cristobalite] G in J/mol SiO2.

    Scalar K/Pa and x64 contracts match the other standards. Each phase
    includes the MELTS alpha/beta switch (alpha at equality), its lambda
    heat-capacity integral and Berman EOS. Quartz and cristobalite switch
    temperatures depend on pressure. Derivatives apply within each branch;
    the tabulated branches are not smoothed at their transitions. Constant
    offsets [-1291, -2625, -450] J/mol match the pinned rhyolite-MELTS 1.0.2
    runtime. Only the quartz offset is present in the public source; the
    other two are runtime transcriptions, independently checked in the
    saved reference. See documents/native_magma.rst for provenance.
    These three low-pressure polymorphs omit coesite and stishovite.
    """
    T, P = _tp(T, P)
    dt, dp = T - _TR, P / 1e5 - 1.0
    k0, k1, k2, k3 = jnp.array([
        [80.01, 75.37, 83.51], [-240.3, 0., -374.7],
        [-35.467e5, -59.581e5, -24.554e5], [49.157e7, 95.825e7, 28.007e7],
    ])
    shift = -jnp.array([.0237, 0., .0480]) * dp
    beta = T > jnp.array([848., 383., 535.]) - shift
    h = jnp.where(beta, jnp.array([-908627., -907045., -906377.]),
                  jnp.array([-910700., -907750., -907753.]))
    s = jnp.where(beta, jnp.array([44.207, 45.524, 46.029]),
                  jnp.array([41.460, 43.770, 43.394]))
    h = (h + k0 * dt + 2*k1*(jnp.sqrt(T) - jnp.sqrt(_TR))
         - k2*(1/T - 1/_TR) - k3/2*(T**-2 - _TR**-2))
    s = (s + k0*jnp.log(T/_TR) - 2*k1*(T**-.5 - _TR**-.5)
         - k2/2*(T**-2 - _TR**-2) - k3/3*(T**-3 - _TR**-3))
    # Expand Cp_lambda = (T+shift) * (l1+l2*(T+shift))**2.
    l1 = jnp.array([-.09187, .42670, -.14216])
    l2 = jnp.array([24.607e-5, -144.575e-5, 44.142e-5])
    x1 = shift*(l1 + l2*shift)**2
    x2 = l1**2 + 4*l1*l2*shift + 3*l2**2*shift**2
    x3, x4 = 2*l1*l2 + 3*l2**2*shift, l2**2
    lower = jnp.array([373., _TR, _TR]) - shift
    dh = (x1*(T-lower) + x2/2*(T**2-lower**2)
          + x3/3*(T**3-lower**3) + x4/4*(T**4-lower**4))
    ds = (x1*jnp.log(T/lower) + x2*(T-lower)
          + x3/2*(T**2-lower**2) + x4/3*(T**3-lower**3))
    v, v1, v2, v3 = jnp.where(beta[None, :], jnp.array([
        [2.370, 2.737, 2.730], [-1.238e-6, -.740e-6, -1.100e-6],
        [7.087e-13, 3.735e-12, 5.535e-12], [0., 4.829e-6, 3.189e-6],
    ]), jnp.array([
        [2.269, 2.675, 2.587], [-2.434e-6, -2.508e-6, -2.515e-6],
        [10.137e-12, 0., 0.], [23.895e-6, 19.339e-6, 20.824e-6],
    ]))
    pressure_g = v*((1 + v3*dt)*dp + v1*dp**2/2 + v2*dp**3/3)
    return (h - T*s + jnp.where(beta, 0., dh - T*ds) + pressure_g
            - jnp.array([1291., 2625., 450.]))


def _silica_h_s(T):
    # The liquid SiO2 special case is not its tabulated fusion model.
    a, b, c, d = 127.200, -10.777e-3, 4.3127e5, -1463.8

    def below(t):
        h = (-901554.0 + a * (t - _TR) + b * (t**2 - _TR**2) / 2.0
             - c * (1.0 / t - 1.0 / _TR)
             + 2.0 * d * (jnp.sqrt(t) - jnp.sqrt(_TR)))
        s = (48.475 + a * jnp.log(t / _TR) + b * (t - _TR)
             - c / 2.0 * (t**-2 - _TR**-2)
             - 2.0 * d * (t**-0.5 - _TR**-0.5))
        return h, s

    def above(t):
        h, s = below(1480.0)
        return h + 81.373 * (t - 1480.0), s + 81.373 * jnp.log(t / 1480.0)

    return jax.lax.cond(T >= 1480.0, above, below, T)


def liquid_standard_gibbs(T, P, *, include_fe=False, include_ca_al=False,
                          include_ferric=False, include_dry11=False):
    """Return liquid standards in J/mol in the selected component order.

    The SiO2 branch switches at 1480 K, preserving G and its first
    temperature derivative; its heat capacity can jump there. The
    Mg2SiO4 standard uses fusion at 2163 K and constant liquid Cp.
    Static ``include_fe=True`` appends Fe2SiO4 (fusion 1490 K, liquid
    Cp=240.2 J/mol/K). Default shape (2,) preserves the binary API.
    Static ``include_ca_al=True`` appends CaSiO3 and Al2O3, yielding
    shape (4,) or (5,). They use fusion at 1817 and 2319.65 K with
    constant liquid Cp=172.4 and 170.3 J/mol/K, respectively.
    Static ``include_ferric=True`` requires both other flags and appends
    Fe2O3, giving FERRIC_COMPONENTS. Its fusion construction includes the
    solid reference's 955 K transition, fusion at 1895 K, and liquid
    Cp=240.9 J/mol/K. Fe(II)/Fe(III) amounts are supplied, not equilibrated.
    Static ``include_dry11=True`` selects DRY11_COMPONENTS, implying all
    three other flags. The five additional standards use the same fusion
    construction, including the KAlSiO4 and Ca3(PO4)2 reference transitions.
    Input validation and x64 configuration belong to the caller.
    """
    if include_dry11:
        include_fe = include_ca_al = include_ferric = True
    if include_ferric and not (include_fe and include_ca_al):
        raise ValueError("include_ferric requires include_fe=True and include_ca_al=True.")
    T, P = _tp(T, P)
    h_q, s_q = _silica_h_s(T)
    h_f, s_f = _olivine_h_s(2163.0)
    h_f = h_f + 2163.0 * 57.2 + 271.0 * (T - 2163.0)
    s_f = s_f + 57.2 + 271.0 * jnp.log(T / 2163.0)
    thermal = jnp.stack((h_q - T * s_q, h_f - T * s_f))
    # Kress EOS, integrated in bar from 1 bar. Reference T is 1673 K.
    dt, dp = T - 1673.0, P / 1e5 - 1.0
    v, dvdt, dvdp, d2vdtp, d2vdp2 = jnp.array([
        [2.690, 4.980], [0.0, 5.24e-4], [-1.89e-5, -1.35e-5],
        [1.3e-8, -1.3e-8], [3.6e-10, 4.14e-10],
    ], dtype=thermal.dtype)
    if include_fe:
        h_a, s_a = _olivine_h_s(1490.0, iron=True)
        h_a = h_a + 1490.0 * 59.9 + 240.2 * (T - 1490.0)
        s_a = s_a + 59.9 + 240.2 * jnp.log(T / 1490.0)
        thermal = jnp.append(thermal, h_a - T * s_a)
        v, dvdt, dvdp, d2vdtp, d2vdp2 = (
            jnp.append(values, extra) for values, extra in zip(
                (v, dvdt, dvdp, d2vdtp, d2vdp2),
                (5.420, 5.84e-4, -2.79e-5, -2.3e-8, 14.6e-10)))
    if include_ca_al:
        tm = jnp.array([1817., 2319.65])
        ds = jnp.array([31.5, 48.61])
        cp = jnp.array([172.4, 170.3])
        h, s = _berman_h_s(
            tm, jnp.array([-1627427., -1675700.]), jnp.array([85.279, 50.82]),
            jnp.array([[141.16, 155.02], [-417.2, -828.4],
                       [-5.8576e6, -3.8614e6], [9.4074e8, 4.0908e8]]))
        thermal = jnp.concatenate((thermal, h + tm * ds + cp * (T - tm)
                                   - T * (s + ds + cp * jnp.log(T / tm))))
        v, dvdt, dvdp, d2vdtp, d2vdp2 = (
            jnp.concatenate((values, jnp.array(extra))) for values, extra in zip(
                (v, dvdt, dvdp, d2vdtp, d2vdp2),
                ([4.347, 3.711], [2.92e-4, 2.62e-4], [-1.55e-5, -2.26e-5],
                 [-1.6e-8, 2.7e-8], [3.89e-10, 4.0e-10])))
    if include_ferric:
        h, s = _berman_h_s(1895., -822000., 87.40, (146.86, 0., -5.5768e6, 5.2563e8))
        # Integrate Cp_lambda = T*(l1+l2*T)**2 to 955 K, then add the
        # transition latent heat. The liquid has no branch at 955 K.
        tc, l1, l2 = 955., -7.403e-2, 27.921e-5
        h += (1287. + l1*l1*(tc**2-_TR**2)/2
              + 2*l1*l2*(tc**3-_TR**3)/3 + l2*l2*(tc**4-_TR**4)/4)
        s += (1287./tc + l1*l1*(tc-_TR)
              + l1*l2*(tc**2-_TR**2) + l2*l2*(tc**3-_TR**3)/3)
        thermal = jnp.append(thermal, h + 1895.*60.41 + 240.9*(T-1895.)
                             - T*(s + 60.41 + 240.9*jnp.log(T/1895.)))
        v, dvdt, dvdp, d2vdtp, d2vdp2 = (
            jnp.append(values, extra) for values, extra in zip(
                (v, dvdt, dvdp, d2vdtp, d2vdp2),
                (4.213, 9.09e-4, -2.53e-5, 3.1e-8, 4.4e-10)))
    if include_dry11:
        tm = jnp.array([1870., 2673.15, 1361., 2023.15, 1943.15])
        ds = jnp.array([35.824, 73.22, 38.34, 24.5, 35.690])
        cp = jnp.array([109.2, 335.1, 180.2, 217., 574.67])
        h, s = _berman_h_s(
            tm, jnp.array([-944750., -1783640., -373190.*4.184, -2111813.55, -4097169.]),
            jnp.array([50.460, 106.02, 27.21*4.184, 133.9653, 235.978]),
            jnp.array([[77.84, 201.981, 234.77, 186., 402.997],
                       [0., -551.9, -2218.9, 0., -2808.35],
                       [-33.678e5, -57.844e5, 0., -131.067e5, 0.],
                       [40.294e7, 57.729e7, 13.530e7, 213.893e7, -32.6230e7]]))
        # K and P solid-reference anomalies end below fusion. These add
        # constants to liquid H/S, not liquid branches at the transitions.
        tc, latent = jnp.array([800.15, 1373.]), jnp.array([1154., 14059.])
        l1, l2 = jnp.array([-7.096454e-2, 2.5427e-2]), jnp.array([21.682e-5, 19.255e-5])
        h = h.at[3:].add(latent + l1**2*(tc**2-_TR**2)/2
                        + 2*l1*l2*(tc**3-_TR**3)/3 + l2**2*(tc**4-_TR**4)/4)
        s = s.at[3:].add(latent/tc + l1**2*(tc-_TR)
                        + l1*l2*(tc**2-_TR**2) + l2**2*(tc**3-_TR**3)/3)
        thermal = jnp.concatenate((thermal, h + tm*ds + cp*(T-tm)
                                   - T*(s + ds + cp*jnp.log(T/tm))))
        v, dvdt, dvdp, d2vdtp, d2vdp2 = (
            jnp.concatenate((values, jnp.array(extra))) for values, extra in zip(
                (v, dvdt, dvdp, d2vdtp, d2vdp2),
                ([2.316, 5.358, 5.568, 6.8375, 10.7382],
                 [7.246e-4, 11.71e-4, 7.41e-4, 7.265e-4, 0.],
                 [-2.310e-5, -2.26e-5, -4.29e-5, -6.395e-5, 0.],
                 [0., 1.8e-8, -5.3e-8, -4.6e-8, 0.],
                 [5.0e-10, 4.67e-10, 8.4e-10, 12.1e-10, 0.])))
    return (thermal + (v + dvdt * dt) * dp
            + (dvdp + d2vdtp * dt) * dp**2 / 2.0
            + d2vdp2 * dp**3 / 6.0)


def liquid_gibbs(T, P, n):
    """Return liquid G in J on a supported two- to eleven-component basis.

    Shape (2,) is COMPONENTS, (3,) TERNARY_COMPONENTS, (4,) CMAS_COMPONENTS,
    (5,) CMFAS_COMPONENTS, and (6,) FERRIC_COMPONENTS (CMFAS plus Fe2O3).
    Shape (11,) selects DRY11_COMPONENTS, with all 55 pair interactions.
    Supply nonnegative finite mol amounts and positive finite
    scalar K/Pa. Only static shapes are checked here. Zero component
    amounts use the continuous energy limit, including G(T,P,[0,0])=0.
    Composition derivatives are supported for strictly positive amounts;
    an absent phase has no chemical potentials, and an absent component's
    ideal chemical potential tends to minus infinity. No floor is added.
    For the existing reduced convention, divide by the caller's R*T.
    """
    T, P = _tp(T, P)
    amounts = jnp.asarray(n) * 1.0
    if amounts.shape not in ((2,), (3,), (4,), (5,), (6,), (11,)):
        raise ValueError("n must have shape (2,), (3,), (4,), (5,), (6,) or (11,) in the documented liquid order.")
    size = amounts.size
    indices = jnp.array((0, 1, 3, 4) if size == 4 else tuple(range(size)))
    matrix = jnp.asarray(_LIQUID_W)[indices[:, None], indices[None, :]]

    def present(values):
        x = values / jnp.sum(values)
        return (jnp.dot(values, liquid_standard_gibbs(
                    T, P, include_fe=size in (3, 5, 6, 11), include_ca_al=size >= 4,
                    include_ferric=size >= 6, include_dry11=size == 11))
                + R * T * jnp.sum(xlogy(values, x))
                + values @ matrix @ x)

    return jax.lax.cond(
        jnp.all(amounts == 0), lambda values: jnp.zeros_like(T + P + values.sum()),
        present, amounts,
    )
