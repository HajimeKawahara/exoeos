"""Native JAX MELTS CaO--MgO--FeO--Al2O3--SiO2 liquid and silicate solids.

The liquid basis is (SiO2, Mg2SiO4[, Fe2SiO4][, CaSiO3, Al2O3]), not oxides.
Iron is exclusively Fe(II). Olivine is the binary forsterite--fayalite face.
Temperatures are K, pressures Pa, amounts mol, and Gibbs energies J.
Enable JAX x64 for thermodynamic derivatives and phase comparisons.

Equations and coefficients: MAGMA revision
705a0fb315e5054d18275a580562f6121c8e458c, sources/gibbs.c,
includes/{liq_struct_data,sol_struct_data,param_struct_data_v34}.h,
and sources/{olivine,clinopyroxene}.c (declared binary faces).
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
OLIVINE_COMPONENTS = ("Mg2SiO4", "Fe2SiO4")
CLINOPYROXENE_COMPONENTS = ("CaMgSi2O6", "CaFeSi2O6")
SILICA_POLYMORPHS = ("quartz", "tridymite", "cristobalite")
R = 8.3143  # J/(mol K); retain the MELTS entropy convention.
W = 3421.0  # J/mol; SiO2--Mg2SiO4 liquid interaction.
_TR = 298.15


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


def liquid_standard_gibbs(T, P, *, include_fe=False, include_ca_al=False):
    """Return [SiO2, Mg2SiO4[, Fe2SiO4][, CaSiO3, Al2O3]] G in J/mol.

    The SiO2 branch switches at 1480 K, preserving G and its first
    temperature derivative; its heat capacity can jump there. The
    Mg2SiO4 standard uses fusion at 2163 K and constant liquid Cp.
    Static ``include_fe=True`` appends Fe2SiO4 (fusion 1490 K, liquid
    Cp=240.2 J/mol/K). Default shape (2,) preserves the binary API.
    Static ``include_ca_al=True`` appends CaSiO3 and Al2O3, yielding
    shape (4,) or (5,). They use fusion at 1817 and 2319.65 K with
    constant liquid Cp=172.4 and 170.3 J/mol/K, respectively.
    Input validation and x64 configuration belong to the caller.
    """
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
    return (thermal + (v + dvdt * dt) * dp
            + (dvdp + d2vdtp * dt) * dp**2 / 2.0
            + d2vdp2 * dp**3 / 6.0)


def liquid_gibbs(T, P, n):
    """Return liquid G in J on the binary, ternary, CMAS or CMFAS basis.

    Shape (2,) is COMPONENTS, (3,) TERNARY_COMPONENTS, (4,) CMAS_COMPONENTS,
    and (5,) CMFAS_COMPONENTS. Supply nonnegative finite mol amounts and positive finite
    scalar K/Pa. Only static shapes are checked here. Zero component
    amounts use the continuous energy limit, including G(T,P,[0,0])=0.
    Composition derivatives are supported for strictly positive amounts;
    an absent phase has no chemical potentials, and an absent component's
    ideal chemical potential tends to minus infinity. No floor is added.
    For the existing reduced convention, divide by the caller's R*T.
    """
    T, P = _tp(T, P)
    amounts = jnp.asarray(n) * 1.0
    if amounts.shape not in ((2,), (3,), (4,), (5,)):
        raise ValueError("n must have shape (2,), (3,), (4,) or (5,) in the documented liquid order.")
    include_fe = amounts.shape in ((3,), (5,))
    include_ca_al = amounts.shape in ((4,), (5,))

    def present(values):
        x = values / jnp.sum(values)
        excess = W * values[0] * x[1]
        if include_fe:
            excess += 23660.9 * values[0] * x[2] - 37256.7 * values[1] * x[2]
        if include_ca_al:
            excess += (-863.7 * values[0] * x[-2] - 39120.0 * values[0] * x[-1]
                       - 31731.9 * values[1] * x[-2] - 32880.3 * values[1] * x[-1]
                       - 57917.9 * values[-2] * x[-1])
            if include_fe:
                excess += -12970.8 * values[2] * x[-2] - 30509.0 * values[2] * x[-1]
        return (jnp.dot(values, liquid_standard_gibbs(
                    T, P, include_fe=include_fe, include_ca_al=include_ca_al))
                + R * T * jnp.sum(xlogy(values, x))
                + excess)

    return jax.lax.cond(
        jnp.all(amounts == 0), lambda values: jnp.zeros_like(T + P + values.sum()),
        present, amounts,
    )
