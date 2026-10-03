"""One full-catalog residual potential for the conditional M2 gas scenarios.

Only the H2/He/H2O block has assigned second virials. Every other pair is
explicitly zero; this is a model assumption, not a bound on missing physics.
The archived three-component diagnostic and source transcription are retained
unchanged alongside this module. No three-component renormalization is used.
"""

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exoeos import SecondVirialEOS, state_tp
from exoeos.constants import MOLAR_GAS_CONSTANT


_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "_m2_archived_gas_virial", _HERE / "gas_virial.py")
_REFERENCE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REFERENCE)
MAJOR_SPECIES = ("H2", "He1", "H2O1")
_OPTION_NAMES = {"h2_he_cm3_mol", "water_cross_temperature_policy", "trace_pair_policy"}
_ORIGINAL_SOURCE_HASHES = {
    "gas_virial.py": "81c7a5b6f182f6c49d3f4fd8dfedbec53427add52071196b0fd650fe96d35388",
    "gas_virial_sources.json": "aa88f50111621db9ab1906b21070e068b4884d96fb263e8102de48e3699cce51",
}


def _scalar(value, name, lower, upper=math.inf):
    result = jnp.asarray(value)
    if result.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    if not isinstance(result, jax.core.Tracer):
        number = float(result)
        if not math.isfinite(number) or not lower <= number <= upper:
            raise ValueError(f"{name} must be finite and in [{lower}, {upper}].")
    return result


def _positive_pressure(value):
    result = _scalar(value, "P_Pa", 0.)
    if not isinstance(result, jax.core.Tracer) and float(result) <= 0:
        raise ValueError("P_Pa must be positive.")
    return result


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class MajorGasSecondVirial:
    """Conditional major-pair EOS on the complete declared species catalog.

    Numeric inputs under JAX tracing retain the existing ExoEOS caller
    contracts: normalized nonnegative x, positive total amount, and an admitted
    T/P state. Call ``parameters`` with concrete T/P before a traced solve.
    The table interpolation is C1; ``hold_2000`` has a temperature derivative
    corner at 2000 K and is not a globally smooth thermal model.
    """

    species: tuple
    h2_he_cm3_mol: float
    water_cross_temperature_policy: str
    trace_pair_policy: str

    @property
    def options(self):
        return {name: getattr(self, name) for name in sorted(_OPTION_NAMES)}

    @property
    def metadata(self):
        """Return state-independent identity for source/column binding."""
        return {
            "schema": "m2_full_catalog_major_gas_second_virial_v1",
            "species": list(self.species), "major_species": list(MAJOR_SPECIES),
            "options": self.options,
            "basis": "Full declared gas mole total; no major-species renormalization",
        }

    def coefficients(self, T_K):
        """Return symmetric B_ij in m3/mol in the full species order."""
        temperature = _scalar(T_K, "T_K", 1000., 3000.)
        dtype = jnp.result_type(temperature, jnp.float64)
        temperature = temperature.astype(dtype)
        cross_temperature = (jnp.minimum(temperature, 2000.)
            if self.water_cross_temperature_policy == "hold_2000" else temperature)
        values = {}
        for name, (coefficients, powers) in _REFERENCE._POWERS.items():
            t = cross_temperature if name in ("H2-H2O", "He-H2O") else temperature
            values[name] = jnp.sum(jnp.asarray(coefficients, dtype=dtype)
                                  * (t / 100.) ** jnp.asarray(powers, dtype=dtype))
        table = jnp.asarray(_REFERENCE._HELIUM, dtype=dtype)
        index = jnp.clip(jnp.searchsorted(table[:, 0], temperature, side="right") - 1,
                         0, len(_REFERENCE._HELIUM) - 2)
        low, high = table[index], table[index + 1]
        width = high[0] - low[0]
        s = (temperature - low[0]) / width
        helium = ((2*s**3 - 3*s**2 + 1)*low[1]
                  + (s**3 - 2*s**2 + s)*width*low[2]/low[0]
                  + (-2*s**3 + 3*s**2)*high[1]
                  + (s**3 - s**2)*width*high[2]/high[0])
        major = 1e-6 * jnp.asarray([
            [values["H2-H2"], self.h2_he_cm3_mol, values["H2-H2O"]],
            [self.h2_he_cm3_mol, helium, values["He-H2O"]],
            [values["H2-H2O"], values["He-H2O"], values["H2O-H2O"]],
        ], dtype=dtype)
        indices = jnp.asarray([self.species.index(name) for name in MAJOR_SPECIES])
        return jnp.zeros((len(self.species), len(self.species)), dtype=dtype).at[
            indices[:, None], indices[None, :]].set(major)

    def state(self, T_K, P_Pa, x):
        """Return density, ln(phi), and residual G/(nRT) from one potential."""
        pressure = _positive_pressure(P_Pa)
        fractions = jnp.asarray(x)
        if fractions.shape != (len(self.species),):
            raise ValueError("x must follow the complete declared species order.")
        if not isinstance(fractions, jax.core.Tracer):
            numeric = np.asarray(fractions)
            if (not np.all(np.isfinite(numeric)) or np.any(numeric < 0)
                    or not np.isclose(numeric.sum(), 1., rtol=0., atol=1e-12)):
                raise ValueError("x must be nonnegative and sum to one on the full catalog.")
        state = state_tp(SecondVirialEOS(self.coefficients(T_K)), T_K, pressure, fractions)
        if not isinstance(state.rho, jax.core.Tracer):
            b = float(fractions @ self.coefficients(T_K) @ fractions)
            if (not math.isfinite(float(state.rho))
                    or 1. + 2. * float(state.rho) * b <= 0.):
                raise ValueError("The gas has no strictly stable low-density root.")
        return state

    def gibbs_residual_rt(self, T_K, P_Pa, n):
        """Return extensive G^r/(RT); its amount gradient is full-catalog lnphi."""
        amounts = jnp.asarray(n)
        if amounts.shape != (len(self.species),):
            raise ValueError("n must follow the complete declared species order.")
        total = jnp.sum(amounts)
        if not isinstance(amounts, jax.core.Tracer):
            numeric = np.asarray(amounts)
            if (not np.all(np.isfinite(numeric)) or np.any(numeric < 0)
                    or float(total) <= 0.):
                raise ValueError("n must be finite, nonnegative, and have positive total.")
        return total * self.state(T_K, P_Pa, amounts / total).gres_RT

    def mass_density(self, T_K, P_Pa, x, molar_masses_kg_mol):
        """Return kg/m3 using molar masses in the same full-catalog order."""
        masses = jnp.asarray(molar_masses_kg_mol)
        if masses.shape != (len(self.species),):
            raise ValueError("Molar masses must follow the complete species order.")
        if not isinstance(masses, jax.core.Tracer):
            numeric = np.asarray(masses)
            if not np.all(np.isfinite(numeric)) or np.any(numeric <= 0):
                raise ValueError("Molar masses must be finite and positive in kg/mol.")
        return self.state(T_K, P_Pa, x).rho * jnp.dot(jnp.asarray(x), masses)

    def parameters(self, T_K, P_Pa):
        """Return frozen scalar coefficients and a non-certified domain diagnostic.

        An independent verifier must enclose the expressions outwards before
        using the entropy curvature as a global proof. Pair coefficients are
        the reported binary64 constants at this T; the receipt does not claim
        a measured error bound or validity of higher virials.
        """
        temperature = float(_scalar(T_K, "T_K", 1000., 3000.))
        pressure = float(_positive_pressure(P_Pa))
        matrix = np.asarray(self.coefficients(temperature))
        minimum, maximum_abs = float(np.min(matrix)), float(np.max(np.abs(matrix)))
        ideal = pressure / (MOLAR_GAS_CONSTANT * temperature)
        discriminant = 1. + 4. * ideal * minimum
        if discriminant <= 0.:
            raise ValueError("The full simplex lacks a strictly stable density bound.")
        dmin = math.sqrt(discriminant)
        density_upper = 2. * ideal / (1. + dmin)
        epsilon = (2. * density_upper * maximum_abs
                   + 4. * density_upper**2 * maximum_abs**2 / dmin)
        if epsilon >= 1.:
            raise ValueError("The full-simplex entropy curvature bound is not positive.")
        files = [Path(__file__), _HERE / "gas_virial.py", _HERE / "gas_virial_sources.json"]
        files += [Path(importlib.import_module("exoeos." + name).__file__)
                  for name in ("helmholtz", "second_virial", "state", "constants", "_arrays")]
        return {
            **self.metadata, "temperature_K": temperature, "pressure_Pa": pressure,
            "gas_constant_J_mol_K": MOLAR_GAS_CONSTANT,
            "coefficients_m3_mol": matrix.tolist(),
            "potential_expression": {
                "basis": "All declared gas species; x=n/sum(n), without conditioning on major species.",
                "residual_helmholtz_over_nRT": "rho * (x.T @ B @ x)",
                "density": "2*I/(1+sqrt(1+4*I*Bmix)); I=P/(R*T)",
                "residual_gibbs_over_nRT": "2*rho*Bmix-log1p(rho*Bmix)",
                "log_fugacity_coefficients": "2*rho*(B @ x)-log1p(rho*Bmix)",
                "extensive_residual_gibbs_over_RT": "sum(n)*residual_gibbs_over_nRT",
            },
            "convexity_diagnostic": {
                "certified": False, "arithmetic": "binary64; independently enclose before proof use",
                "scope": "Entire nonnegative full-catalog simplex at this fixed T/P",
                "minimum_pair_coefficient_m3_mol": minimum,
                "maximum_absolute_pair_coefficient_m3_mol": maximum_abs,
                "density_upper_mol_m3": density_upper,
                "mechanical_denominator_lower": dmin,
                "helmholtz_dimensionless_lower": 1. - 2.*density_upper*maximum_abs,
                "entropy_curvature_lower": 1. - epsilon,
                "bound_expression": "alpha=1-2*rhomax*b-4*rhomax^2*b^2/dmin",
                "bound_definitions": "b=max(abs(B)); dmin=sqrt(1+4*I*min(B)); rhomax=2*I/(1+dmin)",
                "tangent_hessian": "diag(1/x)+2*rho*B-4*rho^2/(1+2*rho*Bmix)*(B@x)*(B@x).T",
            },
            "temperature_assumptions": {
                "implemented_temperature_range_K": [1000., 3000.],
                "water_cross_recommended_maximum_K": 2000.,
                "water_cross_high_temperature_assumption_active": temperature > 2000.,
                "h2_he_temperature_policy": "Explicit constant sensitivity scenario; not empirical bounds",
                "hold_2000_temperature_derivative_corner_K": (
                    2000. if self.water_cross_temperature_policy == "hold_2000" else None),
            },
            "original_source_checkout": {
                "repository": "https://github.com/HajimeKawahara/exoeos.git",
                "commit": "3d36c978c4f60545e2961d6af1d30c549e635b9b",
                "file_sha256": _ORIGINAL_SOURCE_HASHES,
            },
            "file_sha256": {str(path): _sha256(path) for path in files},
            "provider_recipe_file_sha256": {
                str(path.relative_to(_HERE.parents[1])): _sha256(path) for path in files},
            "limitations": [
                "All pairs outside the major three block are explicitly zero, not empirically bounded.",
                "Zero-pair trace species still have lnphi=-ln(Z) on the full gas basis.",
                "The H2-He scenarios and high-temperature water-cross policies are conditional alternatives.",
                "Higher virials and missing pairs have no complete physical error bound.",
                "A local EOS receipt does not establish coupled mass, partition, or phase acceptance.",
            ],
        }


def make_major_gas_eos(gas_species, gas_eos_options):
    """Build the explicit conditional model; a consumer handles None as ideal."""
    if isinstance(gas_species, (str, bytes)):
        raise ValueError("gas_species must be an ordered catalog, not a string.")
    species = tuple(gas_species)
    if (not species or any(not isinstance(name, str) or not name for name in species)
            or len(set(species)) != len(species)
            or not set(MAJOR_SPECIES).issubset(species)):
        raise ValueError("Require unique full-catalog species including H2, He1, and H2O1.")
    if not isinstance(gas_eos_options, dict) or set(gas_eos_options) != _OPTION_NAMES:
        raise ValueError(f"gas_eos_options requires exactly {sorted(_OPTION_NAMES)}.")
    cross = gas_eos_options["h2_he_cm3_mol"]
    if isinstance(cross, (bool, np.bool_)) or not isinstance(cross, (int, float)) or not math.isfinite(cross):
        raise ValueError("h2_he_cm3_mol must be an explicit finite number, not a boolean.")
    policy = gas_eos_options["water_cross_temperature_policy"]
    if policy not in ("extrapolate", "hold_2000"):
        raise ValueError("water_cross_temperature_policy must be extrapolate or hold_2000.")
    if gas_eos_options["trace_pair_policy"] != "zero":
        raise ValueError("trace_pair_policy must explicitly be zero.")
    return MajorGasSecondVirial(species, float(cross), policy, "zero")
