"""Check the global inequalities and their original coordinate normalization."""

from dataclasses import asdict
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exoeos import associated_metal, phosphorus_mqm as binary
from exoeos import phosphorus_mqm_alloy as alloy
from exoeos._conditional_data import R
from exoeos.ma_fe_si_o import MaFeSiOLiquid
from exoeos.phosphorus_mqm_interval import mqm_alloy_insertion_bound


T, P = 2173.15, 6e6
LOWER = np.zeros(20)
LOWER[0] = .75
UPPER = np.array([1., .02, .01, .04, 0., .002, .00002, .0001, .12, .00001,
                  .003, .00002, .0001, .005, .00001, .000001, .002, .000001, .02, .02])


def inputs():
    epsilon, matrix = associated_metal.interaction_parameters(T)
    return dict(temperature_k=T, pressure_pa=P,
        pure_standards_rt=np.full(20, 40.), supporting_potentials_rt=np.zeros(20),
        phosphorus_epsilon=epsilon, additional_matrix=matrix,
        host_lower=LOWER, host_upper=UPPER, model='you2020', delta_v_p_m3_mol=0.)


def scalar(args, point):
    collapsed = alloy.collapse_matrix(model=args['model']) @ point
    return float(alloy.gibbs_j(args['temperature_k'], args['pressure_pa'], point,
        args['pure_standards_rt'], args['phosphorus_epsilon'], args['additional_matrix'],
        model=args['model'], delta_v_p_m3_mol=args['delta_v_p_m3_mol'],
        host_model=args.get('host_model'))/(R*args['temperature_k'])
        -collapsed @ args['supporting_potentials_rt'])


def chemical_to_pairs(host, p_fraction, exchange):
    chemical = (1-p_fraction)*host
    chemical[4] = p_fraction
    pair = np.r_[chemical, 0.]
    unlike = exchange*min(3*chemical[0], 6*chemical[4])
    pair[0], pair[4], pair[20] = 3*chemical[0]-unlike, 3*chemical[4]-.5*unlike, unlike
    return pair/pair.sum()


def test_pair_entropy_correction_is_mutual_information():
    # This identity supplies a global nonnegative term; evaluating a sample
    # only checks that the algebra matches the provider implementation.
    for pair in ([1., 0., 0.], [0., 1., 0.], [.2, .3, .5], [.01, .09, .9]):
        pair = np.asarray(pair)
        marginal = np.array([pair[0]+pair[2]/2, pair[1]+pair[2]/2])
        joint = np.array([[pair[0], pair[2]/2], [pair[2]/2, pair[1]]])
        reference = marginal[:,None]*marginal[None,:]
        mask = joint > 0
        information = np.sum(joint[mask]*np.log(joint[mask]/reference[mask]))
        provider = float(binary._ideal(jnp.asarray(pair))-2*binary._ideal(jnp.asarray(marginal))
                         -pair[2]*jnp.log(2.))
        assert information >= -1e-15
        np.testing.assert_allclose(provider, information, rtol=0, atol=1e-15)


def test_positive_bound_covers_pure_endpoints_and_feasible_mixtures():
    args = inputs()
    bound = mqm_alloy_insertion_bound(**args)
    assert bound.lower_rt > 0
    rng = np.random.default_rng(20261120)
    for p in (0., .01, .5, .99, 1.):
        for exchange in (0., .4, 1.):
            for _ in range(5):
                host = UPPER*rng.uniform(0, .5, 20)
                host[0] = 0.
                host[0] = 1-host.sum()
                point = chemical_to_pairs(host, p, exchange)
                assert scalar(args, point) >= bound.lower_rt
    for name in ('FeFe', 'PP'):
        point = np.zeros(21)
        point[alloy.COMPONENTS.index(name)] = 1.
        assert scalar(args, point) >= bound.lower_rt


def test_negative_bound_retains_correct_normalization_extreme():
    args = inputs()
    args['pure_standards_rt'] = np.full(20, -40.)
    bound = mqm_alloy_insertion_bound(**args)
    assert bound.lower_rt < bound.chemical_lower_rt < 0.
    assert bound.chemical_moles_per_component[1] >= 1.
    point = chemical_to_pairs(np.eye(20)[0], .2, .5)
    assert scalar(args, point) >= bound.lower_rt


@pytest.mark.parametrize('model', binary.MODELS)
def test_named_pair_models_cover_the_full_binary_simplex(model):
    args = inputs()
    args['model'] = model
    result = mqm_alloy_insertion_bound(**args)
    rng = np.random.default_rng(3)
    for pairs in rng.dirichlet(np.ones(3)*.2,size=64):
        point = np.zeros(21)
        point[[0,4,20]] = pairs
        assert scalar(args,point) >= result.lower_rt


@pytest.mark.parametrize('coefficients', [(23243.93,-16500.,-9365.), (0.,0.,0.)])
def test_explicit_ma_host_is_used_without_changing_its_coefficients(coefficients):
    args = inputs()
    args['host_model'] = MaFeSiOLiquid(coefficients)
    result = mqm_alloy_insertion_bound(**args)
    host = np.eye(20)[0]*.97+np.eye(20)[1]*.02+np.eye(20)[2]*.01
    for p in (0.,.5,1.):
        assert scalar(args,chemical_to_pairs(host,p,.5)) >= result.lower_rt


def test_pressure_volume_and_supporting_plane_shift_are_included():
    args = inputs()
    base = mqm_alloy_insertion_bound(**args)
    args['delta_v_p_m3_mol'] = -1e-3
    changed = mqm_alloy_insertion_bound(**args)
    assert changed.phosphorus_linear_lower_rt < base.phosphorus_linear_lower_rt
    args['supporting_potentials_rt'] = np.full(20, 2.)
    shifted = mqm_alloy_insertion_bound(**args)
    assert shifted.lower_rt < changed.lower_rt
    for p in (0., .5, 1.):
        assert scalar(args, chemical_to_pairs(np.eye(20)[0], p, .5)) >= shifted.lower_rt


def test_entropy_rounding_and_pure_host_zero_interaction_control():
    args = inputs()
    args['host_lower'] = args['host_upper'] = np.eye(20)[0]
    args['additional_matrix'] = np.zeros((18, 18))
    result = mqm_alloy_insertion_bound(**args)
    with localcontext() as context:
        context.prec = 70
        assert Decimal.from_float(result.entropy_lower_rt) <= -Decimal(20).ln()
    assert all(np.isfinite(value) for key, value in asdict(result).items()
               if key != 'chemical_moles_per_component')
    assert scalar(args, np.eye(21)[0]) >= result.lower_rt


@pytest.mark.parametrize('field,value,message', [
    ('temperature_k', 0., 'positive'),
    ('pressure_pa', np.inf, 'finite'),
    ('pure_standards_rt', np.zeros(21), 'shape'),
    ('supporting_potentials_rt', np.full(20, np.nan), 'finite'),
    ('host_lower', np.zeros(20), 'positive Fe'),
    ('host_upper', np.ones(20), 'P-free'),
    ('phosphorus_epsilon', [0., -1., np.inf, 0.], 'finite'),
    ('additional_matrix', np.ones((20,20)), 'shape'),
])
def test_unsupported_inputs_are_rejected(field, value, message):
    args = inputs()
    args[field] = value
    with pytest.raises(ValueError, match=message):
        mqm_alloy_insertion_bound(**args)


def test_singular_host_box_is_rejected():
    args = inputs()
    upper = UPPER.copy()
    upper[1] = .8
    args['host_upper'] = upper
    with pytest.raises(ValueError, match='strictly below one'):
        mqm_alloy_insertion_bound(**args)


def test_negative_phosphorus_cross_and_self_terms_are_retained():
    args = inputs()
    original = mqm_alloy_insertion_bound(**args)
    args['phosphorus_epsilon'] = np.array([0., -100., -200., -300.])
    matrix = np.array(args['additional_matrix'])
    matrix[4,4], matrix[4,8], matrix[8,4] = -50., -60., -60.
    args['additional_matrix'] = matrix
    changed = mqm_alloy_insertion_bound(**args)
    assert changed.lower_rt < original.lower_rt
    host = np.eye(20)[0]*.8+np.eye(20)[8]*.1+np.eye(20)[3]*.02+np.eye(20)[18]*.08
    host[0] += .08
    host[18] -= .08
    for p in (0., .01, .5, .99, 1.):
        for exchange in (0., .5, 1.):
            assert scalar(args, chemical_to_pairs(host, p, exchange)) >= changed.lower_rt


def test_reduced_precision_coefficients_are_rejected():
    args = inputs()
    original_x64 = jax.config.x64_enabled
    try:
        jax.config.update('jax_enable_x64', False)
        with pytest.raises(ValueError, match='float64'):
            mqm_alloy_insertion_bound(**args)
    finally:
        jax.config.update('jax_enable_x64', original_x64)


@pytest.mark.parametrize('changes', [
    dict(temperature_k=1e308),
    dict(pressure_pa=1e308,delta_v_p_m3_mol=1e308),
    dict(pure_standards_rt=np.full(20,1e308),supporting_potentials_rt=np.full(20,-1e308)),
    dict(additional_matrix=np.full((18,18),-1e308)),
])
def test_finite_but_overflowing_inputs_cannot_produce_a_certificate(changes):
    args = inputs()
    args.update(changes)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        with pytest.raises(ValueError,match='finite|overflow'):
            mqm_alloy_insertion_bound(**args)
