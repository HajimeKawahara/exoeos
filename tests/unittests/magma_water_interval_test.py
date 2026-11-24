"""Check the global proof's domain, failure modes and analytical control."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from exoeos.magma_water_interval import reduced_two_power_melt_bound


def exact(values):
    values = np.asarray(values, dtype=float)
    return np.stack((values, values), axis=-1)


def control():
    # Constant water terms make the true simplex minimum analytical.
    return dict(
        linear_rt=exact(np.full(11, np.log(11.))),
        quadratic_rt=exact(np.zeros((11, 11))),
        entropy_coefficient=exact(1.),
        water_equivalent=exact(np.ones(11)),
        water_log_terms=exact([-30., -30.]),
        water_log_predictors=exact(np.zeros((2, 11))),
        predictor_counts=np.ones(11),
        water_exponents=np.full(2, .5),
        reference_composition=np.full(11, 1/11),
        local_margin=1.,
    )


def test_complete_simplex_contains_the_analytic_constant_capacity_minimum():
    inputs = control()
    result = reduced_two_power_melt_bound(**inputs)
    with localcontext() as context:
        context.prec = 80
        d = Decimal.from_float
        term = Decimal(-30).exp()
        integral = -(1-2*term).ln()/Decimal('.5')
        minimum = d(float(inputs['linear_rt'][0, 0]))-Decimal(11).ln()-integral
        lower = d(result['lower_rt_per_dry_mole'])
        assert lower <= minimum
        assert minimum-lower < Decimal('1e-10')
    assert result['complete']
    assert result['processed_boxes'] == 1
    assert result['local_boxes'] == 1
    assert result['unresolved_boxes'] == 0


def test_a_negative_global_minimum_is_not_clipped_or_declared_stable():
    inputs = control()
    inputs['linear_rt'] = exact(np.full(11, -1.))
    result = reduced_two_power_melt_bound(**inputs)
    assert result['complete']
    assert result['lower_rt_per_dry_mole'] < -3


def test_an_exhausted_box_budget_retains_an_unknown_global_bound():
    inputs = control()
    inputs.update(local_margin=.01, max_boxes=1)
    result = reduced_two_power_melt_bound(**inputs)
    assert not result['complete']
    assert result['lower_rt_per_dry_mole'] is None
    assert result['unresolved_boxes'] == 2


def test_uncertain_linear_and_entropy_coefficients_are_enclosed():
    inputs = control()
    inputs['linear_rt'][:, 0] -= .01
    inputs['linear_rt'][:, 1] += .01
    inputs['entropy_coefficient'] = [.99, 1.01]
    result = reduced_two_power_melt_bound(**inputs)
    rng = np.random.default_rng(20261124)
    for _ in range(100):
        x = rng.dirichlet(np.ones(11))
        c = rng.uniform(inputs['linear_rt'][:, 0], inputs['linear_rt'][:, 1])
        r = rng.uniform(.99, 1.01)
        value = c @ x + r * np.sum(x*np.log(x)) + 2*np.log1p(-2*np.exp(-30.))
        assert value >= result['lower_rt_per_dry_mole']


@pytest.mark.parametrize('interaction', [-.01, .01])
def test_exterior_mccormick_bound_contains_analytic_symmetric_minimum(interaction):
    inputs = control()
    inputs['linear_rt'] = exact(np.full(11, np.log(11.)+1))
    inputs['quadratic_rt'] = exact(interaction*(np.ones((11, 11))-np.eye(11)))
    inputs['local_margin'] = .01
    result = reduced_two_power_melt_bound(**inputs)
    with localcontext() as context:
        context.prec = 80
        d = Decimal.from_float
        integral = -(1-2*Decimal(-30).exp()).ln()/Decimal('.5')
        minimum = (d(float(inputs['linear_rt'][0, 0])) - Decimal(11).ln()
                   + d(interaction)*(1-1/Decimal(11))/2 - integral)
        assert d(result['lower_rt_per_dry_mole']) <= minimum
    assert result['complete']
    assert result['excluded_boxes'] >= 1
    assert result['lower_rt_per_dry_mole'] > .9


@pytest.mark.parametrize('change', ['asymmetric', 'diagonal', 'sign_uncertain'])
def test_unsupported_quadratic_assumptions_are_rejected(change):
    inputs = control()
    matrix = inputs['quadratic_rt']
    if change == 'asymmetric':
        matrix[0, 1] = [1., 1.]
    elif change == 'diagonal':
        matrix[0, 0] = [1., 1.]
    else:
        matrix[0, 1] = matrix[1, 0] = [-.1, .1]
    with pytest.raises(ValueError):
        reduced_two_power_melt_bound(**inputs)


@pytest.mark.parametrize('field,value', [
    ('curvature_floor', 100.), ('curvature_floor', 0.),
    ('water_log_terms', [[0., 0.], [0., 0.]]),
    ('predictor_counts', np.zeros(11)),
    ('reference_composition', np.zeros(11)),
    ('linear_rt', np.full((11, 2), np.nan)),
    ('max_boxes', True), ('max_boxes', 0),
])
def test_invalid_inputs_and_unproved_local_curvature_do_not_make_certificates(field, value):
    inputs = control()
    inputs[field] = value
    with pytest.raises(ValueError):
        reduced_two_power_melt_bound(**inputs)
