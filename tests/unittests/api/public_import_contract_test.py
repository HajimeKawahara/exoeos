"""Top-level public import contract."""

from exoeos import (
    EquationOfState,
    GibbsExcessModel,
    HelmholtzEOS,
    IdealEOS,
    IdealGas,
    IdealSolution,
    SecondVirialEOS,
    SolutionState,
    TPHelmholtzEOS,
    ThermodynamicState,
    TRhoState,
    __version__,
    psir,
    solution_state,
    state_tp,
    state_trho,
    total_gex_RT,
)


def test_top_level_exports_construct_the_public_state() -> None:
    residual_model = IdealEOS()
    virial_model = SecondVirialEOS([[0.0]])
    solution_model = IdealSolution()
    model = IdealGas([2.0e-3], [29.0])
    state = model.state(300.0, 1.0e5, [1.0])
    trho_state = state_trho(residual_model, 300.0, 40.0, [1.0])
    tp_state = state_tp(virial_model, 300.0, 1.0e5, [1.0])
    solution = solution_state(solution_model, 300.0, 1.0e5, [1.0])

    assert isinstance(state, ThermodynamicState)
    assert isinstance(trho_state, TRhoState)
    assert isinstance(tp_state, TRhoState)
    assert isinstance(solution, SolutionState)
    assert EquationOfState is not None
    assert GibbsExcessModel is not None
    assert HelmholtzEOS is not None
    assert TPHelmholtzEOS is not None
    assert residual_model.alphar(300.0, 40.0, [1.0]) == 0.0
    assert psir(residual_model, 300.0, [40.0]) == 0.0
    assert solution_model.gex_RT(300.0, 1.0e5, [1.0]) == 0.0
    assert total_gex_RT(solution_model, 300.0, 1.0e5, [1.0]) == 0.0
    assert isinstance(__version__, str)
