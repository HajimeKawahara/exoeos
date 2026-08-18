"""Top-level public import contract."""

from exoeos import (
    EquationOfState,
    GibbsExcessModel,
    HelmholtzEOS,
    IdealEOS,
    IdealGas,
    IdealSolution,
    PengRobinsonEOS,
    SecondVirialEOS,
    SolutionState,
    TPHelmholtzEOS,
    ThermodynamicState,
    TRhoState,
    ZhangDuanEOS,
    __version__,
    additive_volume_mass_density,
    mass_density_tp,
    psir,
    solution_state,
    state_tp,
    state_trho,
    total_gex_RT,
)


def test_top_level_exports_construct_the_public_state() -> None:
    residual_model = IdealEOS()
    virial_model = SecondVirialEOS([[0.0]])
    peng_robinson_model = PengRobinsonEOS(
        [190.564],
        [4_599_200.0],
        [0.01142],
    )
    zhang_duan_model = ZhangDuanEOS.from_species(("H2O",))
    solution_model = IdealSolution()
    model = IdealGas([2.0e-3], [29.0])
    state = model.state(300.0, 1.0e5, [1.0])
    trho_state = state_trho(residual_model, 300.0, 40.0, [1.0])
    tp_state = state_tp(virial_model, 300.0, 1.0e5, [1.0])
    peng_robinson_state = state_tp(
        peng_robinson_model,
        300.0,
        1.0e5,
        [1.0],
    )
    zhang_duan_state = state_tp(
        zhang_duan_model,
        1203.15,
        950.0e6,
        [1.0],
    )
    solution = solution_state(solution_model, 300.0, 1.0e5, [1.0])
    mass_density = mass_density_tp(
        residual_model,
        300.0,
        1.0e5,
        [1.0],
        [18.0e-3],
    )
    mixture_density = additive_volume_mass_density([1.0], [1000.0])

    assert isinstance(state, ThermodynamicState)
    assert isinstance(trho_state, TRhoState)
    assert isinstance(tp_state, TRhoState)
    assert isinstance(peng_robinson_state, TRhoState)
    assert isinstance(zhang_duan_state, TRhoState)
    assert isinstance(solution, SolutionState)
    assert mass_density > 0.0
    assert mixture_density == 1000.0
    assert EquationOfState is not None
    assert GibbsExcessModel is not None
    assert HelmholtzEOS is not None
    assert TPHelmholtzEOS is not None
    assert residual_model.alphar(300.0, 40.0, [1.0]) == 0.0
    assert psir(residual_model, 300.0, [40.0]) == 0.0
    assert solution_model.gex_RT(300.0, 1.0e5, [1.0]) == 0.0
    assert total_gex_RT(solution_model, 300.0, 1.0e5, [1.0]) == 0.0
    assert isinstance(__version__, str)
