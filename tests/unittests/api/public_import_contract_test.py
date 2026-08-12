"""Top-level public import contract."""

from exoeos import (
    EquationOfState,
    HelmholtzEOS,
    IdealEOS,
    IdealGas,
    ThermodynamicState,
    TRhoState,
    __version__,
    psir,
    state_trho,
)


def test_top_level_exports_construct_the_public_state() -> None:
    residual_model = IdealEOS()
    model = IdealGas([2.0e-3], [29.0])
    state = model.state(300.0, 1.0e5, [1.0])
    trho_state = state_trho(residual_model, 300.0, 40.0, [1.0])

    assert isinstance(state, ThermodynamicState)
    assert isinstance(trho_state, TRhoState)
    assert EquationOfState is not None
    assert HelmholtzEOS is not None
    assert residual_model.alphar(300.0, 40.0, [1.0]) == 0.0
    assert psir(residual_model, 300.0, [40.0]) == 0.0
    assert isinstance(__version__, str)
