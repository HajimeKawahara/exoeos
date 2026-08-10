"""Top-level public import contract."""

from exoeos import EquationOfState, IdealGas, ThermodynamicState, __version__


def test_top_level_exports_construct_the_public_state() -> None:
    model = IdealGas([2.0e-3], [29.0])
    state = model.state(300.0, 1.0e5, [1.0])

    assert isinstance(state, ThermodynamicState)
    assert EquationOfState is not None
    assert isinstance(__version__, str)
