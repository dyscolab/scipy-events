from pytest import mark

from .. import Event, SmallDerivatives, solve_ivp
from .._core import METHODS


@mark.parametrize("method", METHODS)
def test_small_derivatives_with_solver_tolerances(method):
    tmax = 10
    result = solve_ivp(
        lambda t, y: -(y - 0.5),
        t_span=(0, tmax),
        y0=[1.5],
        events=[Event(SmallDerivatives(), terminal=True)],
        method=method,
    )
    assert result.t_events is not None
    assert result.t_events[0][0] == result.t[-1]


@mark.parametrize("method", METHODS)
def test_small_derivatives(method):
    tmax = 10
    result = solve_ivp(
        lambda t, y: -(y - 0.5),
        t_span=(0, tmax),
        y0=[1.5],
        events=[Event(SmallDerivatives(atol=1e-3, rtol=1e-3), terminal=True)],
        method=method,
    )
    assert result.t_events is not None
    assert result.t_events[0][0] == result.t[-1]


@mark.parametrize("method", METHODS)
def test_small_derivatives_at_null_solution(method):
    tmax = 10
    result = solve_ivp(
        lambda t, y: -y,
        t_span=(0, tmax),
        y0=[1],
        events=[Event(SmallDerivatives(atol=1e-3, rtol=1e-3), terminal=True)],
        method=method,
    )
    assert result.t_events is not None
    assert result.t_events[0][0] == result.t[-1]
