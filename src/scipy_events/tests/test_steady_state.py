from pytest import mark

from .. import SmallDerivatives, solve_ivp
from ..core import METHODS


@mark.parametrize("method", METHODS)
def test_small_derivatives(method):
    tmax = 10
    result = solve_ivp(
        lambda t, y: -(y - 0.5),
        t_span=(0, tmax),
        y0=[1.5],
        events=[SmallDerivatives(atol=1e-3, rtol=1e-3)],
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
        events=[SmallDerivatives(atol=1e-3, rtol=1e-3)],
        method=method,
    )
    assert result.t_events is not None
    assert result.t_events[0][0] == result.t[-1]
