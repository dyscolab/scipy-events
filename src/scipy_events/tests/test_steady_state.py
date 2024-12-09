import numpy as np
from pytest import mark

from .. import Event, SmallDerivatives, solve_ivp
from .._core import METHODS


@mark.parametrize("method", METHODS)
@mark.parametrize("tol", [None, 1e-3])
@mark.parametrize("y0", [0.0, 1.0])
def test_small_derivatives(method, tol: float | None, y0: float):
    result = solve_ivp(
        lambda t, y: -(y - y0),
        t_span=(0, np.inf),
        y0=[y0 + 1],
        events=[Event(condition=SmallDerivatives(atol=tol, rtol=tol), terminal=True)],
        method=method,
    )
    assert result.t_events is not None
    assert result.t_events[0][0] == result.t[-1]
