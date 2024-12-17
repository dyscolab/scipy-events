import numpy as np
from pytest import mark, param

from .. import ChangeWhen, SmallDerivatives, solve_ivp


@mark.parametrize(
    "t_eval",
    [
        param((), id="none"),
        param(None, id="all"),
        param([0.0, 0.5, 1.0], id="some"),
    ],
)
@mark.parametrize(
    "event",
    [
        param(lambda t, y: t > 0.5, id="condition"),
        param(SmallDerivatives(), id="with solver"),
        param(
            ChangeWhen(
                condition=lambda t, y: np.min(y) - 0.5,
                change=lambda t, y, args: (np.ones_like(y), args),
            ),
            id="change",
        ),
    ],
)
def test_t_eval(t_eval, event):
    t_span = (0.0, 1.0)
    result = solve_ivp(
        lambda t, y: -y,
        t_span=t_span,
        y0=[1.0],
        t_eval=t_eval,
        events=[event],
    )
    if t_eval is None:
        assert result.t[0] == t_span[0]
        assert result.t[-1] == t_span[1]
    else:
        assert result.t.tolist() == list(t_eval)
