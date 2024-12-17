import numpy as np
from pytest import mark, param

from .. import ChangeAt, ChangeWhen, solve_ivp


def rhs(t, y):
    return -y


def solution(t, y0, t0):
    return np.atleast_1d(y0)[:, None] * np.exp(-(t - t0))


@mark.parametrize(
    "dense_output",
    [
        param(False, id="dense=False"),
        param(True, id="dense=True"),
    ],
)
def test_change(dense_output: bool):
    t0, y0 = 0.0, 1.0

    y_change, y1 = 0.5, 2.0
    t_end = 1.0

    result = solve_ivp(
        rhs,
        t_span=(t0, t_end),
        y0=[y0],
        dense_output=dense_output,
        events=[
            ChangeWhen(
                condition=lambda t, y: y[0] - y_change,
                change=lambda t, y, args: (np.array([y1]), args),
            )
        ],
        rtol=1e-6,  # to compare with solution
    )

    # Endpoints
    assert result.t[0] == t0
    assert result.t[-1] == t_end
    # Initial
    assert np.allclose(result.y[:, 0], y0)
    # At change
    assert result.y_events is not None
    assert np.allclose(result.y_events[0], y_change)
    # After change
    assert result.t_events is not None
    t_change = result.t_events[0][0]
    ix_post_change = np.searchsorted(result.t, t_change) + 1
    y_post_change = result.y[:, ix_post_change]
    assert np.allclose(y_post_change, y1)
    # Solution
    t_prev, t_post = np.split(result.t, (ix_post_change,))
    y_prev, y_post = np.split(result.y, (ix_post_change,), axis=1)
    assert np.allclose(y_prev, solution(t_prev, y0=y0, t0=t0))
    assert np.allclose(y_post, solution(t_post, y0=y1, t0=t_change))
    # Interpolation
    if dense_output:
        assert result.sol is not None
        assert np.allclose(result.sol(result.t), result.y)


@mark.parametrize(
    "dense_output",
    [
        param(False, id="dense=False"),
        param(True, id="dense=True"),
    ],
)
def test_change_at(dense_output: bool):
    t0, y0 = 0.0, 1.0

    t_change, y1 = 0.5, 2.0
    t_end = 1.0

    result = solve_ivp(
        rhs,
        t_span=(t0, t_end),
        y0=[y0],
        dense_output=dense_output,
        events=[
            ChangeAt(
                times=[t_change],
                change=lambda t, y, args: (np.array([y1]), args),
            )
        ],
        rtol=1e-6,  # to compare with solution
    )

    # Endpoints
    assert result.t[0] == t0
    assert result.t[-1] == t_end
    # Initial
    assert np.allclose(result.y[:, 0], y0)
    # At change
    assert result.t_events is not None
    assert np.allclose(result.t_events[0], t_change)
    # After change
    assert result.y_events is not None
    ix_post_change = np.searchsorted(result.t, t_change) + 1
    y_post_change = result.y[:, ix_post_change]
    assert np.allclose(y_post_change, y1)
    # Solution
    t_prev, t_post = np.split(result.t, (ix_post_change,))
    y_prev, y_post = np.split(result.y, (ix_post_change,), axis=1)
    assert np.allclose(y_prev, solution(t_prev, y0=y0, t0=t0))
    assert np.allclose(y_post, solution(t_post, y0=y1, t0=t_change))
    # Interpolation
    if dense_output:
        assert result.sol is not None
        assert np.allclose(result.sol(result.t), result.y)
