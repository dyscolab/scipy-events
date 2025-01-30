import numpy as np

from scipy_events import ChangeAt, solve_ivp


def test_issue_1():
    """Argument t_eval was not sliced after ChangeAt events.

    https://github.com/maurosilber/scipy-events/issues/1

    Raised:
        ValueError: Values in t_eval are not within t_span
    """
    solve_ivp(
        lambda t, y: -y,
        t_eval=[1],
        t_span=(0, 16),
        y0=[1],
        events=[
            ChangeAt(
                times=[4, 8],
                change=lambda t, y, args: (np.full_like(y, 1), args),
            )
        ],
    )
