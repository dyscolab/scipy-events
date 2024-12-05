from .. import SmallDerivatives, solve_ivp


def test_small_derivatives():
    tmax = 10
    result = solve_ivp(
        lambda t, y: -(y - 0.5),
        t_span=(0, tmax),
        y0=[1.5],
        events=[SmallDerivatives(atol=1e-3, rtol=1e-3)],
    )
    assert result.t_events is not None
    assert result.t_events[0][0] == result.t[-1]


def test_small_derivatives_at_null_solution():
    tmax = 10
    result = solve_ivp(
        lambda t, y: -y,
        t_span=(0, tmax),
        y0=[1],
        events=[SmallDerivatives(atol=1e-3, rtol=1e-3)],
    )
    assert result.t_events is not None
    assert result.t_events[0][0] == result.t[-1]
