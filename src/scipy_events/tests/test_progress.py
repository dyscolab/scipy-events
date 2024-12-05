from .. import Progress, solve_ivp


def test_iterations():
    progress = Progress.from_tqdm()
    result = solve_ivp(
        lambda t, y: -y,
        t_span=(0, 1),
        y0=[1],
        events=[progress],
    )
    assert progress.pbar.n == result.t.size


def test_total_time():
    progress = Progress.from_tqdm(total_time=1)
    result = solve_ivp(
        lambda t, y: -y,
        t_span=(0, 1),
        y0=[1],
        events=[progress],
    )
    assert progress.pbar.n == result.t[-1]
