from pytest import mark, param

from .. import ChangeAt, ChangeWhen, Event, SmallDerivatives, solve_ivp


@mark.parametrize(
    "event",
    [
        param(lambda t, y, *args: t - 0.5, id="Condition"),
        param(Event(condition=lambda t, y, *args: t - 0.5), id="Event"),
        param(
            ChangeAt(times=(0.5,), change=lambda t, y, args: (y + 1, args)),
            id="ChangeAt",
        ),
        param(
            ChangeWhen(
                condition=lambda t, y, *args: t - 0.5,
                change=lambda t, y, args: (y + 1, args),
            ),
            id="ChangeWhen",
        ),
        param(
            Event(condition=SmallDerivatives(), terminal=True), id="SmallDerivatives"
        ),
    ],
)
@mark.parametrize(
    "f_and_args",
    [
        param((lambda t, y: -y, ()), id="p=()"),
        param((lambda t, y, p: -p * y, (1,)), id="p=(1,)"),
    ],
)
def test_params(f_and_args, event):
    f, args = f_and_args
    solve_ivp(f, (0, 1), [1], args=args, events=[event])
