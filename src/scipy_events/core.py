from typing import Any, Callable, Literal, Protocol, Sequence

from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp as _solve_ivp
from scipy.integrate._ivp.ivp import METHODS
from scipy.integrate._ivp.ivp import OdeSolver as _OdeSolver

from .typing import OdeResult, OdeSolver


class _OdeWrapper(type):
    """Allows access to the solver instance created inside scipy.integrate.solve_ivp.

    solve_ivp's method parameter requires a type[OdeSolver], which is instanced inside solve_ivp.
    Instances of this class save a reference to the instanced solver before returning it.
    """

    solver_cls: type[OdeSolver]
    solver: OdeSolver

    def __new__(cls, solver_cls: type[OdeSolver], /):
        return super().__new__(cls, "OdeWrapperInstance", (_OdeSolver,), {})

    def __init__(self, solver_cls: type[OdeSolver], /):
        self.solver_cls = solver_cls

    def __call__(self, *args, **kwargs):
        """Saves reference to the solver instance"""
        self.solver = self.solver_cls(*args, **kwargs)
        return self.solver


class Event(Protocol):
    terminal: bool | int = False
    "Whether to terminate integration if this event occurs, or after the specified number of times."
    direction: float = 0.0
    """Direction of a zero crossing.
    If direction is positive, event will only trigger when going from negative to positive,
    and vice versa if direction is negative. If 0, then either direction will trigger event."""

    def __call__(self, t: float, y: NDArray) -> float: ...


class EventWithSolver(Event):
    """An event with access to the solver instance."""

    _ode_wrapper: _OdeWrapper

    @property
    def solver(self) -> OdeSolver:
        return self._ode_wrapper.solver


def solve_ivp(
    fun: Callable[[float, NDArray], NDArray],
    /,
    t_span: tuple[float, float],
    y0: ArrayLike,
    *,
    method: type[OdeSolver]
    | Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = "RK45",
    t_eval: ArrayLike | None = None,
    dense_output: bool = False,
    events: Sequence[Event] = (),
    vectorized: bool = False,
    args: tuple[Any] | None = None,
    **options,
) -> OdeResult:
    """Solve an initial value problem for a system of ODEs.

    All parameters are passed unmodified to scipy.integrate.solve_ivp.
    Read its documentation.

    It is only necessary to call this function if you need to use EventWithSolver events.
    Otherwise, scipy.integrate.solve_ivp can be used.
    """
    if isinstance(method, str):
        method = METHODS[method]
    ode_wrapper = _OdeWrapper(method)  # type: ignore

    wrapped_events = []
    for e in events:
        if isinstance(e, EventWithSolver):
            e._ode_wrapper = ode_wrapper
        wrapped_events.append(e)

    result = _solve_ivp(
        fun,
        t_span=t_span,
        y0=y0,
        method=ode_wrapper,  # type: ignore
        t_eval=t_eval,
        dense_output=dense_output,
        events=wrapped_events,
        vectorized=vectorized,
        args=args,
        **options,
    )

    for e in wrapped_events:
        if isinstance(e, EventWithSolver):
            del e._ode_wrapper

    return result
