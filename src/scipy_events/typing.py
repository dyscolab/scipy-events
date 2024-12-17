from typing import Any, Callable, Protocol, runtime_checkable

from numpy.typing import NDArray
from scipy.integrate import OdeSolution


class OdeSolver(Protocol):
    fun: Callable[[float, NDArray], NDArray]
    "RHS function."
    t_bound: float
    "Boundary time."
    direction: float
    "Integration direction: +1 or -1."
    t: float
    "Current time."
    y: NDArray
    "Current state."
    f: NDArray
    "Current evaluation of RHS."
    t_old: float | None
    "Previous time. None if no steps were made yet."
    step_size: float | None
    "Size of the last successful step. None if no steps were made yet."

    atol: float
    "Absolute tolerance."
    rtol: float
    "Relative tolerance."


class OdeResult(Protocol):
    t: NDArray
    "Time points."
    y: NDArray
    "Values of the solution at t."
    sol: OdeSolution | None
    "Found solution as OdeSolution instance; None if dense_output was set to False."
    t_events: list[NDArray] | None
    "Contains for each event type a list of arrays at which an event of that type event was detected. None if events was None."
    y_events: list[NDArray] | None
    "For each value of t_events, the corresponding value of the solution. None if events was None."
    nfev: int
    "Number of evaluations of the right-hand side."
    njev: int
    "Number of evaluations of the Jacobian."
    nlu: int
    "Number of LU decompositions."
    status: int
    """Reason for algorithm termination:
    - -1: Integration step failed.
    - 0: The solver successfully reached the end of tspan.
    - 1: A termination event occurred."""
    message: str
    "Human-readable description of the termination reason."
    success: bool
    "True if the solver reached the interval end or a termination event occurred (status >= 0)."


@runtime_checkable
class Condition(Protocol):
    """Events occur at the zeros of continuous function of time and state.

    Solvers will find an accurate value of t at which condition(t, y(t)) = 0 using a root-finding algorithm.
    """

    def __call__(self, t: float, y: NDArray, /) -> float: ...


@runtime_checkable
class Change(Protocol):
    """Change the solver state y and parameters args from the current (t, y)."""

    def __call__(self, t: float, y: NDArray, /, args) -> tuple[NDArray, tuple[Any]]: ...
