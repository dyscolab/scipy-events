from typing import Protocol

from numpy.typing import NDArray
from scipy.integrate import OdeSolution


class OdeSolver(Protocol):
    n: int
    "Number of equations."
    status: str
    "Current status of the solver: 'running', 'finished' or 'failed'."
    t_bound: float
    "Boundary time."
    direction: float
    "Integration direction: +1 or -1."
    t: float
    "Current time."
    y: NDArray
    "Current state."
    t_old: float
    "Previous time. None if no steps were made yet."
    step_size: float
    "Size of the last successful step. None if no steps were made yet."
    nfev: int
    "Number of the system's rhs evaluations."
    njev: int
    "Number of the Jacobian evaluations."
    nlu: int
    "Number of LU decompositions."

    atol: float
    "Absolute tolerance."
    rtol: float
    "Relative tolerance."
    f: float
    "Last evaluation of RHS."


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
