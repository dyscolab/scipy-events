import itertools
from dataclasses import KW_ONLY, dataclass
from typing import Any, Callable, Literal, Sequence, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp as _solve_ivp
from scipy.integrate._ivp.ivp import BDF, LSODA, METHODS, OdeSolution
from scipy.integrate._ivp.ivp import OdeSolver as _OdeSolver

from .change import Change
from .typing import Condition, OdeResult, OdeSolver


@dataclass
class _LSODAWrapper:
    solver: LSODA

    def __getattr__(self, name):
        return getattr(self.solver, name)

    @property
    def f(self) -> NDArray:
        return self.solver.fun(self.solver.t, self.solver.y)

    @property
    def atol(self):
        return self.solver._lsoda_solver._integrator.atol

    @property
    def rtol(self):
        return self.solver._lsoda_solver._integrator.rtol


@dataclass
class _BDFWrapper:
    solver: BDF

    def __getattr__(self, name):
        return getattr(self.solver, name)

    @property
    def f(self) -> NDArray:
        return self.solver.fun(self.solver.t, self.solver.y)


_wrappers = {
    LSODA: _LSODAWrapper,
    BDF: _BDFWrapper,
}


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
        solver = self.solver_cls(*args, **kwargs)
        try:
            self.solver = _wrappers[self.solver_cls](solver)  # type: ignore
        except KeyError:
            self.solver = solver
        return solver


@dataclass
class Event:
    condition: Condition
    _: KW_ONLY
    terminal: bool | int = False
    "Whether to terminate integration if this event occurs, or after the specified number of times."
    direction: float = 0.0
    """Direction of a zero crossing.
    If direction is positive, event will only trigger when going from negative to positive,
    and vice versa if direction is negative. If 0, then either direction will trigger event."""

    def __call__(self, t: float, y: NDArray, /) -> float:
        return self.condition(t, y)


class WithSolver:
    """Mixin to get access to the solver instance."""

    _ode_wrapper: _OdeWrapper

    @property
    def solver(self) -> OdeSolver:
        return self._ode_wrapper.solver  # type: ignore


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
    events: Sequence[Condition | Event | Change] = (),
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
    if t_eval is not None:
        t_eval = np.asarray(t_eval)

    if isinstance(method, str):
        method = METHODS[method]
    ode_wrapper = _OdeWrapper(method)  # type: ignore

    normalized_events: list[Event] = []
    change_events: list[tuple[int, Change]] = []
    for i, e in enumerate(events):
        if isinstance(e, Change):
            change_events.append((i, e))
            e = Event(
                e.event,
                terminal=True,
                direction=e.direction,
            )

        if not isinstance(e, Event):
            e = Event(
                e,
                terminal=getattr(e, "terminal", False),
                direction=getattr(e, "direction", 0.0),
            )

        if isinstance(e.condition, WithSolver):
            e.condition._ode_wrapper = ode_wrapper

        normalized_events.append(e)

    t0, t_end = t_span
    results = []
    while True:
        r: OdeResult = _solve_ivp(
            fun,
            t_span=(t0, t_end),
            y0=y0,
            method=ode_wrapper,  # type: ignore
            t_eval=t_eval,
            dense_output=dense_output,
            events=normalized_events,
            vectorized=vectorized,
            args=args,
            **options,
        )
        results.append(r)
        if r.status != 1:
            break

        assert r.t_events is not None
        assert r.y_events is not None
        for i, e in change_events:
            if r.t_events[i].size == 1:
                t0 = r.t_events[i][0]
                y0 = r.y_events[i][:, 0].copy()
                y0 = e.change(t0, y0)
                t0 = np.nextafter(t0, np.inf)
                if t_eval is not None:
                    t_eval = t_eval[np.searchsorted(t_eval, t0) :]
                break
        else:
            break  # not a change event

    for e in normalized_events:
        if isinstance(e, WithSolver):
            del e._ode_wrapper

    return _join_results(results)


def _join_results(results: list[OdeResult], /) -> OdeResult:
    # Modify last to retain its message and success attributes.
    last = results[-1]
    last.t = np.concatenate([r.t for r in results])
    last.y = np.concatenate([r.y for r in results], axis=-1)
    last.nfev = sum(r.nfev for r in results)
    last.njev = sum(r.njev for r in results)
    last.nlu = sum(r.nlu for r in results)

    if last.sol is not None:
        sols = cast(list[OdeSolution], [r.sol for r in results])
        ts = [sols[0].ts]
        ts.extend((s.ts[1:] for s in sols[1:]))
        last.sol = OdeSolution(
            ts=np.concatenate(ts),
            interpolants=list(
                itertools.chain.from_iterable(s.interpolants for s in sols)
            ),
        )

    if last.t_events is not None and last.y_events is not None:
        for i in range(len(last.t_events)):
            last.t_events[i] = np.concatenate(
                [
                    r.t_events[i]  # type: ignore
                    for r in results
                ]
            )
            last.y_events[i] = np.concatenate(
                [
                    np.atleast_2d(
                        r.y_events[i]  # type: ignore
                    )
                    for r in results
                ],
                axis=-1,
            )

    return last
