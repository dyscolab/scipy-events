from dataclasses import dataclass
from typing import Callable, Iterable

from numpy.typing import NDArray

from .typing import Condition


@dataclass(kw_only=True)
class Change:
    event: Condition
    change: Callable[[float, NDArray], NDArray]
    direction: float = 0.0


@dataclass(kw_only=True)
class ChangeAt:
    times: Iterable[float]
    change: Callable[[float, NDArray], NDArray]
    direction: float = 0.0
