from dataclasses import dataclass
from typing import Callable

from numpy.typing import NDArray

from .typing import Condition


@dataclass(kw_only=True)
class Change:
    event: Condition
    change: Callable[[float, NDArray], NDArray]
    direction: float = 0.0
