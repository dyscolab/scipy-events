from dataclasses import dataclass
from typing import Iterable

from .typing import Change, Condition


@dataclass(kw_only=True, frozen=True)
class ChangeWhen:
    """Apply change when a given condition happens."""

    condition: Condition
    change: Change
    direction: float = 0.0


@dataclass(kw_only=True, frozen=True)
class ChangeAt:
    """Apply change at the specified times."""

    times: Iterable[float]
    change: Change
    direction: float = 0.0
