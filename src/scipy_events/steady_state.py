from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .core import EventWithSolver


@dataclass(kw_only=True)
class SmallDerivatives(EventWithSolver):
    terminal: bool | int = True
    direction: float = 0.0
    atol: float | NDArray | None = None
    "Absolute tolerance. Uses solver tolerance by default."
    rtol: float | NDArray | None = None
    "Relative tolerance. Uses solver tolerance by default."

    def __call__(self, t: float, y: NDArray, *args) -> float:
        atol = self.atol if self.atol is not None else self.solver.atol
        rtol = self.rtol if self.rtol is not None else self.solver.rtol

        dy = np.abs(self.solver.f)
        if np.all((dy < atol) | (dy < rtol * np.abs(y))):
            return 0
        return np.nan
