from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._core import WithSolver


@dataclass(kw_only=True, frozen=True)
class SmallDerivatives(WithSolver):
    atol: float | NDArray | None = None
    "Absolute tolerance. Uses solver.atol * factor by default."
    rtol: float | NDArray | None = None
    "Relative tolerance. Uses solver.rtol * factor by default."
    factor: float | NDArray = 10.0
    "Factor relative to solver tolerance."

    def __call__(self, t: float, y: NDArray, /, *args) -> float:
        atol = self.atol if self.atol is not None else self.solver.atol * self.factor
        rtol = self.rtol if self.rtol is not None else self.solver.rtol * self.factor

        dy = np.abs(self.solver.f)
        if np.all((dy < atol) | (dy < rtol * np.abs(y))):
            return 0
        return np.nan
