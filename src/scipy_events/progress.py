from typing import Protocol


class ProgressBar(Protocol):
    desc: str
    n: int | float
    total: int | float | None

    def update(self, n: float = 1, /) -> bool | None: ...


class Progress:
    def __init__(self, pbar: ProgressBar, /):
        self.pbar = pbar

    @classmethod
    def from_tqdm(cls, *, total_time: float | None = None):
        from tqdm.auto import tqdm

        return cls(tqdm(total=total_time))

    def __call__(self, t: float, y, *args):
        if self.pbar.total is None:
            self.pbar.desc = f"{t:=.3f}"
            self.pbar.update()
        else:
            self.pbar.update(t - self.pbar.n)
        return 1
