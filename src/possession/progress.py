"""One-line progress for long runs: a bar, step k of n, elapsed and estimated time left.

    bar = Progress("train", total=10)
    for ...:
        ...
        bar.step("J03WMX done")
Prints to stdout with flush, so logs show it live.
"""

import time


class Progress:
    def __init__(self, name: str, total: int, width: int = 20):
        self.name, self.total, self.width = name, max(1, total), width
        self.done, self.start = 0, time.monotonic()
        print(f"[{name}] {'.' * width} 0/{self.total}  starting", flush=True)

    def step(self, note: str = "") -> None:
        self.done += 1
        elapsed = time.monotonic() - self.start
        left = elapsed / self.done * (self.total - self.done)
        filled = int(self.width * self.done / self.total)
        print(f"[{self.name}] {'#' * filled}{'.' * (self.width - filled)} {self.done}/{self.total}"
              f"  {_fmt(elapsed)} elapsed, ~{_fmt(left)} left  {note}", flush=True)


def _fmt(s: float) -> str:
    return f"{int(s // 60)}m{int(s % 60):02d}s" if s >= 60 else f"{s:.0f}s"
