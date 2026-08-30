"""Lightweight terminal progress reporting for long-running CLI commands."""

from __future__ import annotations

import shutil
import sys
import time
from types import TracebackType
from typing import TextIO


class ProgressReporter:
    """Renders an in-place terminal progress bar with an elapsed/ETA estimate.

    Silently no-ops when the target stream is not a TTY (e.g. output is
    redirected to a file, piped through ``tee``, or captured in CI), so
    non-interactive logs are never polluted with carriage-return updates.
    """

    def __init__(
        self,
        total: int,
        *,
        label: str = "",
        stream: TextIO | None = None,
        bar_width: int = 30,
    ) -> None:
        self._total = max(total, 1)
        self._label = label
        self._stream = stream if stream is not None else sys.stderr
        self._bar_width = bar_width
        self._completed = 0
        self._start = time.monotonic()
        self._enabled = bool(getattr(self._stream, "isatty", lambda: False)())
        self._closed = False

    def update(self, completed: int | None = None, *, suffix: str = "") -> None:
        """Advance the bar to ``completed`` steps, or by one step if omitted."""
        self._completed = completed if completed is not None else self._completed + 1
        self._render(suffix)

    def _render(self, suffix: str) -> None:
        if not self._enabled:
            return
        fraction = min(self._completed / self._total, 1.0)
        filled = int(self._bar_width * fraction)
        bar = "#" * filled + "-" * (self._bar_width - filled)
        eta = self._format_eta(fraction, time.monotonic() - self._start)
        prefix = f"{self._label} " if self._label else ""
        line = f"\r{prefix}[{bar}] {self._completed}/{self._total} ({fraction:.0%}) {eta}"
        if suffix:
            line += f" {suffix}"
        width = shutil.get_terminal_size(fallback=(80, 24)).columns
        self._stream.write(line[: max(width, 0)].ljust(width))
        self._stream.flush()

    @staticmethod
    def _format_eta(fraction: float, elapsed: float) -> str:
        if fraction <= 0:
            return "ETA --:--"
        if fraction >= 1:
            return "done"
        remaining = elapsed * (1 - fraction) / fraction
        minutes, seconds = divmod(int(remaining), 60)
        return f"ETA {minutes:02d}:{seconds:02d}"

    def close(self) -> None:
        """Finish the bar, moving the cursor to a fresh line. Safe to call once."""
        if self._enabled and not self._closed:
            self._stream.write("\n")
            self._stream.flush()
        self._closed = True

    def __enter__(self) -> ProgressReporter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
