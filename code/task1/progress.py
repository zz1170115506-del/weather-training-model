"""Console-only fold progress; never changes a model, data or random state."""
from __future__ import annotations

from contextlib import contextmanager
import math
import sys
from threading import Event, Thread
from time import perf_counter


class _FoldProgress:
    def __init__(self, model_name, fold, fold_count, fit_rows, validation_rows,
                 interval_seconds, stream, clock):
        if (isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int, float))
                or not math.isfinite(interval_seconds) or interval_seconds <= 0):
            raise ValueError("Progress interval_seconds must be finite and positive.")
        self.prefix = f"[CV] model={model_name} fold={fold}/{fold_count}"
        self.interval_seconds = interval_seconds
        self.stream = stream
        self.clock = clock
        self.started = clock()
        self.stopped = Event()
        self.thread = Thread(target=self._heartbeat, name="task1-fold-heartbeat", daemon=True)
        self.fit_rows = fit_rows
        self.validation_rows = validation_rows

    def elapsed_seconds(self):
        return max(0.0, self.clock() - self.started)

    def _write(self, message):
        try:
            print(message, file=self.stream, flush=True)
        except (OSError, ValueError):
            # A redirected/closed console must not change fitting or prediction.
            pass

    def start(self):
        self._write(f"{self.prefix} 开始计算 fit_rows={self.fit_rows} valid_rows={self.validation_rows}")
        self.thread.start()

    def _heartbeat(self):
        while not self.stopped.wait(self.interval_seconds):
            self._write(f"{self.prefix} 仍在计算 elapsed={self.elapsed_seconds():.1f}s")

    def stop(self):
        self.stopped.set()
        self.thread.join()


@contextmanager
def fold_progress(model_name, fold, fold_count, fit_rows, validation_rows, *,
                  interval_seconds=30.0, stream=None, clock=None):
    """Print a start message and periodic elapsed time until this fold exits.

    ``fold`` is one-based. A daemon thread waits on an Event, so finishing or
    raising stops it immediately instead of waiting for the next heartbeat.
    There is no percentage, CPU estimate or predicted completion time. The
    optional interval/stream/clock support short deterministic synthetic tests.
    The yielded reporter exposes elapsed_seconds for the completion log only.
    """
    progress = _FoldProgress(model_name, fold, fold_count, fit_rows, validation_rows,
                             interval_seconds, sys.stdout if stream is None else stream,
                             perf_counter if clock is None else clock)
    progress.start()
    try:
        yield progress
    finally:
        progress.stop()
