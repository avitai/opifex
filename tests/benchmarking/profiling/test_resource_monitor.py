"""The ResourceMonitor opifex's profiling package serves: a window ends with a reading."""

import threading
import time
import types

import psutil
import pytest

from opifex.benchmarking.profiling import ResourceMonitor


class _ProcessHolding:
    """A process whose resident memory reads ``first`` MB once, then ``then`` MB on every read."""

    def __init__(self, first: float, *, then: float) -> None:
        self._readings = iter([first])
        self._then = then

    def cpu_percent(self) -> float:
        return 0.0

    def memory_info(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(rss=next(self._readings, self._then) * 1024 * 1024)


def _wait_for_a_sample(monitor: ResourceMonitor, timeout_sec: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not monitor.samples:
        assert time.monotonic() < deadline, "no sample within the timeout"
        threading.Event().wait(0.01)


def test_peak_growth_and_duration_reach_the_end_of_the_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The process holds 100 MB at the one sample and 300 MB when the block ends.

    A sampling interval far longer than the block leaves exactly one sample, so only the reading
    that closes the window can give the 300 MB peak and the 200 MB growth.
    """
    monkeypatch.setattr(psutil, "Process", lambda: _ProcessHolding(100.0, then=300.0))

    with ResourceMonitor(sample_interval_sec=60.0) as monitor:
        _wait_for_a_sample(monitor)

    summary = monitor.summary
    assert summary.num_samples == 2
    assert summary.peak_rss_mb == 300.0
    assert summary.memory_growth_mb == 200.0
    assert summary.duration_sec > 0.0
