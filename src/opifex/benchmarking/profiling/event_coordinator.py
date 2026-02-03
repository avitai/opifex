"""
Event Coordinator for JAX Profiling Harness.

Coordinates timing and events across multiple profilers to ensure consistent
measurements and prevent interference between profiling components.
"""

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax


@dataclass
class ProfilingEvent:
    """Represents a profiling event with timing information."""

    timestamp: float
    event_type: str
    profiler_id: str
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None


class ProfilingTimeline:
    """Thread-safe timeline for profiling events."""

    def __init__(self):
        self._events: list[ProfilingEvent] = []
        self._lock = threading.Lock()
        self._start_time: float | None = None

    def start_timeline(self):
        """Start the profiling timeline."""
        with self._lock:
            self._start_time = time.time()
            self._events.clear()

    def add_event(
        self,
        event_type: str,
        profiler_id: str,
        data: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ):
        """Add an event to the timeline."""
        with self._lock:
            if self._start_time is None:
                self._start_time = time.time()

            timestamp = time.time() - self._start_time
            event = ProfilingEvent(
                timestamp=timestamp,
                event_type=event_type,
                profiler_id=profiler_id,
                data=data or {},
                duration_ms=duration_ms,
            )
            self._events.append(event)

    def get_events(self, profiler_id: str | None = None) -> list[ProfilingEvent]:
        """Get events, optionally filtered by profiler ID."""
        with self._lock:
            if profiler_id is None:
                return self._events.copy()
            return [e for e in self._events if e.profiler_id == profiler_id]

    def get_timeline_duration(self) -> float:
        """Get total timeline duration in seconds."""
        with self._lock:
            if not self._events:
                return 0.0
            return max(e.timestamp for e in self._events)


class EventCoordinator:
    """Coordinates profiling events and timing across multiple profilers."""

    def __init__(self):
        self.timeline = ProfilingTimeline()
        self.active_profilers: set[str] = set()
        self._profiling_active = False
        self._jax_trace_dir: str | None = None

    def register_profiler(self, profiler_id: str):
        """Register a profiler with the coordinator."""
        self.active_profilers.add(profiler_id)

    def unregister_profiler(self, profiler_id: str):
        """Unregister a profiler from the coordinator."""
        self.active_profilers.discard(profiler_id)

    @contextmanager
    def profiling_session(
        self, enable_jax_profiler: bool = True, trace_dir: str | None = None
    ):
        """Context manager for coordinated profiling session."""
        if trace_dir is None:
            import tempfile

            trace_dir = str(Path(tempfile.gettempdir()) / "jax_trace")

        if self._profiling_active:
            # Nested session - just yield without starting new session
            yield self
            return

        self._profiling_active = True
        self._jax_trace_dir = trace_dir if enable_jax_profiler else None

        try:
            # Start timeline
            self.timeline.start_timeline()

            # Start JAX profiler if requested
            if enable_jax_profiler:
                try:
                    jax.profiler.start_trace(trace_dir)
                    self.timeline.add_event(
                        "jax_profiler_start", "coordinator", {"trace_dir": trace_dir}
                    )
                except Exception as e:
                    # JAX profiler might not be available in all environments
                    self.timeline.add_event(
                        "jax_profiler_error", "coordinator", {"error": str(e)}
                    )

            # Notify profilers of session start
            for profiler_id in self.active_profilers:
                self.timeline.add_event("profiler_session_start", profiler_id)

            yield self

        finally:
            # Stop JAX profiler
            if enable_jax_profiler and self._jax_trace_dir:
                try:
                    jax.profiler.stop_trace()
                    self.timeline.add_event("jax_profiler_stop", "coordinator")
                except Exception:
                    pass  # Ignore errors during cleanup

            # Notify profilers of session end
            for profiler_id in self.active_profilers:
                self.timeline.add_event("profiler_session_end", profiler_id)

            self._profiling_active = False
            self._jax_trace_dir = None

    def add_event(
        self,
        event_type: str,
        profiler_id: str,
        data: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ):
        """Add an event to the coordinated timeline."""
        if not self._profiling_active:
            return  # Silently ignore events outside profiling sessions

        self.timeline.add_event(event_type, profiler_id, data, duration_ms)

    def time_function(
        self,
        func,
        *args,
        profiler_id: str = "unknown",
        operation_name: str = "operation",
        **kwargs,
    ):
        """Time a function execution and record the event."""

        start_time = time.time()
        self.add_event(
            f"{operation_name}_start",
            profiler_id,
            {"args_shapes": [getattr(arg, "shape", None) for arg in args]},
        )

        try:
            result = func(*args, **kwargs)

            # Ensure JAX computation completes
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            elif isinstance(result, (tuple, list)):
                for item in result:
                    if hasattr(item, "block_until_ready"):
                        item.block_until_ready()

            execution_time = time.time() - start_time

            self.add_event(
                f"{operation_name}_complete",
                profiler_id,
                {
                    "execution_time_ms": execution_time * 1000,
                    "result_shape": getattr(result, "shape", None),
                    "success": True,
                },
                duration_ms=execution_time * 1000,
            )

            return result, execution_time

        except Exception as e:
            execution_time = time.time() - start_time

            self.add_event(
                f"{operation_name}_error",
                profiler_id,
                {
                    "execution_time_ms": execution_time * 1000,
                    "error": str(e),
                    "success": False,
                },
                duration_ms=execution_time * 1000,
            )

            raise

    def get_profiling_summary(self) -> dict[str, Any]:
        """Get a summary of the profiling session."""

        events = self.timeline.get_events()
        total_duration = self.timeline.get_timeline_duration()

        # Group events by profiler
        events_by_profiler = defaultdict(list)
        for event in events:
            events_by_profiler[event.profiler_id].append(event)

        # Calculate statistics
        profiler_stats = {}
        for profiler_id, profiler_events in events_by_profiler.items():
            execution_events = [e for e in profiler_events if e.duration_ms is not None]

            if execution_events:
                total_execution_time = sum(e.duration_ms for e in execution_events)
                avg_execution_time = total_execution_time / len(execution_events)
                max_execution_time = max(e.duration_ms for e in execution_events)
                min_execution_time = min(e.duration_ms for e in execution_events)
            else:
                total_execution_time = avg_execution_time = max_execution_time = (
                    min_execution_time
                ) = 0

            profiler_stats[profiler_id] = {
                "total_events": len(profiler_events),
                "execution_events": len(execution_events),
                "total_execution_time_ms": total_execution_time,
                "avg_execution_time_ms": avg_execution_time,
                "max_execution_time_ms": max_execution_time,
                "min_execution_time_ms": min_execution_time,
            }

        return {
            "session_duration_s": total_duration,
            "total_events": len(events),
            "active_profilers": list(self.active_profilers),
            "profiler_statistics": profiler_stats,
            "jax_trace_dir": self._jax_trace_dir,
        }

    def export_timeline(self, output_format: str = "json") -> str:
        """Export timeline in specified format."""

        events = self.timeline.get_events()

        if output_format == "json":
            import json

            timeline_data = {
                "session_duration_s": self.timeline.get_timeline_duration(),
                "events": [
                    {
                        "timestamp": e.timestamp,
                        "event_type": e.event_type,
                        "profiler_id": e.profiler_id,
                        "data": e.data,
                        "duration_ms": e.duration_ms,
                    }
                    for e in events
                ],
            }

            return json.dumps(timeline_data, indent=2)

        if output_format == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # Header
            writer.writerow(
                ["timestamp", "event_type", "profiler_id", "duration_ms", "data"]
            )

            # Events
            for event in events:
                writer.writerow(
                    [
                        event.timestamp,
                        event.event_type,
                        event.profiler_id,
                        event.duration_ms or "",
                        str(event.data),
                    ]
                )

            return output.getvalue()

        raise ValueError(f"Unsupported export format: {format}")


def create_shared_coordinator() -> EventCoordinator:
    """Create a shared event coordinator instance."""
    return EventCoordinator()
